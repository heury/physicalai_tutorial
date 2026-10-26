"""Train two Hub datasets under equal ACT compute budgets and paired simulation evaluation.

    .venv/bin/python ch06/compare_act_datasets.py
    .venv/bin/python ch06/compare_act_datasets.py --prepare-only

The final checkpoint is evaluated, without selecting it on test outcomes.
Existing incomplete training directories are preserved and require manual resume.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "so101_pickplace_sim": "9b1c5189e3858e9254fc5e48259a035d22da497a",
    "so101_pickplace_sim_150": "b4fb342927bb0ee7dcbe4b90472a0c257ce4f5de",
}


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def run_logged(command, log_path, env):
    print("Running:", " ".join(command), flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        if process.wait():
            raise RuntimeError(f"Command failed ({process.returncode}); see {log_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/ch06/act_comparison")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_episodes) < 1:
        parser.error("steps, batch-size, eval-episodes must be positive")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HF_HOME": str(ROOT / "outputs/cache/huggingface"),
           "TORCH_HOME": str(ROOT / "outputs/cache/torch"),
           "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
           "MUJOCO_GL": "egl", "TOKENIZERS_PARALLELISM": "false"}
    for key in ("HF_HOME", "TORCH_HOME", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = env[key]
    import torch
    import lerobot
    from huggingface_hub import snapshot_download
    from lerobot.datasets import LeRobotDataset

    plan = {"datasets": DATASETS, "steps": args.steps, "batch_size": args.batch_size,
            "training_seed": args.seed, "eval_seed": args.eval_seed,
            "eval_episodes": args.eval_episodes, "chunk_size": 50, "n_action_steps": 50,
            "use_amp": False, "torch": torch.__version__, "lerobot": lerobot.__version__,
            "selection": "final checkpoint only", "budget": "equal optimizer steps and batch size",
            "evaluation": "paired fresh simulation placements, no test-driven tuning"}
    plan_path = out / "experiment.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("Experiment differs from existing experiment.json; use a new --out-dir")
    save_json(plan_path, plan)
    inventory = []
    for name, revision in DATASETS.items():
        repo = f"caspar/{name}"
        root = ROOT / "outputs/datasets/caspar" / name
        snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=root)
        info = json.loads((root / "meta/info.json").read_text())
        dataset = LeRobotDataset(repo, root=root, revision=revision, video_backend="pyav")
        sample = dataset[0]
        for key in ("observation.images.top", "observation.images.wrist"):
            assert sample[key].shape == (3, 240, 320), (key, sample[key].shape)
            assert torch.isfinite(sample[key]).all()
        assert sample["action"].shape == (6,) and sample["observation.state"].shape == (6,)
        files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in root.rglob("*") if p.is_file() and ".cache" not in p.relative_to(root).parts}
        inventory.append({"repo_id": repo, "revision": revision, "root": str(root),
                          "episodes": info["total_episodes"], "frames": info["total_frames"],
                          "fps": info["fps"], "features": info["features"],
                          "sample_decoded": True, "sha256": files})
        del dataset
    assert inventory[0]["fps"] == inventory[1]["fps"] == 20
    assert inventory[0]["features"] == inventory[1]["features"]
    save_json(out / "data_inventory.json", inventory)
    if args.prepare_only:
        print("Datasets downloaded, decoded, and hashed. Training not requested.")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this execution environment")
    save_json(out / "hardware.json", {"device": torch.cuda.get_device_name(0),
              "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
              "torch": torch.__version__, "cuda": torch.version.cuda})
    subprocess.run(["nvidia-smi"], check=True)
    summaries = []
    for item in inventory:
        name = item["repo_id"].split("/")[-1]
        train_root = out / name
        checkpoint = train_root / "checkpoints/last/pretrained_model"
        record = out / f"{name}_training.json"
        if not checkpoint.is_dir():
            command = [sys.executable, "-m", "lerobot.scripts.lerobot_train",
                       f"--dataset.repo_id={item['repo_id']}", f"--dataset.root={item['root']}",
                       f"--dataset.revision={item['revision']}", "--dataset.video_backend=pyav",
                       "--policy.type=act", "--policy.chunk_size=50", "--policy.n_action_steps=50",
                       "--policy.device=cuda", "--policy.use_amp=false", "--policy.push_to_hub=false",
                       f"--output_dir={train_root}", f"--job_name=act_{name}_comparison",
                       f"--steps={args.steps}", f"--batch_size={args.batch_size}",
                       f"--seed={args.seed}", "--num_workers=4", "--env_eval_freq=0",
                       f"--save_freq={max(1, args.steps // 4)}", "--log_freq=100", "--wandb.enable=false"]
            started = time.monotonic()
            run_logged(command, out / f"{name}_train.log", env)
            save_json(record, {"command": command, "seconds": time.monotonic() - started})
        if not (checkpoint / "model.safetensors").is_file():
            raise RuntimeError(f"Missing final model: {checkpoint}")
        results_path = out / f"{name}_eval.json"
        if not results_path.exists():
            run_logged([sys.executable, "ch06/eval_act.py", "--checkpoint", str(checkpoint),
                        "--episodes", str(args.eval_episodes), "--seed", str(args.eval_seed),
                        "--results-json", str(results_path), "--video", "--out", str(out / f"{name}.mp4")],
                       out / f"{name}_eval.log", env)
        result = json.loads(results_path.read_text())
        assert result["episode_count"] == args.eval_episodes and result["seed"] == args.eval_seed
        summaries.append({"dataset": item["repo_id"], "training_episodes": item["episodes"],
                          "training_frames": item["frames"],
                          "approx_frame_passes": args.steps * args.batch_size / item["frames"],
                          "checkpoint": str(checkpoint), "evaluation": result})
    first, second = (s["evaluation"] for s in summaries)
    assert [e["cube_xy"] for e in first["episodes"]] == [e["cube_xy"] for e in second["episodes"]]
    paired = {"both_succeeded": 0, "first_only": 0, "second_only": 0, "both_failed": 0}
    for a, b in zip(first["episodes"], second["episodes"], strict=True):
        key = ("both_succeeded" if a["success"] and b["success"] else
               "first_only" if a["success"] else "second_only" if b["success"] else "both_failed")
        paired[key] += 1
    save_json(out / "comparison.json", {"experiment": plan, "results": summaries, "paired": paired,
              "success_rate_difference_second_minus_first": second["success_rate"] - first["success_rate"],
              "limitation": "One training seed; equal compute, not equal epochs. Training losses use different data normalizations."})
    print(json.dumps({"success_rates": {s["dataset"]: s["evaluation"]["success_rate"] for s in summaries},
                      "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
