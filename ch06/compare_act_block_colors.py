"""Evaluate the two saved ACT models on blue cubes, paired with existing red-cube results.

    .venv/bin/python ch06/compare_act_block_colors.py
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ch06.compare_act_datasets import run_logged, save_json

BLUE = [0.1, 0.1, 0.9, 1.0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "outputs/ch06/act_comparison")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/ch06/act_color_comparison")
    args = parser.parse_args()
    baseline = json.loads((args.baseline_dir / "comparison.json").read_text())
    verified = json.loads((args.baseline_dir / "completion_verified.json").read_text())
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HF_HOME": str(ROOT / "outputs/cache/huggingface"),
           "TORCH_HOME": str(ROOT / "outputs/cache/torch"), "HF_HUB_OFFLINE": "1",
           "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1", "MUJOCO_GL": "egl"}
    rows = []
    for original in baseline["results"]:
        repo = original["dataset"]
        name = repo.split("/")[-1]
        red = original["evaluation"]
        checkpoint = Path(original["checkpoint"])
        checksum = hashlib.sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest()
        assert checksum == verified["model_sha256"][repo], "Checkpoint differs from original evaluation"
        result_path = out / f"{name}_blue_eval.json"
        if not result_path.exists():
            run_logged([sys.executable, "ch06/eval_act.py", "--checkpoint", str(checkpoint),
                        "--seed", str(red["seed"]), "--episodes", str(red["episode_count"]),
                        "--max-seconds", str(red["max_seconds"]), "--top-camera", red["top_camera"],
                        "--scene-kw", json.dumps({"cube_rgba": BLUE}), "--results-json", str(result_path),
                        "--video", "--out", str(out / f"{name}_blue.mp4")],
                       out / f"{name}_blue_eval.log", env)
        blue = json.loads(result_path.read_text())
        assert blue["scene_kw"] == {"cube_rgba": BLUE}
        assert blue["seed"] == red["seed"] and blue["episode_count"] == red["episode_count"]
        assert Path(blue["checkpoint"]).resolve() == checkpoint.resolve()
        for key in ("max_seconds", "top_camera", "chunk_size", "n_action_steps", "temporal_ensemble_coeff"):
            assert blue[key] == red[key], key
        assert [e["cube_xy"] for e in red["episodes"]] == [e["cube_xy"] for e in blue["episodes"]]
        paired = {"both_succeeded": 0, "red_only": 0, "blue_only": 0, "both_failed": 0}
        for r, b in zip(red["episodes"], blue["episodes"], strict=True):
            key = ("both_succeeded" if r["success"] and b["success"] else "red_only" if r["success"]
                   else "blue_only" if b["success"] else "both_failed")
            paired[key] += 1
        rows.append({"dataset": repo, "checkpoint": str(checkpoint), "model_sha256": checksum,
                     "red_successes": red["successes"], "blue_successes": blue["successes"],
                     "episode_count": blue["episode_count"], "red_success_rate": red["success_rate"],
                     "blue_success_rate": blue["success_rate"],
                     "change_percentage_points": 100 * (blue["success_rate"] - red["success_rate"]),
                     "paired": paired, "blue_evaluation": blue})
        assert hashlib.sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest() == checksum
    assert ([e["cube_xy"] for e in rows[0]["blue_evaluation"]["episodes"]] ==
            [e["cube_xy"] for e in rows[1]["blue_evaluation"]["episodes"]])
    save_json(out / "comparison.json", {"cube_rgba": BLUE, "baseline_dir": str(args.baseline_dir),
              "retraining": False, "results": rows, "paired_positions_verified": True,
              "limitation": "One training seed and 50 paired positions; only the main cube appearance changes."})
    print(json.dumps([{k: v for k, v in r.items() if k != "blue_evaluation"} for r in rows], indent=2))


if __name__ == "__main__":
    main()
