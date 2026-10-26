"""6부 프로젝트 — 학습한 ACT에게 Pick and Place 시키기

학습된 ACT 체크포인트를 불러와 시뮬레이션에서 여러 episode를 실행하고 성공률을 잽니다.
    Camera + 관절각 → ACT → 행동(목표 각도 6개) → 로봇
매 1/FPS 초마다 관측을 새로 찍어 Policy에 넣습니다 (닫힌 고리).

실행:
    MUJOCO_GL=egl python ch06/eval_act.py --checkpoint outputs/train/act_so101/checkpoints/last/pretrained_model
    MUJOCO_GL=egl python ch06/eval_act.py --checkpoint ... --episodes 20 --video
"""

import argparse
import os

os.environ.setdefault("SVT_LOG", "0")          # 영상 인코더(SVT-AV1)의 장황한 로그 끄기
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ch03.pick_and_place import is_in_box  # noqa: E402
from ch06.record_demos import FPS, IMG_H, IMG_W, TASK, random_cube  # noqa: E402
from common.robot import SO101Sim  # noqa: E402
from common.vision import SimCamera  # noqa: E402


DEFAULT_KEYS = {"top": "observation.images.top", "wrist": "observation.images.wrist"}


def run_episode(policy, preprocess, postprocess, device, cube, max_seconds=12.0, video=False, out=None,
                image_keys=DEFAULT_KEYS, task=TASK, box=(0.05, 0.22), scene_kw=None, top_camera="top"):
    """Policy로 episode 하나를 실행. image_keys는 카메라 이름 → 관측 키 (SmolVLA는 camera1/2를 쓴다)."""
    robot = SO101Sim(cube_pos=cube, box_pos=box, render=video, camera="fixed", **(scene_kw or {}))
    cams = {"top": SimCamera(robot.model, top_camera, IMG_W, IMG_H),
            "wrist": SimCamera(robot.model, "wrist_cam", IMG_W, IMG_H)}
    steps_per_frame = int(round(1 / (FPS * robot.model.opt.timestep)))
    policy.reset()
    n_frames = int(max_seconds * FPS)
    success, t_success = False, None
    for i in range(n_frames):
        state = torch.from_numpy(robot.data.qpos[:6].astype(np.float32))
        obs = {"observation.state": state.unsqueeze(0).to(device), "task": [task]}
        for k, cam in cams.items():
            rgb, _ = cam.capture(robot.data)
            img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0      # (C,H,W) 0~1
            obs[image_keys[k]] = img.unsqueeze(0).to(device)
        obs = preprocess(obs)
        with torch.inference_mode():
            action = policy.select_action(obs)
        action = postprocess(action)
        robot.data.ctrl[:6] = action.squeeze(0).cpu().numpy()
        robot.step(steps_per_frame)
        if not success and is_in_box(robot.cube_pos(), robot.box_pos()):
            success, t_success = True, (i + 1) / FPS
            robot.hold(0.5)
            break
    for cam in cams.values():
        cam.close()
    if video and out:
        robot.save_video(out)
    robot.close()
    return success, t_success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-root", default=None, help="정규화 통계를 데이터셋에서 다시 읽고 싶을 때만 지정 (기본: 체크포인트에 저장된 통계 사용)")
    ap.add_argument("--repo-id", default="physicalai/so101_pickplace_sim")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--results-json", default=None, help="회차별 결과와 성공률을 JSON으로 저장")
    ap.add_argument("--max-seconds", type=float, default=12.0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", default="outputs/ch06_act_eval.mp4")
    ap.add_argument("--cube", type=float, nargs=2, default=None, help="블록 위치 고정 (8-2)")
    ap.add_argument("--scene-kw", default=None, help='장면 변화 JSON, 예: {"light_pos": [0.5, 0.5, 1.0]} (8-3)')
    ap.add_argument("--n-action-steps", type=int, default=None, help="chunk 중 실제 실행할 스텝 수 (기본: 학습 설정값 50)")
    ap.add_argument("--ensemble", type=float, default=None, help="Temporal Ensembling 계수 (예: 0.01). 매 스텝 재추론 + 가중 평균")
    ap.add_argument("--top-camera", default="top", help="데이터셋을 기록할 때 쓴 위 카메라 이름")
    args = ap.parse_args()
    if args.episodes < 1 or args.max_seconds <= 0:
        ap.error("episodes와 max-seconds는 양수여야 합니다.")
    import json
    scene_kw = json.loads(args.scene_kw) if args.scene_kw else None

    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act import ACTPolicy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    policy = ACTPolicy.from_pretrained(args.checkpoint).to(device).eval()
    # 체크포인트 폴더에 정규화 통계(전처리기·후처리기)가 함께 저장되어 있으므로 보통은 그대로 쓴다
    stats = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root).stats if args.dataset_root else None
    preprocess, postprocess = make_pre_post_processors(
        policy.config, args.checkpoint, dataset_stats=stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}})
    if args.ensemble is not None:                       # 6-6의 Temporal Ensembling을 실행 시점에 켠다
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        policy.config.temporal_ensemble_coeff = args.ensemble
        policy.config.n_action_steps = 1
        policy.temporal_ensembler = ACTTemporalEnsembler(args.ensemble, policy.config.chunk_size)
    elif args.n_action_steps is not None:
        policy.config.n_action_steps = args.n_action_steps
    print(f"policy: ACT ({sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M params), device {device}")
    print(f"chunk_size {policy.config.chunk_size}, n_action_steps {policy.config.n_action_steps}, "
          f"temporal_ensemble_coeff {policy.config.temporal_ensemble_coeff}")

    rng = np.random.default_rng(args.seed)
    results, records, t0 = [], [], time.time()
    for i in range(args.episodes):
        cube = tuple(args.cube) if args.cube else random_cube(rng)
        ok, ts = run_episode(policy, preprocess, postprocess, device, cube,
                             max_seconds=args.max_seconds, video=(args.video and i == 0),
                             out=args.out, scene_kw=scene_kw, top_camera=args.top_camera)
        results.append(ok)
        records.append({"episode": i + 1, "cube_xy": list(cube), "success": bool(ok),
                        "time_to_success_seconds": ts})
        print(f"episode {i + 1:2d}  블록 ({cube[0]:+.3f}, {cube[1]:+.3f})  → "
              f"{'성공 (%.1f s)' % ts if ok else '실패'}   ({time.time() - t0:4.0f} s)", flush=True)
    print(f"\n성공률 {sum(results)}/{len(results)} = {100 * sum(results) / len(results):.0f}%")
    if args.results_json:
        from pathlib import Path
        path = Path(args.results_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "checkpoint": args.checkpoint, "device": str(device), "seed": args.seed,
            "episodes": records, "successes": sum(results), "episode_count": len(results),
            "success_rate": sum(results) / len(results), "elapsed_seconds": time.time() - t0,
            "max_seconds": args.max_seconds, "top_camera": args.top_camera,
            "scene_kw": scene_kw or {}, "task": TASK,
            "chunk_size": policy.config.chunk_size, "n_action_steps": policy.config.n_action_steps,
            "temporal_ensemble_coeff": policy.config.temporal_ensemble_coeff,
            "success_definition": "ch03.pick_and_place.is_in_box; first entry within episode",
        }, indent=2))


if __name__ == "__main__":
    main()
