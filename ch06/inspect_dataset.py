"""6부 실습 — LeRobot 데이터셋 열어보기 / 뜯어보기

1) Hugging Face Hub의 공개 데이터셋 메타데이터를 읽고 (다운로드 거의 없음)
2) 로컬(또는 Hub) 데이터셋에서 프레임 하나를 꺼내 구조를 출력하고
3) 한 episode의 관절각·행동을 그림으로 저장합니다.

실행:
    python ch06/inspect_dataset.py                                   # Hub 공개 데이터셋
    python ch06/inspect_dataset.py --root outputs/datasets/so101_pickplace_sim --repo-id physicalai/so101_pickplace_sim
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("SVT_LOG", "0")          # 영상 인코더(SVT-AV1)의 장황한 로그 끄기

import numpy as np

OUT = "outputs/ch06"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="lerobot/svla_so101_pickplace")
    ap.add_argument("--root", default=None, help="로컬 데이터셋 폴더 (없으면 Hub에서 받음)")
    ap.add_argument("--episode", type=int, default=0)
    args = ap.parse_args()
    if args.root and not (Path(args.root) / "meta" / "info.json").is_file():
        ap.error(
            f"로컬 데이터셋 메타데이터가 없습니다: {Path(args.root) / 'meta' / 'info.json'}. "
            "저장소 루트에서 실행 중인지 확인하고, 먼저 ch06/record_demos.py로 "
            "같은 --root와 --repo-id에 시연 데이터를 생성하세요."
        )
    os.makedirs(OUT, exist_ok=True)

    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

    # 1) 메타데이터
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    print("=== 메타데이터 ===")
    print(f"repo_id       : {args.repo_id}")
    print(f"robot_type    : {meta.robot_type}")
    print(f"fps           : {meta.fps}")
    print(f"episodes      : {meta.total_episodes}")
    print(f"frames        : {meta.total_frames}  (episode당 평균 {meta.total_frames / meta.total_episodes:.1f})")
    print(f"camera keys   : {meta.camera_keys}")
    print(f"tasks         : {list(meta.tasks.index) if hasattr(meta.tasks, 'index') else meta.tasks}")
    print("features:")
    for k, v in meta.features.items():
        print(f"  {k:<28s} dtype={v['dtype']:<8s} shape={tuple(v['shape'])}")

    # 2) episode 하나 로드
    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=[args.episode])
    print(f"\n=== episode {args.episode} ===")
    print(f"frames: {ds.num_frames}")
    sample = ds[0]
    print("frame 0의 내용:")
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(f"  {k:<28s} {str(tuple(v.shape)):<16s} {v.dtype}")
        else:
            print(f"  {k:<28s} {v!r}")

    # 3) 관절각·행동 그림, 이미지 저장
    states = np.stack([ds[i]["observation.state"].numpy() for i in range(ds.num_frames)])
    actions = np.stack([ds[i]["action"].numpy() for i in range(ds.num_frames)])
    t = np.arange(ds.num_frames) / meta.fps

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = states.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(7, 1.4 * n), sharex=True)
    for j in range(n):
        axes[j].plot(t, states[:, j], label="state")
        axes[j].plot(t, actions[:, j], "--", label="action")
        axes[j].set_ylabel(f"joint {j}", fontsize=8)
        axes[j].grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    tag = "local" if args.root else "hub"
    fig.savefig(f"{OUT}/episode_{tag}_state_action.png", dpi=120)

    import cv2
    for k in meta.camera_keys:
        img = sample[k]                                   # (C, H, W) float 0~1
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cv2.imwrite(f"{OUT}/frame0_{tag}_{k.split('.')[-1]}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"\n저장: {OUT}/episode_{tag}_state_action.png, frame0_{tag}_*.png")
    print(f"state 범위: min {np.round(states.min(0), 3)}  max {np.round(states.max(0), 3)}")


if __name__ == "__main__":
    main()
