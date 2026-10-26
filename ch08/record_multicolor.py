"""8부 프로젝트 1 — 블록 세 개(빨강·초록·노랑) 장면에서 색을 지정하는 시연 데이터셋 만들기

6-5의 record_demos.py와 같지만 장면에 블록이 세 개 있고, episode 마다 집을 색이 바뀌며
task 문장에 그 색이 들어갑니다. SmolVLA가 문장의 색을 보고 다른 블록을 집는지 시험하는 데 씁니다.

    task = "Pick up the red cube and put it in the blue box."   (red / green / yellow 순환)

실행:
    MUJOCO_GL=egl python ch08/record_multicolor.py --episodes 60
    MUJOCO_GL=egl python ch08/record_multicolor.py --episodes 150 --retry-failed
"""

import argparse
import os

os.environ.setdefault("SVT_LOG", "0")
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ch03.pick_and_place import APPROACH_Z, GRASP_Z, LIFT_Z, PLACE_Z, above, is_in_box  # noqa: E402
from ch06.record_demos import FPS, IMG_H, IMG_W, JOINT_NAMES, RecordingRobot, prepare_root  # noqa: E402

COLORS = ["red", "green", "yellow"]
TASK_FMT = "Pick up the {color} cube and put it in the blue box."


def random_layout(rng, n=3, min_gap=0.07):
    """작업 영역 안에서 서로 7 cm 이상 떨어진 위치 n 개."""
    pts = []
    while len(pts) < n:
        r, th = rng.uniform(0.17, 0.27), rng.uniform(-0.7, 0.7)
        p = np.array([r * np.cos(th), r * np.sin(th)])
        if all(np.linalg.norm(p - q) > min_gap for q in pts):
            pts.append(p)
    return [tuple(map(float, p)) for p in pts]


def make_robot(layout, dataset=None, task=None, top_camera="top", **kw):
    """빨간 블록은 기본 cube, 초록·노랑은 extra_cubes로."""
    red, green, yellow = layout
    return RecordingRobot(dataset=dataset, task=task, top_camera=top_camera, cube_pos=red, box_pos=(0.05, 0.22),
                          extra_cubes=[("green", green), ("yellow", yellow)], **kw)


def scripted_pick_color(robot, color):
    target = robot.cube_pos(color)
    box = robot.box_pos()
    robot.open_gripper(0.5)
    robot.move_to(above(target, APPROACH_Z), seconds=1.2)
    robot.move_to(above(target, GRASP_Z), seconds=0.8)
    robot.hold(0.2)
    robot.close_gripper(0.8)
    robot.move_to(above(target, LIFT_Z), seconds=1.0)
    if robot.cube_pos(color)[2] < 0.06:
        return False
    robot.move_to(above(box, LIFT_Z), seconds=1.5)
    robot.move_to(above(box, PLACE_Z), seconds=0.8)
    robot.hold(0.2)
    robot.open_gripper(0.5)
    robot.move_to(above(box, LIFT_Z), seconds=0.8)
    robot.hold(0.3)
    return is_in_box(robot.cube_pos(color), box)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--root", default="outputs/datasets/so101_multicolor_sim")
    ap.add_argument("--repo-id", default="physicalai/so101_multicolor_sim")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retry-failed", action="store_true", help="성공한 시연이 --episodes개가 될 때까지 실패를 재시도 (최대 10배 시도)")
    ap.add_argument("--top-camera", default="top", help="위 카메라 이름 (top 또는 top_zoom)")
    ap.add_argument("--overwrite", action="store_true", help="--root 폴더가 이미 있으면 지우고 새로 기록")
    args = ap.parse_args()
    if args.episodes < 1:
        ap.error("episodes는 양수여야 합니다.")
    prepare_root(args.root, args.overwrite)

    from lerobot.datasets import LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "observation.images.top": {"dtype": "video", "shape": (IMG_H, IMG_W, 3), "names": ["height", "width", "channels"]},
        "observation.images.wrist": {"dtype": "video", "shape": (IMG_H, IMG_W, 3), "names": ["height", "width", "channels"]},
    }
    dataset = LeRobotDataset.create(repo_id=args.repo_id, fps=FPS, features=features,
                                    root=args.root, robot_type="so101_sim", use_videos=True)
    rng = np.random.default_rng(args.seed)
    t0, saved, failed = time.time(), {c: 0 for c in COLORS}, 0
    max_attempts = args.episodes * (10 if args.retry_failed else 1)
    for ep in range(max_attempts):
        if args.retry_failed and sum(saved.values()) >= args.episodes:
            break
        color = COLORS[(sum(saved.values()) if args.retry_failed else ep) % 3]
        layout = random_layout(rng)
        robot = make_robot(layout, dataset=dataset, task=TASK_FMT.format(color=color), top_camera=args.top_camera)
        ok = scripted_pick_color(robot, color)
        n = robot.n_recorded
        robot.close()
        if ok:
            dataset.save_episode()
            saved[color] += 1
        else:
            dataset.clear_episode_buffer()
            failed += 1
        print(f"attempt {ep + 1:3d}/{max_attempts}  {color:<6s} {n:3d} frames  {'저장' if ok else '실패→버림'}  저장 {sum(saved.values())}/{args.episodes}  ({time.time() - t0:5.0f} s)", flush=True)
    dataset.finalize()
    print(f"\n완료: 저장 {saved}, 실패 {failed}, 총 {time.time() - t0:.0f} 초, 프레임 {dataset.meta.total_frames}")
    if args.retry_failed and sum(saved.values()) < args.episodes:
        raise RuntimeError(f"최대 시도 횟수 내에 성공 시연 {args.episodes}개를 채우지 못했습니다.")


if __name__ == "__main__":
    main()
