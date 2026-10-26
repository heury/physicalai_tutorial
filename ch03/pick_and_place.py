"""3부 프로젝트 — Rule 기반 Pick and Place

빨간 블록을 집어서 파란 상자 안에 넣습니다. AI는 없습니다.
Approach → Grasp → Lift → Move → Place의 다섯 단계를 상태 기계(state machine)로 짭니다.
블록과 상자의 위치는 시뮬레이터에서 직접 읽습니다 (4부에서 카메라로 바꿉니다).

실행:
    MUJOCO_GL=egl python ch03/pick_and_place.py
    python ch03/pick_and_place.py --no-video --cube 0.20 0.08 --box 0.05 0.22
    python ch03/pick_and_place.py --no-video --trials 10      # 블록 위치를 무작위로 바꿔 10회
    MUJOCO_GL=egl python ch03/pick_and_place.py --trials 10   # 회차별 영상 + 전체 영상
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.robot import SO101Sim  # noqa: E402

np.set_printoptions(precision=3, suppress=True)

# 손끝(두 손가락 사이 기준점)의 목표 높이. 모두 바닥 기준 절대 높이(m)
APPROACH_Z = 0.08     # 블록 위 8 cm에서 접근
GRASP_Z = 0.02        # 블록 중심(0.015)보다 5 mm 위. 손가락이 블록 옆면을 감싸는 높이
LIFT_Z = 0.12         # 집은 뒤 들어 올리는 높이
PLACE_Z = 0.06        # 상자 위에서 놓는 높이


def above(xy, z):
    """(x, y) 위 높이 z의 목표점."""
    return np.array([xy[0], xy[1], z])


def is_in_box(cube_pos, box_pos, half=0.05):
    dx, dy = abs(cube_pos[0] - box_pos[0]), abs(cube_pos[1] - box_pos[1])
    return dx < half and dy < half and cube_pos[2] < 0.05


def pick_and_place(robot, verbose=True):
    """다섯 단계를 차례로 실행하고 성공 여부를 돌려준다."""
    cube = robot.cube_pos()          # 시뮬레이터에서 직접 읽는 State (4부에서 Observation으로 바뀜)
    box = robot.box_pos()
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)

    # 1. Approach: 집게를 벌리고 블록 위 8 cm로
    log("1 Approach")
    robot.open_gripper(0.5)
    _, err = robot.move_to(above(cube, APPROACH_Z), seconds=1.2)
    log(f"   IK 오차 {err:.4f}  손끝 {robot.site_pose()[0]}")

    # 2. Grasp: 수직으로 내려가 집게를 오므린다
    log("2 Grasp")
    robot.move_to(above(cube, GRASP_Z), seconds=0.8)
    robot.hold(0.2)
    robot.close_gripper(0.8)
    log(f"   집게 각도 {robot.gripper_opening():+.3f} rad (블록에 막혀 완전히 닫히지 않으면 성공 신호)")

    # 3. Lift: 들어 올린다
    log("3 Lift")
    robot.move_to(above(cube, LIFT_Z), seconds=1.0)
    robot.hold(0.2)
    lifted = robot.cube_pos()[2] > 0.06
    log(f"   블록 높이 {robot.cube_pos()[2]:.3f} m → {'집었음' if lifted else '놓쳤음'}")
    if not lifted:
        return False

    # 4. Move: 상자 위로 (높이 유지)
    log("4 Move")
    robot.move_to(above(box, LIFT_Z), seconds=1.5)
    robot.hold(0.2)

    # 5. Place: 내려가서 집게를 벌린다
    log("5 Place")
    robot.move_to(above(box, PLACE_Z), seconds=0.8)
    robot.hold(0.2)
    robot.open_gripper(0.5)
    robot.move_to(above(box, LIFT_Z), seconds=0.8)
    robot.hold(0.5)

    ok = is_in_box(robot.cube_pos(), box)
    log(f"   블록 최종 위치 {robot.cube_pos()}  상자 {box[:2]} → {'성공' if ok else '실패'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", type=float, nargs=2, default=[0.24, 0.0])
    ap.add_argument("--box", type=float, nargs=2, default=[0.05, 0.22])
    ap.add_argument("--trials", type=int, default=1, help="2 이상이면 블록 위치를 무작위로 바꿔 반복")
    ap.add_argument("--seed", type=int, default=None,
                    help="생략하면 실행마다 새 위치를 생성. 정수를 지정하면 위치 재현")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default="outputs/ch03_pick_and_place.mp4")
    args = ap.parse_args()
    if args.trials < 1:
        ap.error("--trials는 1 이상이어야 합니다.")

    if args.trials == 1:
        robot = SO101Sim(cube_pos=tuple(args.cube), box_pos=tuple(args.box),
                         render=not args.no_video, camera="fixed")
        ok = pick_and_place(robot)
        print("\n결과:", "성공" if ok else "실패")
        robot.save_video(args.out)
        robot.close()
        return

    if args.seed is None:
        args.seed = int(np.random.SeedSequence().generate_state(1)[0])
    print(f"Random seed: {args.seed}", flush=True)
    rng = np.random.default_rng(args.seed)
    results = []
    records = []
    writer = None
    if not args.no_video:
        import imageio

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        writer = imageio.get_writer(args.out, fps=30)
    try:
        for i in range(args.trials):
            # 로봇 앞쪽 부채꼴 영역(반지름 0.17~0.27, 각도 ±40도)에서 무작위 위치
            r, th = rng.uniform(0.17, 0.27), rng.uniform(-0.7, 0.7)
            cube = (r * np.cos(th), r * np.sin(th))
            print(f"trial {i + 1:2d}/{args.trials} 시작  블록 ({cube[0]:+.3f}, {cube[1]:+.3f})", flush=True)
            robot = SO101Sim(cube_pos=cube, box_pos=tuple(args.box),
                             render=not args.no_video, camera="fixed")
            try:
                ok = pick_and_place(robot, verbose=False)
                results.append(ok)
                trial_path = None
                if writer is not None:
                    stem, ext = os.path.splitext(args.out)
                    trial_path = f"{stem}_trial_{i + 1:02d}{ext}"
                    robot.save_video(trial_path)
                    for frame in robot.frames:
                        writer.append_data(frame)
                records.append({"trial": i + 1, "initial_cube_xy": list(cube),
                                "final_cube_xyz": robot.cube_pos().tolist(),
                                "success": bool(ok), "video": trial_path})
                print(f"trial {i + 1:2d}  → {'성공' if ok else '실패'}", flush=True)
            finally:
                robot.close()
    finally:
        if writer is not None:
            writer.close()
    if writer is not None:
        import json

        report_path = os.path.splitext(args.out)[0] + "_results.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"seed": args.seed, "box_xy": args.box, "trials": records},
                      f, ensure_ascii=False, indent=2)
        print(f"전체 영상: {args.out}, 회차별 결과: {report_path}")
    print(f"\n성공률 {sum(results)}/{len(results)} = {100 * sum(results) / len(results):.0f}%")


if __name__ == "__main__":
    main()
