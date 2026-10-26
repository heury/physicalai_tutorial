"""4부 프로젝트 — Vision과 Robot을 연결하자

3부의 Pick and Place에서 "시뮬레이터에서 위치를 읽는" 두 줄만 "카메라로 찾는" 코드로 바꿉니다.
    Camera → Color Detection → Object Position → Robot Controller → Robot Arm

실행:
    MUJOCO_GL=egl python ch04/vision_pick_and_place.py
    python ch04/vision_pick_and_place.py --no-video --trials 10
    MUJOCO_GL=egl python ch04/vision_pick_and_place.py --trials 2  # 회차별 영상 + 전체 영상
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ch03.pick_and_place import APPROACH_Z, GRASP_Z, LIFT_Z, PLACE_Z, above, is_in_box  # noqa: E402
from common.robot import SO101Sim  # noqa: E402
from common.vision import SimCamera, locate_object  # noqa: E402

np.set_printoptions(precision=3, suppress=True)


def perceive(robot, cam, verbose=True):
    """카메라 한 장으로 블록과 상자의 위치를 추정한다 (Perception)."""
    rgb, depth = cam.capture(robot.data)
    cube_est, _ = locate_object(cam, robot.data, "red", rgb, depth)
    box_est, _ = locate_object(cam, robot.data, "blue", rgb, depth)
    if verbose:
        print(f"   [카메라] 블록 추정 {cube_est}  (실제 {robot.cube_pos()})")
        print(f"   [카메라] 상자 추정 {box_est}  (실제 {robot.box_pos()})")
    return cube_est, box_est


def vision_pick_and_place(robot, cam, verbose=True):
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)

    log("0 Perceive")
    cube, box = perceive(robot, cam, verbose)      # 3부에서는 robot.cube_pos() 였던 자리
    if cube is None or box is None:
        log("   검출 실패")
        return False

    log("1 Approach")
    robot.open_gripper(0.5)
    robot.move_to(above(cube, APPROACH_Z), seconds=1.2)
    log("2 Grasp")
    robot.move_to(above(cube, GRASP_Z), seconds=0.8)
    robot.hold(0.2)
    robot.close_gripper(0.8)
    log("3 Lift")
    robot.move_to(above(cube, LIFT_Z), seconds=1.0)
    robot.hold(0.2)
    if robot.cube_pos()[2] < 0.06:
        log(f"   블록 높이 {robot.cube_pos()[2]:.3f} m → 놓쳤음")
        return False
    log("4 Move")
    robot.move_to(above(box, LIFT_Z), seconds=1.5)
    log("5 Place")
    robot.move_to(above(box, PLACE_Z), seconds=0.8)
    robot.hold(0.2)
    robot.open_gripper(0.5)
    robot.move_to(above(box, LIFT_Z), seconds=0.8)
    robot.hold(0.5)
    ok = is_in_box(robot.cube_pos(), robot.box_pos())
    log(f"   블록 최종 위치 {robot.cube_pos()} → {'성공' if ok else '실패'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", type=float, nargs=2, default=[0.24, 0.0])
    ap.add_argument("--box", type=float, nargs=2, default=[0.05, 0.22])
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None,
                    help="생략하면 실행마다 새 위치를 생성. 정수를 지정하면 위치 재현")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default="outputs/ch04_vision_pick_and_place.mp4")
    args = ap.parse_args()
    if args.trials < 1:
        ap.error("--trials는 1 이상이어야 합니다.")

    if args.trials == 1:
        robot = SO101Sim(cube_pos=tuple(args.cube), box_pos=tuple(args.box),
                         render=not args.no_video, camera="fixed")
        cam = SimCamera(robot.model, camera="top")
        ok = vision_pick_and_place(robot, cam)
        print("\n결과:", "성공" if ok else "실패")
        robot.save_video(args.out)
        cam.close()
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
            r, th = rng.uniform(0.17, 0.27), rng.uniform(-0.7, 0.7)
            cube = (r * np.cos(th), r * np.sin(th))
            print(f"trial {i + 1:2d}/{args.trials} 시작  블록 ({cube[0]:+.3f}, {cube[1]:+.3f})", flush=True)
            robot = SO101Sim(cube_pos=cube, box_pos=tuple(args.box),
                             render=not args.no_video, camera="fixed")
            cam = None
            try:
                cam = SimCamera(robot.model, camera="top")
                ok = vision_pick_and_place(robot, cam)
                results.append(ok)
                trial_path = None
                if writer is not None:
                    # 검출 실패로 즉시 끝나도 확인할 수 있는 장면을 저장한다.
                    if not robot.frames:
                        robot.hold(1.0)
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
                if cam is not None:
                    cam.close()
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
