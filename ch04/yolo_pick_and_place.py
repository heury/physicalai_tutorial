"""사전학습 YOLOE → RGB-D 좌표 복원 → Pick and Place.

실행(최초 실행 시 공개 모델 다운로드, 추가 학습 없음):
    MUJOCO_GL=egl .venv/bin/python ch04/yolo_pick_and_place.py --trials 2
동일한 위치 재현:
    MUJOCO_GL=egl .venv/bin/python ch04/yolo_pick_and_place.py --trials 2 --seed 0

고정된 별도 참조 이미지에서 블록·상자를 한 번 지정하는 visual prompt를 사용한다.
가중치를 학습하지 않으며 테스트의 무작위 위치/실제 좌표는 프롬프트로 주지 않는다.
위치 추정에는 YOLO 검출 bbox와 깊이 영상만 사용한다. HSV 대체 검출은 없다.
실제 좌표는 검출 오차 기록 및 집기·놓기 성공 판정에만 사용한다.
"""

import argparse
import json
import os
from pathlib import Path
import sys

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/physicalai_yolo_config")

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from ultralytics import YOLOE, settings
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ch03.pick_and_place import APPROACH_Z, GRASP_Z, LIFT_Z, PLACE_Z, above, is_in_box
from common.robot import SO101Sim
from common.vision import SimCamera


def random_cube(rng):
    radius, angle = rng.uniform(0.17, 0.27), rng.uniform(-0.7, 0.7)
    return np.array([radius * np.cos(angle), radius * np.sin(angle)])


def perceive_yolo(robot, cam, detector, device, confidence):
    rgb, depth = cam.capture(robot.data)
    # Ultralytics ndarray 입력은 BGR이므로 카메라 RGB를 변환한다.
    prediction = detector.predict(source=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                  imgsz=640, conf=confidence, device=device, verbose=False)[0]
    detections = {}
    for cls, name in ((0, "cube"), (1, "box")):
        candidates = [b for b in prediction.boxes if int(b.cls.item()) == cls]
        if not candidates:
            continue
        best = max(candidates, key=lambda b: float(b.conf.item()))
        x1, y1, x2, y2 = best.xyxy[0].cpu().numpy()
        u, v = float((x1+x2)/2), float((y1+y2)/2)
        ui, vi = int(round(u)), int(round(v))
        patch = depth[max(0, vi-1):min(cam.height, vi+2), max(0, ui-1):min(cam.width, ui+2)]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if not valid.size:
            continue
        xyz = cam.pixel_to_world(robot.data, u, v, float(np.median(valid)))
        detections[name] = {"confidence": float(best.conf.item()),
                            "bbox_xyxy": [float(n) for n in (x1, y1, x2, y2)],
                            "estimated_xyz": xyz.tolist()}
    annotated = cv2.cvtColor(prediction.plot(), cv2.COLOR_BGR2RGB)
    return detections, annotated


def execute(robot, detections):
    if "cube" not in detections or "box" not in detections:
        return False, "detection_failed"
    cube = np.array(detections["cube"]["estimated_xyz"])
    box = np.array(detections["box"]["estimated_xyz"])
    robot.open_gripper(0.5)
    robot.move_to(above(cube, APPROACH_Z), seconds=1.2)
    robot.move_to(above(cube, GRASP_Z), seconds=0.8)
    robot.hold(0.2)
    robot.close_gripper(0.8)
    robot.move_to(above(cube, LIFT_Z), seconds=1.0)
    robot.hold(0.2)
    if robot.cube_pos()[2] < 0.06:  # 집기 성공 판정에만 사용
        return False, "grasp_failed"
    robot.move_to(above(box, LIFT_Z), seconds=1.5)
    robot.move_to(above(box, PLACE_Z), seconds=0.8)
    robot.hold(0.2)
    robot.open_gripper(0.5)
    robot.move_to(above(box, LIFT_Z), seconds=0.8)
    robot.hold(0.5)
    ok = bool(is_in_box(robot.cube_pos(), robot.box_pos()))
    return ok, "success" if ok else "place_failed"


def run_trials(detector, root, trials, seed, device, confidence):
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    print(f"Random seed: {seed}", flush=True)
    rng = np.random.default_rng(seed)
    records = []
    combined = root / "random2.mp4" if trials == 2 else root / f"random{trials}.mp4"
    with imageio.get_writer(combined, fps=30) as all_writer:
        for i in range(1, trials + 1):
            cube = random_cube(rng)
            print(f"trial {i}/{trials}: initial cube={cube.tolist()}", flush=True)
            robot = SO101Sim(cube_pos=tuple(cube), box_pos=(0.05, 0.22), camera="fixed")
            cam = None
            try:
                cam = SimCamera(robot.model, camera="top_zoom")
                detections, annotated = perceive_yolo(robot, cam, detector, device, confidence)
                print(f"YOLO: {detections}", flush=True)
                # 실제 좌표는 검출이 끝난 뒤 평가 기록에만 사용한다.
                for name, actual in (("cube", robot.cube_pos()), ("box", robot.box_pos())):
                    if name in detections:
                        estimate = np.array(detections[name]["estimated_xyz"])
                        detections[name]["xy_error_m"] = float(np.linalg.norm(estimate[:2] - actual[:2]))
                imageio.imwrite(root / f"trial_{i:02d}_yolo.png", annotated)
                ok, reason = execute(robot, detections)
                if not robot.frames:
                    robot.hold(1.0)
                video = root / f"trial_{i:02d}.mp4"
                title = annotated.copy()
                cv2.putText(title, f"Trial {i}: initial YOLO detection", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
                with imageio.get_writer(video, fps=30) as writer:
                    for frame in [title] * 30 + robot.frames:
                        writer.append_data(frame)
                        all_writer.append_data(frame)
                imageio.imwrite(root / f"trial_{i:02d}_last.png", robot.frames[-1])
                records.append({"trial": i, "initial_cube_xy": cube.tolist(),
                                "detections": detections, "success": ok, "reason": reason,
                                "final_cube_xyz": robot.cube_pos().tolist(),
                                "video": str(video), "frames": 30 + len(robot.frames)})
                (root / "results.json").write_text(json.dumps(
                    {"seed": seed, "weights": str(detector.ckpt_path),
                     "confidence_threshold": confidence, "image_size": 640,
                     "prompts": detector.names, "additional_training": False,
                     "visual_prompt_profile": str(root / "visual_prompts.npz"),
                     "reference_image": str(root / "reference.png"),
                     "trials": records, "combined_video": str(combined)}, indent=2))
                print(f"trial {i}: {reason}, video={video}", flush=True)
            finally:
                if cam is not None:
                    cam.close()
                robot.close()
    print(f"Completed: {sum(r['success'] for r in records)}/{trials}; {combined}", flush=True)


def load_detector(root, weights, device):
    detector = YOLOE(str(weights))
    profile = root / "visual_prompts.npz"
    if profile.is_file():
        detector.load_prompt_embeddings(profile)
        return detector
    reference = root / "reference.png"
    # 참조 장면은 테스트와 독립적인 고정 위치다. 아래 bbox는 이 640×480 이미지의
    # 블록/상자를 사람이 확인해 지정한 픽셀 좌표이며, 테스트 위치를 입력하지 않는다.
    prompts = {"bboxes": np.array([[394, 274, 427, 307], [165, 14, 274, 126]]),
               "cls": np.array([0, 1])}
    robot = SO101Sim(cube_pos=(0.24, 0), box_pos=(0.05, 0.22), render=False)
    cam = None
    try:
        cam = SimCamera(robot.model, camera="top_zoom")
        rgb, _ = cam.capture(robot.data)
        imageio.imwrite(reference, rgb)
    finally:
        if cam is not None:
            cam.close()
        robot.close()
    (root / "reference.json").write_text(json.dumps(
        {"image": str(reference), "bbox_annotation": "manual, fixed reference only",
         "bboxes": prompts["bboxes"].tolist(), "classes": ["cube", "box"]}, indent=2))
    detector.predict(str(reference), refer_image=str(reference), visual_prompts=prompts,
                     predictor=YOLOEVPSegPredictor, imgsz=640, conf=0.1,
                     device=device, verbose=False)
    detector.set_classes(["cube", "box"], detector.model.pe)
    detector.save_prompt_embeddings(profile)
    return detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None,
                        help="생략하면 실행마다 새 위치를 생성. 정수를 지정하면 위치 재현")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/yolo_pick_place"))
    args = parser.parse_args()
    if args.trials < 1 or not 0 < args.conf <= 1:
        parser.error("trials는 1 이상, conf는 0 초과 1 이하여야 합니다.")
    torch.set_num_threads(4)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    settings.update({"weights_dir": str(args.out_dir.resolve()), "sync": False})
    weights = args.weights or args.out_dir / "yoloe-11s-seg.pt"
    detector = load_detector(args.out_dir, weights, args.device)
    run_trials(detector, args.out_dir, args.trials, args.seed, args.device, args.conf)


if __name__ == "__main__":
    main()
