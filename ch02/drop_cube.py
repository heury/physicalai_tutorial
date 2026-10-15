"""2부 실습 — 가상 세계에 물체 만들기: 책상 위에 블록 떨어뜨리기

Plane(바닥) + 책상(box) + 블록(cube) + 공(sphere)을 만들고,
블록이 책상 위로 떨어져 멈추는 과정을 시뮬레이션합니다.
화면(GUI) 없이도 동작하도록 Headless 렌더링으로 영상을 저장합니다.

실행:
    uv run ch02/drop_cube.py --gravity -1.62          # 화면 없는 Linux에서는 EGL 자동 선택
    python ch02/drop_cube.py --no-video               # 렌더링 없이 숫자만 확인

옵션:
    --mass 0.5        블록 질량(kg)
    --friction 0.1    마찰 계수
    --gravity -1.62   중력 가속도 (달 = -1.62)
"""

import argparse
import os
import sys

import numpy as np

WORLD_XML = """
<mujoco model="drop_cube">
  <option gravity="0 0 {gravity}" timestep="0.002"/>
  <visual>
    <global offwidth="640" offheight="480"/>
  </visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1"/>

    <!-- 책상: joint가 없으므로 세계에 고정. 바닥에서 윗면까지 높이 0.4m -->
    <body name="table" pos="0 0 0.2">
      <geom type="box" size="0.4 0.3 0.2" rgba="0.6 0.4 0.2 1" friction="{friction} 0.005 0.0001"/>
    </body>

    <!-- 빨간 블록: 책상 위 0.5m 공중에서 시작, freejoint로 자유롭게 움직임 -->
    <body name="cube" pos="0.1 0 0.9" euler="20 30 0">
      <freejoint/>
      <geom type="box" size="0.03 0.03 0.03" mass="{mass}" rgba="1 0.1 0.1 1"
            friction="{friction} 0.005 0.0001"/>
    </body>

    <!-- 파란 공 -->
    <body name="ball" pos="-0.2 0.1 1.1">
      <freejoint/>
      <geom type="sphere" size="0.03" mass="0.1" rgba="0.1 0.3 1 1"/>
    </body>

    <camera name="side" mode="targetbody" target="table" pos="0.9 -0.9 0.9"/>
  </worldbody>
</mujoco>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=float, default=0.1)
    ap.add_argument("--friction", type=float, default=1.0)
    ap.add_argument("--gravity", type=float, default=-9.81)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default="outputs/ch02_drop_cube.mp4")
    ap.add_argument("--frame", default="outputs/ch02_drop_cube_last.png")
    args = ap.parse_args()

    # MuJoCo는 import 시 렌더링 백엔드를 결정한다.
    if args.no_video:
        os.environ["MUJOCO_GL"] = "disable"
    elif sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")

    import mujoco

    xml = WORLD_XML.format(mass=args.mass, friction=args.friction, gravity=args.gravity)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    mujoco.mj_forward(model, data)   # 초기 위치 계산

    renderer = None
    frames = []
    if not args.no_video:
        renderer = mujoco.Renderer(model, height=480, width=640)

    fps = 30
    steps_per_frame = int(1 / (fps * model.opt.timestep))
    n_steps = int(args.seconds / model.opt.timestep)

    print(f"mass={args.mass}kg friction={args.friction} gravity={args.gravity}")
    print(" time(s)   cube z(m)   ball z(m)   cube speed(m/s)")
    for step in range(n_steps + 1):
        if step % 100 == 0:   # 0.2초마다 출력
            cz, bz = data.xpos[cube_id][2], data.xpos[ball_id][2]
            speed = np.linalg.norm(data.cvel[cube_id][3:])  # cvel: [각속도 3, 선속도 3]
            print(f"{data.time:7.3f}   {cz:8.4f}    {bz:8.4f}    {speed:8.4f}")
        if renderer is not None and step % steps_per_frame == 0:
            renderer.update_scene(data, camera="side")
            frames.append(renderer.render().copy())
        mujoco.mj_step(model, data)

    if renderer is not None:
        renderer.close()

    if frames:
        import imageio

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        imageio.mimsave(args.out, frames, fps=fps)
        imageio.imwrite(args.frame, frames[-1])
        print(f"\n영상 저장: {args.out} ({len(frames)} frames), 마지막 프레임: {args.frame}")


if __name__ == "__main__":
    main()
