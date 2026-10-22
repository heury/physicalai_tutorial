# 사용자 요청과 Agent 응답 요약

Hugging Face의 `lerobot/smolvla_base` 사전학습 모델로 Pick and Place를 평가하고, 명령 문장 변화와 행동 생성 noise의 영향을 비교한 시나리오입니다.

사용한 코드는 [`ch07/eval_smolvla.py`](../ch07/eval_smolvla.py), [`ch07/task_variation.py`](../ch07/task_variation.py)입니다. 모델을 추가 학습하지 않고 평가했습니다.

## 1. SmolVLA 사전학습 모델 평가 요청

**사용자:** ch07/eval_smolvla.py를 이용하여 lerobot/smolvla_base 모델을 테스트해.

**Agent:** 스크립트와 실행 환경을 확인했다. 제한된 샌드박스에서 `torch.cuda.is_available()`이 `False`이고 `nvidia-smi`도 실패하자 CPU 평가를 시작했으나, 모델 다운로드 중 네트워크 접근도 실패했다. 이 실행에서는 평가 결과가 생성되지 않았다.

## 2. 로컬 CUDA 환경 확인 후 GPU 평가

**사용자:** 이 컴퓨터는 cuda를 지원해

**Agent:** 샌드박스에서 GPU 장치가 노출되지 않는 것을 확인하고, 샌드박스 밖에서 RTX 3080 Ti 12GB와 실제 CUDA 연산을 검증했다. 컴퓨터가 CUDA를 지원하지 않는다고 잘못 판단한 점을 정정하고 GPU로 평가를 다시 실행했다.

- PyTorch `2.11.0+cu130`, LeRobot `0.6.1`, SmolVLA 약 4억 5천만 파라미터.
- 모델: `lerobot/smolvla_base`, 리비전 `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`.
- 명령: `Pick up the red cube and put it in the blue box.`
- 평가 5회, 회당 시뮬레이션 제한 시간 12초, 블록 위치 seed 1000, noise seed는 0부터 회차마다 1씩 증가.
- `chunk_size=50`, `n_action_steps=50`, 위·손목 카메라를 각각 `camera1`, `camera2`에 연결했다.
- 성공 기준은 기존 `is_in_box()` 조건으로, 제한 시간 내 블록이 상자 안에 처음 들어오는 경우였다.

**결과:** 성공률 **0% (0/5회)**. 모델 로딩 약 63초, 평가 약 17초였다. CUDA 추론은 정상 실행됐지만 이번 조건에서는 과제를 수행하지 못했다고 보고했다. 첫 회 영상을 저장하고 영상 읽기와 프레임 수를 확인했다.

[상세 결과](../outputs/ch07/smolvla_base_eval_results.json) · [첫 회 영상](../outputs/ch07/smolvla_base_eval.mp4) · [실행 로그](../outputs/ch07/smolvla_base_eval.log)

## 3. 명령 문장 변화와 noise 영향 비교

**사용자:** ch07/task_variation.py를 이용하여 lerobot/smolvla_base 모델을 테스트해.

**Agent:** RTX 3080 Ti에서 스크립트의 기본 설정으로 실행했다. 같은 장면에서 명령 4개와 noise seed 5개(`0, 1, 2, 3, 4`)를 조합해 각각 시뮬레이션 4초씩 실행했다. 블록 위치는 `(0.24, 0.0)`으로 고정하고 위 카메라는 `top`을 사용했다. 본 비교 20회에 앞서 같은 명령·같은 seed로 2회 실행해 재현성을 확인했다.

비교한 명령은 다음과 같다.

- `Pick up the red cube and put it in the blue box.`
- `Move the cube to the left.`
- `Do nothing.`
- `Raise your arm up high.`

| 비교 조건 | 손끝 최종 위치 차이 평균 |
|---|---:|
| seed를 고정하고 문장만 변경 | 23.7cm |
| 문장을 고정하고 noise seed만 변경 | 24.7cm |
| 같은 문장·같은 seed 재실행 | 0.00cm |

문장 효과는 각 seed에서 명령 간 최종 위치의 모든 쌍별 거리를 평균한 뒤 seed 전체를 평균했고, noise 효과는 각 명령에서 seed 간 모든 쌍별 거리를 평균한 뒤 명령 전체를 평균했다. 재현성 차이는 로그에 표시된 소수 둘째 자리 기준이다.

문장에 따라 행동은 달라졌지만 문장 효과가 noise 효과보다 크지 않아, 명령을 의미대로 따른다고 판단하기 어렵다고 보고했다. `Do nothing.`에서도 손끝의 누적 경로 길이가 161~297cm였다. 이 실험은 과제 성공률이 아니라 언어에 따른 행동 변화를 측정한 결과라고 구분했다.

[전체 실행 로그](../outputs/ch07/smolvla_base_task_variation.log)
