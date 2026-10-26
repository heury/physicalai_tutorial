# 사용자 요청과 Agent 응답 요약

Hugging Face의 SO-101 Pick and Place 데이터셋 두 개로 LeRobot ACT를 학습하고, 블록 색상 변화에 따른 성능을 비교한 시나리오입니다.

사용한 코드는 [`ch06/compare_act_datasets.py`](../ch06/compare_act_datasets.py), [`ch06/eval_act.py`](../ch06/eval_act.py), [`ch06/compare_act_block_colors.py`](../ch06/compare_act_block_colors.py)입니다.

## 1. 두 데이터셋으로 ACT 학습 및 결과 비교 요청

**사용자:** caspar/so101_pickplace_sim, caspar/so101_pickplace_sim_150 이 두 데이터셋으로 lerobot의 act 모델로 학습하고 결과를 비교해.

**Agent:** 로컬 경로와 Hugging Face 저장소를 확인했다. 두 데이터셋 모두 Hugging Face의 공개 저장소에 있으며, 실제 데이터는 각각 43개 에피소드·7,396프레임과 117개 에피소드·20,124프레임이라고 확인했다. 두 데이터셋 모두 20 FPS, 위·손목 카메라 이미지와 관절 상태 6개를 관측으로 사용하고, 목표 관절각 6개를 행동으로 기록한 형식이었다.

처음에는 제한된 실행 환경에서 CUDA GPU가 보이지 않아 데이터 위치와 GPU 환경을 확인했다.

## 2. Hugging Face 데이터 및 로컬 CUDA 환경 확인 후 학습

**사용자:** 데이터는 huggingface에 있고, 이 컴퓨터는 cuda를 지원해.

**Agent:** 호스트 권한으로 다시 확인해 RTX 3080 Ti 12GB와 PyTorch CUDA 사용을 확인했다. 두 데이터셋을 고정 리비전으로 내려받고 영상 디코딩과 입력 형식을 검사한 뒤, 같은 조건으로 ACT 두 모델을 학습했다.

- LeRobot 0.6.1, ACT 약 5,160만 파라미터, ResNet18 백본.
- 모델마다 2,500스텝, 배치 16, 학습 시드 0, 학습률 `1e-5`.
- `chunk_size=50`, `n_action_steps=50`, AMP와 Temporal Ensembling은 사용하지 않았다.
- 최종 체크포인트를 다시 불러와 동일한 블록 초기 위치 50개에서 평가했다. 평가 시드는 10000, 회당 제한 시간은 12초였다.
- 성공 기준은 기존 `is_in_box()` 조건으로, 제한 시간 내 블록이 상자 안에 처음 들어오는 경우였다.

| 학습 데이터셋 | 실제 에피소드 | 프레임 | 빨간 블록 성공률 | 학습 시간 |
|---|---:|---:|---:|---:|
| `caspar/so101_pickplace_sim` | 43 | 7,396 | 38% (19/50회) | 약 201초 |
| `caspar/so101_pickplace_sim_150` | 117 | 20,124 | 58% (29/50회) | 약 199초 |

`_150` 데이터셋 모델의 성공률이 20%p 높았다고 보고했다. 같은 위치에서 두 모델 모두 성공한 경우는 13회, 기본 데이터셋 모델만 성공한 경우는 6회, `_150` 모델만 성공한 경우는 16회, 둘 다 실패한 경우는 15회였다.

단일 학습 시드·50회 평가의 결과이며, 동일한 학습 스텝 수를 사용해 데이터 반복 횟수는 각각 약 5.41회와 1.99회였다. 따라서 충분히 수렴한 성능의 우열로 일반화하지 않았다. 체크포인트의 학습 스텝·설정과 모델 파일 해시를 검증했고, 학습·평가 종료 후 GPU 작업 프로세스가 남지 않았음을 확인했다.

[상세 결과](../outputs/ch06/act_comparison/comparison_analysis.json) · [비교 그래프](../outputs/ch06/act_comparison/comparison.png) · [동일 위치 비교 영상](../outputs/ch06/act_comparison/paired_trial_01.mp4) · [체크포인트 검증](../outputs/ch06/act_comparison/completion_verified.json)

## 3. 블록을 파란색으로 변경해 비교 테스트

**사용자:** 블럭을 파란색으로 변경해서 비교 테스트해.

**Agent:** 기존 ACT 모델 두 개를 재학습 없이 사용하고, 주 블록의 색상만 빨간색에서 파란색으로 바꿨다. [`common/scene.py`](../common/scene.py)에 `cube_rgba` 옵션을 추가하되 기본 색상은 빨간색으로 유지했다.

빨간 블록의 RGBA는 `(0.9, 0.1, 0.1, 1)`, 파란 블록은 `(0.1, 0.1, 0.9, 1)`로 설정했다. MuJoCo 모델 배열에서 주 블록의 색상만 달라졌고, 200스텝의 물리 시뮬레이션 결과도 동일함을 확인했다. 모델 파일 해시, 블록 초기 위치 50개, 카메라, 행동 실행 설정, 회당 제한 시간 12초를 이전 평가와 대조했다.

| 학습 데이터셋 | 빨간 블록 | 파란 블록 | 성공률 변화 |
|---|---:|---:|---:|
| `caspar/so101_pickplace_sim` | 38% (19/50회) | 6% (3/50회) | −32%p |
| `caspar/so101_pickplace_sim_150` | 58% (29/50회) | 26% (13/50회) | −32%p |

이번 테스트에서 `_150` 모델은 파란 블록에서도 더 높은 성공률을 보였지만, 두 모델 모두 색상 변경 후 성공률이 크게 낮아졌다고 보고했다. 결과 JSON에 회차별 위치·성공 여부·성공까지 걸린 시간을 저장하고, 같은 첫 평가 위치의 빨강·파랑 영상을 4분할 비교 영상으로 남겼다. 평가가 끝난 뒤 CUDA 작업 프로세스가 모두 종료됐음을 확인했다.

[상세 결과](../outputs/ch06/act_color_comparison/comparison.json) · [색상별 비교 그래프](../outputs/ch06/act_color_comparison/color_comparison.png) · [빨강·파랑 비교 영상](../outputs/ch06/act_color_comparison/color_comparison_trial_01.mp4) · [색상만 변경됐는지 검증](../outputs/ch06/act_color_comparison/preflight.json)
