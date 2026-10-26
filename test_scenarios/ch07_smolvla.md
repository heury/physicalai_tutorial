# 사용자 요청과 Agent 응답 요약

Hugging Face의 `lerobot/smolvla_base` 사전학습 모델로 Pick and Place를 평가하고, 명령 문장 변화와 행동 생성 noise의 영향을 비교한 시나리오입니다. 공유 볼륨에 저장된 파인튜닝 모델 두 개를 Runpod A40에서 평가한 대화도 포함합니다.

사용한 코드는 [`ch07/eval_smolvla.py`](../ch07/eval_smolvla.py), [`ch07/task_variation.py`](../ch07/task_variation.py)입니다. 1~3절은 사전학습 모델을 추가 학습 없이 평가했고, 4절은 이미 파인튜닝된 체크포인트를 불러와 평가했습니다.

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

성공률이 높지 않은 이유는 모델이 lerobot/libero로 학습이 되어 테스트에 사용한 로봇과 환경이 다르기 때문이다.

[전체 실행 로그](../outputs/ch07/smolvla_base_task_variation.log)

## 4. 공유 볼륨의 학습 모델을 Runpod A40에서 평가

**사용자:** runpod에서 A40 pod를 띄워서 학습한 smolvla 모델 테스트해. 모델은 shared-volume에 있어

**Agent:** Runpod의 `shared-volume`(`aiz2tt6eps`)이 CA-MTL-1에 있는 것을 확인하고, 같은 센터에 A40 Pod 한 대를 생성했다. 공유 볼륨에서 각각 `caspar/so101_pickplace_sim`, `caspar/so101_pickplace_sim_150`으로 10,000스텝 학습한 SmolVLA 체크포인트 두 개를 찾았다. 평가 실행일은 2026-10-23이다.

체크포인트는 `/workspace/smolvla/20261022/<데이터셋 이름>/train/checkpoints/last/pretrained_model`에서 불러왔다. [`평가 실행 코드`](../output/smolvla_test_20261023/evaluate.py)는 `ch07/eval_smolvla.py`의 모델 로더·카메라 키 매핑과 `ch06/eval_act.py`의 닫힌 고리 평가 함수를 사용했다.

- NVIDIA A40, LeRobot 0.6.1, PyTorch 2.8.0, MuJoCo SO-101 시뮬레이션에서 실행했다.
- 두 모델에 같은 블록 초기 위치 20개를 사용했다. 위치 seed는 1000, 행동 noise seed는 회차별 `0–19`로 고정했다.
- 명령은 `Pick up the red cube and put it in the blue box.`이며, 회당 시뮬레이션 제한 시간은 12초였다.
- 저장된 전처리기·후처리기의 정규화 통계를 사용하고, 위·손목 카메라를 각각 `camera1`, `camera2`에 연결했다.
- `chunk_size=50`, `n_action_steps=50`을 사용했다.
- 성공은 기존 `ch03.pick_and_place.is_in_box()` 조건에 따라 블록이 상자 안에 처음 들어오는 경우로 집계했다. 상자 안에 지속적으로 머무르는지는 별도로 측정하지 않았다.

| 학습 데이터셋 | 성공 | 평가 횟수 | 성공률 |
|---|---:|---:|---:|
| `caspar/so101_pickplace_sim` | 4 | 20 | 20% |
| `caspar/so101_pickplace_sim_150` | 7 | 20 | 35% |

**결과:** 이번 20개 초기 위치에서는 `_150` 데이터셋으로 학습한 모델의 성공률이 15%p 높았다. 작은 시뮬레이션 표본의 결과이므로 일반적인 성능 우열이나 실제 로봇 성능으로 확대해 해석하지 않았다.

모델별 첫 회 영상을 저장했으며, 첫 회는 두 모델 모두 실패했다. 회차별 초기 위치·성공 여부·성공까지 걸린 시간·실행 시간을 JSON으로 보존했다. 결과와 영상을 로컬에 내려받은 뒤 이번 평가용 Pod `dt8hzu8r9691a9`를 삭제하고 목록에서 사라진 것을 확인했다. 기존 Pod와 공유 볼륨은 보존했으며, 평가 코드·로그·결과·영상은 공유 볼륨의 `/workspace/smolvla/test_20261023`에도 남겼다.

[평가 보고서](../output/smolvla_test_20261023/REPORT.md) · [회차별 결과](../output/smolvla_test_20261023/results.json) · [실행 로그](../output/smolvla_test_20261023/evaluate.log) · [기본 데이터셋 모델 영상](../output/smolvla_test_20261023/so101_pickplace_sim.mp4) · [`_150` 데이터셋 모델 영상](../output/smolvla_test_20261023/so101_pickplace_sim_150.mp4)

## 5. 공유 볼륨의 학습 로그를 이용한 결과 보고

**사용자:** caspar/so101_pickplace_sim와 caspar/so101_pickplace_sim_150로 학습한 모델이 shared-volume에 저장이 되어 있어. 로그를 이용하여 결과 보고해

**Agent:** 이미 실행 중인 Pod를 통해 공유 볼륨의 원본 학습 로그·설정·체크포인트를 읽기 전용으로 확인했다. 새 Pod 생성, 재학습, 추가 추론이나 시뮬레이션 평가는 하지 않았다. 학습 요청 원문을 재구성하지 않고, 남아 있는 로그와 파일을 근거로 결과를 보고했다.

| 학습 데이터셋 | 실제 에피소드 | 프레임 | 학습 스텝 | 50스텝 손실 | 최종 학습 손실 | 학습 루프 시간 | 데이터 반복 횟수 |
|---|---:|---:|---:|---:|---:|---|---:|
| `caspar/so101_pickplace_sim` | 43 | 7,396 | 10,000 | 0.544 | 0.013 | 32분 4초 | 10.82회 |
| `caspar/so101_pickplace_sim_150` | 117 | 20,124 | 10,000 | 0.544 | 0.013 | 37분 58초 | 3.98회 |

`_150`은 저장소 이름이며 실제 학습 에피소드는 117개였다. 50스텝 손실은 처음 로그가 기록된 시점의 값으로, 초기화 직후의 손실은 아니다. 학습 루프 시간은 중간 체크포인트 저장을 포함하고 환경 설치·데이터 준비·최종 재로드 검증 시간은 제외한다.

- 두 모델 모두 `lerobot/smolvla_base`에서 시작했으며, A40·LeRobot 0.6.1·PyTorch `2.8.0+cu128`을 사용했다.
- 배치 8, 학습 seed 1000, DataLoader worker 4로 실행했다. 전체 약 4억 5천만 파라미터 중 약 9,988만 개를 학습했고, AMP는 사용하지 않았다.
- 설정 학습률은 `1e-4`, 마지막 기록 학습률은 `2.5e-6`이었다. 로그에서 scheduler의 warmup이 333스텝, decay가 10,000스텝으로 자동 조정됐음을 확인했다.
- 체크포인트는 1,000스텝마다 저장하고, 지표는 50스텝마다 기록했다. 각 실행에서 손실 기록 200개, 최종 10,000스텝 저장과 `End of training` 메시지를 확인했다.
- 전체 시연을 학습에 사용했으며 별도의 검증 데이터 분할이나 검증 손실은 없었다.

**결과:** 두 모델 모두 10,000스텝 학습을 정상 완료했고 최종 학습 손실은 0.013이었다. 데이터 크기가 달라 같은 80,000개 프레임 샘플을 처리하는 동안 반복 횟수는 약 10.82회와 3.98회로 달랐다. 최종 정규화 학습 손실이 같다는 사실만으로 실제 작업 성능이 같다고 판단하지 않았다.

공유 볼륨의 `/workspace/smolvla/20261022/<데이터셋 이름>/train/checkpoints/last/pretrained_model`에서 모델·설정·전처리기·후처리기 파일을 다시 읽어 SHA-256을 계산했다. 두 모델 모두 학습 완료 당시 기록과 일치했고, 원본 학습 로그도 로컬 보관 로그와 해시가 일치했다. 당시 완료 기록에 모델 재로드와 유한한 6차원 행동 출력 검증이 남아 있었지만, 이번 보고에서 해당 추론을 새로 실행한 것은 아니라고 구분했다.

기존에 별도로 수행했던 4절의 시뮬레이션 평가 결과인 **20% (4/20회)**와 **35% (7/20회)**도 함께 제시했다. 이 수치는 학습 로그에서 추정한 성공률이 아니라 이전 실제 평가 기록이며, 이번 요청에서는 재평가하지 않았다.

[로그 분석 보고서](../output/smolvla_training_log_report_20261023/REPORT.md) · [비교 요약](../output/smolvla_training_log_report_20261023/summary.json) · [학습 손실 비교 그래프](../output/smolvla_training_log_report_20261023/learning_curves.png) · [학습 지표 CSV](../output/smolvla_training_log_report_20261023/metrics.csv) · [공유 볼륨 확인 기록](../output/smolvla_training_log_report_20261023/shared_volume_evidence.json)
