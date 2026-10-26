# 사용자 요청과 Agent 응답 요약

빨강·초록·노랑 블록이 함께 있는 장면에서 색상을 지정하는 시연 150개를 만들어 Hugging Face에 업로드하고, 이 데이터로 SmolVLA를 학습한 뒤 색상별 Pick and Place 성능을 평가한 시나리오입니다.

사용한 코드는 [`ch08/record_multicolor.py`](../ch08/record_multicolor.py), [`ch08/eval_multicolor.py`](../ch08/eval_multicolor.py)입니다. 실행일은 2026-10-23입니다.

## 1. 멀티컬러 시연 150개 생성 및 Hugging Face 업로드

**사용자:** ch08/record_multicolor.py 을 이용하여 150개 데이터를 만들어서 huggingface에 so101_multicolor_sim로 업로드해

**Agent:** 기존 `--episodes` 옵션이 성공 시연 수가 아니라 시도 횟수를 뜻하는 것을 확인했다. 실패한 시연을 버리고도 성공 시연 150개를 확보할 수 있도록 `--retry-failed` 옵션을 추가했다. 이 옵션은 성공 시연이 지정한 수에 도달할 때까지 같은 목표 색상을 재시도하며, 최대 시도 횟수는 목표 수의 10배로 제한했다.

처음에는 로컬 소프트웨어 렌더링으로 생성을 시작했다. 사용자가 Hugging Face 인증을 완료하고 A40 사용을 선택한 뒤 Runpod으로 전환했다. 최종 데이터에는 A40에서 생성한 시연만 포함했다.

- 빨강·초록·노랑 블록을 함께 배치하고, 명령은 `Pick up the {color} cube and put it in the blue box.`로 지정했다.
- 독립된 작업 5개에서 각각 성공 시연 30개를 저장했다. 각 작업은 색상별 10개씩 생성했고, seed는 `0, 10000, 20000, 30000, 40000`을 사용했다.
- LeRobot의 데이터셋 통합 기능으로 에피소드·프레임·작업 인덱스와 통계를 통합했다.
- 성공 여부는 스크립트의 기존 `is_in_box()` 조건으로 확인했으며, 실패한 시연은 저장하지 않았다.

| 목표 색상 | 저장한 성공 시연 |
|---|---:|
| 빨강 | 50개 |
| 초록 | 50개 |
| 노랑 | 50개 |
| **합계** | **150개** |

**결과:** 총 **25,800프레임**, 20 FPS, 시뮬레이션 시간 약 21.5분의 데이터셋을 생성했다. 위·손목 카메라의 240 × 320 RGB 영상, 관절 상태 6개, 목표 관절각 6개를 LeRobot v3 형식으로 기록했다. 전체 데이터를 train split에 넣었으며 별도의 검증 분할은 만들지 않았다.

에피소드·프레임 수, 색상별 개수, 연속된 인덱스, 관절 상태·행동의 유한값, 영상 디코딩을 검증하고 [caspar/so101_multicolor_sim](https://huggingface.co/datasets/caspar/so101_multicolor_sim)에 업로드했다. 업로드 커밋은 `1d47ca0df5c9b958acd005817fdb5bf6a1e42bd7`이다. Hub의 메타데이터와 파일 목록을 다시 확인하고, 해당 커밋의 로컬 사본도 내려받았다.

공유 볼륨에는 `/workspace/multicolor/20261023/so101_multicolor_sim`, 로컬에는 `outputs/datasets/caspar/so101_multicolor_sim`으로 보존했다. 데이터 생성용 A40 Pod는 업로드와 검증 후 삭제했다.

[생성·업로드 보고서](../output/multicolor_dataset_20261023/REPORT.md) · [검증 결과](../output/multicolor_dataset_20261023/a40_validation.json) · [업로드 확인](../output/multicolor_dataset_20261023/a40_upload_complete.json)

## 2. 멀티컬러 데이터로 SmolVLA 학습

**사용자:** 이 데이터를 이용하여 smolvla-base 모델을 학습해

**Agent:** 앞서 업로드한 `caspar/so101_multicolor_sim`의 성공 시연 150개 전체로 `lerobot/smolvla_base`를 파인튜닝했다. 데이터와 기본 모델의 리비전을 고정하고, Runpod A40 한 대에서 학습 결과와 로그를 공유 볼륨에 저장했다.

- 데이터 리비전: `1d47ca0df5c9b958acd005817fdb5bf6a1e42bd7`.
- 기본 모델 리비전: `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`.
- 10,000스텝, 배치 8, 학습 seed 1000, LeRobot 0.6.1, PyTorch 2.8.0.
- 전체 약 4억 5천만 파라미터 중 기본 expert 학습 설정에 따라 약 1억 개를 학습했으며, AMP는 사용하지 않았다.
- 위·손목 카메라 관측 이름을 각각 `camera1`, `camera2`로 연결하고, 체크포인트는 1,000스텝마다 저장했다.
- 전체 데이터를 학습에 사용했으며 별도의 검증 데이터 분할은 만들지 않았다.

**결과:** 10,000스텝 학습을 완료했고, 최종 기록된 학습 손실은 **0.018**이었다. 최종 체크포인트는 공유 볼륨의 `/workspace/smolvla/20261023/so101_multicolor_sim/train/checkpoints/last/pretrained_model`에 저장했다.

학습 후 CUDA에서 체크포인트를 다시 불러와 빨강·초록·노랑 작업의 관측으로 추론했다. 세 입력 모두 유한한 6차원 행동을 출력했으며, 저장한 모델·전처리기·후처리기 파일의 해시를 별도의 SSH 읽기로 다시 확인했다. 이 단계는 모델 파일과 추론 실행 검증으로, 실제 집기·상자 넣기 성공률을 측정한 것은 아니라고 구분했다.

학습용 A40 Pod는 검증과 로그 보존 후 삭제하고, Pod 목록에서 사라진 것을 확인했다. 체크포인트·정규화 통계·설정·학습 코드는 공유 볼륨에 보존했다.

[학습 보고서](../output/smolvla_multicolor_train_20261023/REPORT.md) · [학습 결과 요약](../output/smolvla_multicolor_train_20261023/summary.json) · [손실 곡선](../output/smolvla_multicolor_train_20261023/learning_curve.png) · [체크포인트 파일 검증](../output/smolvla_multicolor_train_20261023/artifact_readback.json)

## 3. 학습한 SmolVLA의 색상별 평가

**사용자:** 이 모델로 색깔별로 테스트해서 결과를 보고해

**Agent:** 앞서 만든 150개 시연으로 `lerobot/smolvla_base`를 10,000스텝, 배치 8, 학습 seed 1000으로 파인튜닝한 최종 체크포인트를 사용했다. 최종 학습 손실은 0.018이었고, 평가 전에 모델 재로드와 세 색상 입력의 행동 출력이 유한한지 확인한 상태였다.

체크포인트는 공유 볼륨의 `/workspace/smolvla/20261023/so101_multicolor_sim/train/checkpoints/last/pretrained_model`이다. 학습 손실이나 행동 출력 검증을 실제 작업 성공률과 구분하고, A40에서 `ch08/eval_multicolor.py`의 `run_episode()`를 그대로 호출해 평가했다.

- 학습 시연과 별도로 배치 seed 2000으로 블록 배치 20개를 만들었다.
- 각 배치에서 블록 위치와 행동 noise seed를 고정하고, 명령 색상만 빨강·초록·노랑으로 바꿨다. noise seed는 배치별 `0–19`를 사용했다.
- 색상별 20회, 총 60회를 실행했고 회당 시뮬레이션 제한 시간은 12초였다.
- 위 카메라는 `top`, `chunk_size=50`, `n_action_steps=50`, `num_steps=10`을 사용했다.
- 목표 색상만 상자에 처음 들어오면 성공, 다른 색상이나 복수 색상이 먼저 들어오면 오답, 시간 안에 아무 블록도 들어오지 않으면 실패로 집계했다.

| 목표 색상 | 성공 | 다른 색·복수 색 오답 | 상자 진입 없음 | 성공률 |
|---|---:|---:|---:|---:|
| 빨강 | 0 | 0 | 20 | 0% |
| 초록 | 0 | 0 | 20 | 0% |
| 노랑 | 0 | 0 | 20 | 0% |
| **합계** | **0** | **0** | **60** | **0%** |

**결과:** 전체 성공률은 **0% (0/60회)**였다. 세 색상 모두 제한 시간 내 상자 진입을 완료하지 못했으며, 다른 색 블록을 상자에 넣은 오답도 없었다. 낮은 학습 손실이 이번 평가의 작업 성공으로 이어지지는 않았다고 보고했다.

같은 첫 배치의 영상을 비교하면 초록·노랑 명령에서도 팔이 빨간 블록 쪽으로 접근하는 모습이 관찰됐다. 이 관찰은 배치 1에 대한 설명으로 한정했으며, 전체 배치의 색상 선택 정확도를 정량화한 결과로 해석하지 않았다. 장면·로봇·카메라·IK·시연 관련 코드 6개의 해시가 데이터 생성 당시 보관한 코드와 일치함을 확인했다.

회차별 배치·명령·seed·결과를 JSON과 CSV로 저장하고, 같은 첫 배치의 색상별 영상도 보존했다. 평가용 A40 Pod는 결과와 영상을 내려받은 뒤 삭제했으며, 모델과 공유 볼륨은 보존했다.

[학습 결과](../output/smolvla_multicolor_train_20261023/REPORT.md) · [평가 보고서](../output/smolvla_multicolor_eval_20261023/REPORT.md) · [회차별 결과](../output/smolvla_multicolor_eval_20261023/results.json) · [CSV](../output/smolvla_multicolor_eval_20261023/episodes.csv)

[빨강 첫 회 영상](../output/smolvla_multicolor_eval_20261023/red_layout01.mp4) · [초록 첫 회 영상](../output/smolvla_multicolor_eval_20261023/green_layout01.mp4) · [노랑 첫 회 영상](../output/smolvla_multicolor_eval_20261023/yellow_layout01.mp4) · [동작 비교 이미지](../output/smolvla_multicolor_eval_20261023/first_layout_contact_sheet.jpg)
