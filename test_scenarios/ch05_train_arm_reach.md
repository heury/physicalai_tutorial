# 사용자 요청과 Agent 응답 요약

`ch05/train_arm_reach.py`를 사용한 시나리오입니다.

## 1. 사용하는 보상 함수 확인

**사용자:** ch05/train_arm_reach.py에서 사용하는 reward 함수 보여줘.

**Agent:** 보상 함수는 [`ch05/arm_reach_env.py`](../ch05/arm_reach_env.py)의 `ArmReachEnv.step()`에 정의돼 있다고 설명하고 코드를 보여줬다.

```python
dist = self._dist()       # 손끝과 목표 사이의 3차원 거리(m)
success = dist < 0.02     # 2cm 이내 도달

if self.reward_mode == "sparse":
    reward = 10.0 if success else 0.0
elif self.reward_mode == "dense":
    reward = -dist
else:  # shaped
    reward = (
        -dist
        + 10.0 * (self.prev_dist - dist)
        + (10.0 if success else 0.0)
    )

self.prev_dist = dist
```

기본값과 당시 실행한 실험은 `dense`이며, `--reward shaped`를 지정하면 거리 변화와 성공 보너스까지 사용한다고 안내했다.

## 2. 각 보상 모드 테스트 및 결과 보고

**사용자:** 각 reward mode에 대해서 테스트하고 결과로 보고해.

**Agent:** 원본 스크립트의 `sparse`, `dense`, `shaped`를 CPU·병렬 환경 8개·학습 시드 0의 동일한 설정으로 비교했다. 완료된 `dense` 실험을 재사용하고 나머지 두 모드를 학습했다. 모드별로 30만 스텝을 요청했으며, 롤아웃 단위로 실행돼 실제 학습량은 각각 311,296스텝이었다.

| 보상 모드 | 원본 평가 50회 성공률 | 별도 평가 100회 성공률 | 평균 최종 거리¹ | 평균 종료 스텝¹ |
|---|---:|---:|---:|---:|
| `sparse` | 42% | 38% | 18.60cm | 76.18 |
| `dense` | 94% | 97% | 1.82cm | 25.48 |
| `shaped` | 100% | 99% | 1.85cm | 23.03 |

¹ 세 모델에 동일한 별도 평가 위치 100곳을 사용한 결과다. 성공은 손끝이 목표에서 2cm 이내에 도달하는 조건이다.

이번 시드 0 실험에서는 `shaped`의 성공률이 가장 높았다고 보고했다. 원본 코드는 수정하지 않았으며, 저장 모델의 재로드와 영상 읽기를 확인했다.

[상세 결과](../outputs/ch05/reward_modes_300k_seed0/summary.json) · 영상: [sparse](../outputs/ch05/reward_modes_300k_seed0/sparse/arm_reach_sparse.mp4) / [dense](../outputs/ch05/reward_modes_300k_seed0/dense/arm_reach_dense.mp4) / [shaped](../outputs/ch05/reward_modes_300k_seed0/shaped/arm_reach_shaped.mp4)
