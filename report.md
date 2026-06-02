# 모티베이션 & 인트로

본 프로젝트의 목표는 18개 기지국에서 얻은 RTT 기반 거리 추정값 `d_hat`을 이용하여 각 사용자 단말의 2차원 위치를 추정하는 것이다. 입력은 사용자별 18차원 거리 벡터와 기지국 좌표이며, 출력은 각 사용자에 대한 `(x, y)` 위치이다. 이 문제는 단순히 거리 18개를 위치 2개로 바꾸는 회귀 문제가 아니라, 다수의 기지국이 만드는 고차원 거리 공간에서 일부 기지국의 오차, 거리 구간별 bias, 기하학적 배치의 불안정성이 동시에 작용하는 위치 추정 문제이다.

중간발표 단계에서는 크게 두 가지 방향의 실험을 진행하였다. 첫 번째는 Random Forest 기반 회귀 모델을 이용하여 거리 벡터에서 위치를 직접 예측하는 방법이었다. 이 방식은 전체적인 평균 성능은 안정적이었지만, 거리 구조를 직접 만족시키는 기하학적 제약을 충분히 반영하지 못한다는 한계가 있었다. 특히 18개 기지국에서 만들어지는 feature 수가 많아질수록 모델은 물리적 거리 방정식보다 학습 데이터 안의 수치적 상관관계에 더 의존할 수 있고, hidden test에서 분포가 달라질 경우 과적합 위험이 생긴다.

두 번째는 거리 보정 후 멀티래터레이션을 수행하는 방식이었다. 이 방식은 기지국 좌표와 거리 방정식을 직접 사용하므로 물리적으로 해석 가능하다는 장점이 있었다. 그러나 다중 기지국 환경에서는 일부 기지국의 큰 오차가 전체 목적함수와 기하학적 해를 왜곡할 수 있다. 기지국이 많다는 것은 정보가 많다는 뜻이기도 하지만, 동시에 outlier가 목적함수에 들어올 가능성도 커진다는 의미이다. 따라서 단순 멀티래터레이션은 평균적으로는 합리적인 위치를 찾을 수 있지만, 특정 샘플에서는 큰 잔차를 가진 기지국 때문에 위치가 크게 흔들렸다.

중간 실험에서는 RF가 큰 실패 사례를 비교적 억제하는 반면, multilateration은 중앙값 성능을 개선하면서도 일부 샘플에서 큰 max error를 만들 수 있다는 상반된 결과가 관찰되었다. 이 관찰을 바탕으로 최종 알고리즘은 데이터 기반 예측과 기하학적 제약을 분리해서 경쟁시키는 것이 아니라, 두 계열의 장점을 결합하는 하이브리드 구조로 발전하였다. Random Forest는 거리 벡터와 위치 사이의 비선형 관계를 안정적으로 학습하고, robust multilateration은 보정 거리와 기지국 좌표가 만드는 물리적 제약을 반영한다. 이후 두 후보를 단순 평균하지 않고, 샘플별 상태에 따라 RF, multilateration, 규칙 기반 결합, teacher 기반 결합 중 적합한 후보를 선택하는 4-Way Expert Router를 사용하였다. 즉, 최종 구조는 18개 기지국의 고차원 입력에서 발생하는 데이터 기반 예측의 과적합 위험과 기하학 기반 해의 outlier 취약성을 동시에 완화하기 위해 설계되었다.

최종 구조는 크게 보정, RF 예측, robust multilateration, 결합 후보 생성, 후보 선택의 5단계로 이루어진다. 먼저 학습 데이터에서 기지국별 오차 profile을 만들고, 거리 구간별 bias를 보정한다. 이후 보정된 거리와 기지국 신뢰도 정보를 이용하여 Random Forest 위치 예측값을 만든다. 이 예측값을 초기 위치로 사용하여 Huber IRLS 기반의 robust multilateration을 수행하고, 잔차가 큰 기지국의 영향을 다시 낮추는 reweighting을 적용한다. 마지막으로 RF 예측값, robust multilateration 예측값, 규칙 기반 결합값, Counterfactual Gamma Teacher 결합값의 네 후보 중 하나를 4-Way Expert Router가 선택한다.

# 알고리즘 설명

입력으로 한 사용자에 대한 거리 추정 벡터를 다음과 같이 둔다.

`d = [d_1, d_2, ..., d_18]^T`

기지국 좌표는 다음과 같이 둔다.

`b_i = (x_i, y_i), i = 1, ..., 18`

추정할 사용자 위치는 다음과 같다.

`p = (x, y)`

## 1. 기지국별 오차 profile과 거리 구간 bias 보정

먼저 학습 데이터에서 기지국별 실제 거리와 측정 거리의 차이를 계산한다. 학습 데이터의 정답 위치를 `p_j`라고 하면, 기지국 `i`와 샘플 `j` 사이의 실제 거리는 다음과 같다.

`r_ij = ||p_j - b_i||_2`

측정 오차는 다음과 같이 정의한다.

`e_ij = d_ij - r_ij`

각 기지국에 대해 평균 오차, 중앙값 오차, 표준편차, 평균-중앙값 차이, 이상치 비율을 계산한다. 먼저 표준편차는 전체 기지국의 중앙 표준편차로 정규화한다.

`std_norm_i = std(e_i) / median_k(std(e_k))`

평균-중앙값 차이는 다음과 같이 계산한다.

`gap_i = |mean(e_i) - median(e_i)|`

이를 전체 기지국의 중앙 gap으로 정규화한다.

`gap_norm_i = gap_i / median_k(gap_k)`

이상치 비율은 기지국별 절대 오차 `|e_ij|`에 대해 IQR 기준으로 계산한다.

`threshold_i = Q3_i + 1.5 IQR_i`

`outlier_rate_i = mean_j(|e_ij| > threshold_i)`

이상치 비율도 0이 아닌 값들의 중앙값으로 나누어 정규화한다.

`outlier_norm_i = outlier_rate_i / median_k(outlier_rate_k)`

최종 이상치 점수는 평균-중앙값 차이와 이상치 비율을 같은 비중으로 결합한다.

`anomaly_i = 0.5 gap_norm_i + 0.5 outlier_norm_i`

기지국별 신뢰도는 오차 표준편차와 이상치 점수를 함께 반영하여 계산한다.

`reliability_i = 1 / (epsilon + std_norm_i + anomaly_i)`

이후 모든 신뢰도는 최대값이 1이 되도록 정규화한다.

단일 중앙값 bias만 제거하면 모든 거리 구간에 같은 보정이 적용되므로, 가까운 거리와 먼 거리에서 서로 다른 오차 패턴을 반영하기 어렵다. 이를 보완하기 위해 각 기지국의 측정 거리값을 기지국별로 독립적으로 6개의 quantile 구간으로 나누고, 구간별 중앙 오차를 계산한다. 구간 경계는 해당 기지국의 학습 데이터 측정 거리 `d_ij`의 0, 16.7, 33.3, 50, 66.7, 83.3, 100 percentile을 사용한다. 어떤 구간의 샘플 수가 20개보다 적으면 해당 구간의 bias 대신 기지국 전체 median error를 사용한다. 어떤 샘플의 측정 거리 `d_i`가 특정 구간에 속하면, 해당 구간에서 학습된 bias를 빼서 보정 거리를 만든다.

`d_i^c = max(d_i - bias_i(d_i), 0)`

여기서 `d_i^c`는 보정된 거리이고, `bias_i(d_i)`는 기지국 `i`에서 측정 거리 `d_i`가 속한 구간의 중앙 오차이다. 음수 거리는 물리적으로 의미가 없으므로 0 이상으로 제한한다.

## 2. Random Forest 입력 feature와 초기 위치 예측

Random Forest 모델의 입력 feature는 각 기지국에 대해 원래 거리, 보정 거리, 기지국 오차 불안정성, 이상치 점수, 기지국 신뢰도를 묶어서 만든다.

`x_i = [d_i, d_i^c, std_norm_i, anomaly_i, reliability_i]`

기지국이 18개이고 각 기지국마다 5개의 값이 있으므로 RF 입력 feature는 총 90차원이다.

`X_RF = [x_1, x_2, ..., x_18] \in \mathbb{R}^{90}`

Random Forest는 이 feature를 입력받아 초기 위치 `p_RF`를 예측한다.

`p_RF = f_RF(X_RF)`

이 단계는 거리 벡터와 위치 사이의 비선형 관계를 데이터 기반으로 학습하는 역할을 한다. 단, RF 예측값은 거리 방정식을 직접 만족하도록 강제되지 않으므로 이후 단계에서 기하학적 보정을 추가한다.

## 3. Huber IRLS 기반 robust multilateration

`p_RF`를 초기값으로 사용하여 robust multilateration을 수행한다. 일반적인 weighted multilateration은 다음 목적함수를 최소화하는 문제로 볼 수 있다.

`E(p) = 1/2 Σ_i w_i (||p - b_i||_2 - d_i^c)^2`

여기서 `w_i`는 기지국 `i`의 가중치이다. 본 알고리즘에서는 기지국의 기본 신뢰도와 현재 샘플에서의 거리 일관성을 함께 반영하여 가중치를 정한다. 현재 위치 후보 `p_RF`에서 각 기지국까지의 거리와 보정 거리의 차이를 비교한다.

`res_i = | ||p_RF - b_i||_2 - d_i^c |`

현재 샘플 기준 신뢰도는 잔차가 클수록 작아지도록 정한다. 여기서 `scale`은 고정 상수가 아니라 현재 샘플의 유효 기지국 잔차 분포에서 계산되는 값이다.

`scale = percentile_75({res_i | d_i is valid}) + epsilon`

`current_i = 1 / (1 + res_i / scale)`

따라서 초기 가중치는 다음과 같다.

`w_i = reliability_i × current_i`

단순 squared loss는 큰 잔차에 민감하다. 따라서 본 알고리즘은 Huber loss를 적용한다. 위치 `p`에서의 거리 잔차를 다음과 같이 둔다.

`r_i(p) = ||p - b_i||_2 - d_i^c`

MAD 기반 scale을 `σ`라고 할 때 정규화 잔차는 다음과 같다.

`u_i = r_i(p) / σ`

Huber loss는 다음과 같이 정의된다.

`rho_delta(u_i) = 1/2 u_i^2, if |u_i| <= delta`

`rho_delta(u_i) = delta(|u_i| - 1/2 delta), if |u_i| > delta`

이에 대응되는 IRLS weight는 다음과 같다.

`h_i = 1, if |u_i| <= delta`

`h_i = delta / |u_i|, if |u_i| > delta`

최종 반복 가중치는 기존 기지국 가중치와 Huber weight의 곱이다.

`w_i^total = w_i h_i`

따라서 각 반복에서 최소화하는 목적함수는 다음과 같이 쓸 수 있다.

`E_H(p) = 1/2 Σ_i w_i^total r_i(p)^2`

위치 `p`에 대한 잔차의 gradient는 다음과 같다.

`∂r_i(p)/∂p = (p - b_i) / ||p - b_i||_2`

따라서 전체 목적함수의 gradient는 다음과 같다.

`∇E_H(p) = Σ_i w_i^total r_i(p) (p - b_i) / ||p - b_i||_2`

각 반복에서 이 gradient를 이용해 위치를 갱신한다.

`p^(t+1) = p^(t) - eta_t ∇E_H(p^(t))`

여기서 `eta_t`는 Armijo line search로 선택한다. 즉, 너무 큰 step으로 목적함수가 증가하지 않도록 step size를 줄여가며 안정적인 갱신만 허용한다. 반복은 최대 12회 수행하며, `||∇E_H(p)||_2 < 1e-5`이거나 `||p^(t+1) - p^(t)||_2 < 1e-5`이면 조기 종료한다.

## 4. Residual-based anchor reweighting

Huber IRLS 이후에도 특정 기지국의 잔차가 위치를 불안정하게 만들 수 있으므로 residual-based anchor reweighting을 적용한다. 현재 `p_ML`에서 각 기지국의 잔차와 기존 weight를 함께 사용하여 influence를 계산한다.

`influence_i = | ||p_ML - b_i||_2 - d_i^c | × normalized_weight_i`

influence가 큰 기지국은 위치 추정에 과도한 영향을 줄 가능성이 있으므로 weight를 낮춘다. 여기서 `scale_influence`도 고정 상수가 아니라 현재 샘플에서 유효한 influence 값들의 75 percentile로 계산한다.

`scale_influence = percentile_75({influence_i | i is valid}) + epsilon`

`factor_i = 1 / (1 + influence_i / scale_influence)`

`w_i^refined = w_i × factor_i`

조정된 weight로 multilateration을 다시 수행한 뒤, 기존 ML 위치와 refined 위치를 비교한다. 기존 위치의 평균 거리 잔차를 `R_base`, refined 위치의 평균 거리 잔차를 `R_refined`, RF에서 기존 ML까지의 이동량을 `M_base`, RF에서 refined 위치까지의 이동량을 `M_refined`라고 하면 채택 조건은 다음과 같다.

`accept = (R_refined <= 1.02 R_base) or (M_refined <= 0.95 M_base)`

즉, refined 위치가 평균 거리 잔차를 거의 악화시키지 않거나, RF 기준 이동량을 충분히 줄이면 더 안정적인 해로 보고 채택한다. 단, reweighting 전 단계에서 이미 `R_ML < 0.92 R_RF`이고 `||p_ML - p_RF||_2 < 10`이면, ML 해가 충분히 안정적이라고 보고 추가 재계산은 생략한다. 이 결과를 `p_ML`이라고 둔다. 여기서 ML은 machine learning이 아니라 multilateration을 의미한다.

## 5. Rule Gamma와 Counterfactual Gamma Teacher

RF와 multilateration 결과를 결합하기 위해 규칙 기반 결합 후보를 만든다.

`p_rule = \gamma_{rule} p_RF + (1 - \gamma_{rule}) p_ML`

`\gamma_{rule}`은 multilateration이 RF보다 거리 잔차를 얼마나 줄였는지, 그리고 `p_RF`에서 `p_ML`로 이동한 거리가 지나치게 크지 않은지에 따라 정한다. RF 위치에서의 거리 잔차를 `R_RF`, multilateration 위치에서의 거리 잔차를 `R_ML`, 두 후보 사이의 이동 거리를 `m = ||p_ML - p_RF||_2`라고 하면 규칙은 다음과 같다.

| 조건 | `\gamma_{rule}` | 의미 |
|---|---:|---|
| `R_ML < 0.85 R_RF` and `m < 15` | 0.5 | ML을 강하게 반영 |
| `R_ML < 0.95 R_RF` and `m < 25` | 0.7 | ML을 일부 반영 |
| otherwise | 0.9 | RF 중심으로 보수적 결합 |

따라서 multilateration이 잔차를 충분히 줄이고 이동량도 과도하지 않을 때만 `p_ML`의 비중을 키우며, 그렇지 않으면 RF의 비중을 크게 유지한다.

Counterfactual Gamma Teacher는 train 내부 OOF 방식으로 학습된다. 각 샘플에 대해 `p_RF`와 `p_ML` 사이의 선분 위에서 실제 정답 위치에 가장 가까운 최적 `\gamma`를 계산한다.

`p(gamma) = gamma p_RF + (1 - gamma) p_ML`

최적 `\gamma*`는 다음 최적화 문제의 해이다.

`\gamma* = argmin_{0 \leq \gamma \leq 1} ||p_true - (\gamma p_RF + (1 - \gamma)p_ML)||_2`

이를 전개하면 `v = p_RF - p_ML`에 대해 다음과 같이 계산할 수 있다.

`\gamma* = clip(((p_true - p_ML) \cdot v) / (v \cdot v), 0, 1)`

단, 이 정답 위치는 train 내부 OOF 학습에서만 사용된다. 최종 추론에서는 hidden test의 정답 위치를 사용하지 않는다. Gamma Teacher는 Random Forest Regressor와 Random Forest Classifier의 결합으로 구현하였다. Gamma Regressor는 250개 tree, 최대 깊이 5, leaf 최소 샘플 수 8을 사용하며, 상태 feature `z`를 입력받아 원시 결합계수 `\gamma_{raw}`를 예측한다.

`\gamma_{raw} = f_{gamma}(z)`

또한 Trust Classifier는 같은 상태 feature `z`를 입력받아 ML 방향의 보정을 신뢰해도 되는 확률을 예측한다. train 내부 OOF에서 `\gamma* < 0.97`인 샘플을 ML 반영이 유효한 샘플로 두고 이진 label을 만든다. 이때 Trust Classifier의 출력 확률을 `trust_prob`로 정의한다.

`trust_prob = P(\gamma* < 0.97 | z)`

최종 Teacher 결합계수는 `\gamma_{raw}`와 `trust_prob`를 함께 사용하여 계산한다.

`\gamma_{teacher} = 1 - trust_prob(1 - \gamma_{raw})`

따라서 `trust_prob`가 작으면 RF에 가까운 보수적 결합이 되고, `trust_prob`가 크면 `\gamma_{raw}`에 가까운 결합이 된다. 이 값을 이용해 다음 후보를 만든다.

`p_teacher = \gamma_{teacher} p_RF + (1 - \gamma_{teacher})p_ML`

## 6. 4-Way Expert Router와 hidden test 독립성

최종 후보는 총 네 개이다.

| 후보 | 의미 |
|---|---|
| RF | Random Forest가 직접 예측한 위치 |
| ML | Huber IRLS 기반 robust multilateration 위치 |
| Rule | 규칙 기반으로 RF와 ML을 결합한 위치 |
| Teacher | Counterfactual Gamma Teacher가 예측한 gamma로 결합한 위치 |

Router는 네 후보 중 어떤 후보를 최종 위치로 사용할지 분류한다. Router는 Random Forest Classifier로 구현하였으며, 350개 tree, 최대 깊이 7, leaf 최소 샘플 수 6을 사용한다. 중요한 점은 Router가 hidden test 추론 시점에 정답 위치를 사용하지 않는다는 것이다. Router 학습에서는 train 내부 OOF 결과를 이용해 네 후보 중 정답에 가장 가까운 후보를 label로 만들지만, 추론 시에는 현재 입력 거리와 네 후보의 관계만 사용한다.

Router의 입력은 RF의 90차원 입력 feature와 구분된다. RF는 `X_RF \in \mathbb{R}^{90}`을 사용하지만, Router는 현재 샘플에서 계산 가능한 상태 feature와 후보 간 discrepancy feature를 사용한다. 상태 feature `z`는 다음 17개 값으로 구성된다.

| 상태 feature | 의미 |
|---|---|
| `rf_uncertainty` | RF tree 예측 분산 기반 불확실도 |
| `rf_residual` | `p_RF`에서의 평균 거리 잔차 |
| `ml_residual` | `p_ML`에서의 평균 거리 잔차 |
| `residual_ratio` | `ml_residual / rf_residual` |
| `residual_gain` | `rf_residual - ml_residual` |
| `move_distance` | `||p_ML - p_RF||_2` |
| `n_effective` | IRLS weight의 유효 기지국 수 |
| `sigma` | MAD 기반 잔차 scale |
| `n_valid` | 유효 기지국 수 |
| `weight_entropy` | 기지국 weight 분포의 entropy |
| `max_weight` | 최대 기지국 weight |
| `weight_sum` | 기지국 weight 총합 |
| `geometry_condition` | 기지국 방향 행렬의 log condition 값 |
| `ml_improved_flag` | ML이 초기 loss를 악화시키지 않았는지 여부 |
| `jackknife_mean_influence` | 기지국 influence 평균 |
| `jackknife_max_influence` | 기지국 influence 최댓값 |
| `jackknife_accepted_flag` | reweighting 결과 채택 여부 |

후보 간 discrepancy feature는 네 후보 `p_RF`, `p_ML`, `p_rule`, `p_teacher` 사이의 유클리드 거리 차이로 만든다. 네 후보 사이에는 총 6개의 pairwise distance가 존재한다.

`D_ab = ||p_a - p_b||_2, a,b in {RF, ML, Rule, Teacher}, a < b`

여기에 `\gamma_{rule}`, `\gamma_{teacher}`, `\gamma_{raw}`, `trust_prob`, 후보 spread의 평균, 최댓값, 표준편차를 추가한다. 여기서 `\gamma_{raw}`는 Gamma Regressor가 직접 예측한 원시 결합계수이고, `trust_prob`는 Trust Classifier가 예측한 ML 보정 신뢰 확률이다. 따라서 Router의 추론 시 입력은 다음과 같이 표현할 수 있다.

`X_{router} = [z, \gamma\ values, candidate\ pairwise\ distances, spread\ statistics]`

이 값들은 모두 hidden test에서 제공되는 `d`, `b_i`, 저장된 모델 `model.pkl`, 그리고 네 후보 예측값으로부터 계산된다. 즉, hidden test 정답 `p_true`는 포함되지 않는다.

`X_{router} = \phi(d, {b_i}, p_RF, p_ML, p_rule, p_teacher)`

`p_{true} \notin X_{router}`

학습 시 label은 다음과 같이 train 내부 OOF에서만 만든다.

`y_{router} = argmin_k ||p_{true} - p_k||_2, k \in {RF, ML, Rule, Teacher}`

하지만 추론 시에는 학습된 classifier가 다음과 같이 후보를 선택한다.

`k_hat = g_{router}(X_{router})`

`p_hat = p_k_hat`

## 7. 참고문헌 기반 이론과 본 구현의 차이

본 프로젝트의 참고문헌은 알고리즘을 그대로 가져오기 위해 사용한 것이 아니라, RTT 기반 실내 측위 문제를 해결하는 과정에서 필요한 이론적 근거를 확인하기 위해 사용하였다. 실제 구현에서는 제공된 DH_FR1.mat 데이터의 구조, 18개 기지국 배치, public/hidden test 조건, 제출용 main.py 규격에 맞게 각 방법을 변형하였다. 최종 코드는 train.py에서 public 데이터로 model.pkl을 학습하고, main.py에서 model.pkl을 불러와 hidden test의 d_hat과 BS_positions만으로 p_hat을 반환하는 구조로 작성하였다. 특히 model.pkl에는 rf_model, anchor profile, gamma_teacher, expert_router, settings가 함께 저장되도록 구성하였다. :contentReference[oaicite:0]{index=0}

[1] P. J. Huber, “Robust Estimation of a Location Parameter,” The Annals of Mathematical Statistics, 1964.

Huber의 논문에서는 이상치가 포함된 데이터에서 평균이나 일반적인 least squares 추정이 크게 흔들릴 수 있다는 문제를 다루고, 큰 잔차의 영향력을 제한하는 robust estimation의 기본 아이디어를 제시한다. 본 프로젝트에서는 이 논문의 핵심 아이디어인 “작은 잔차는 일반적인 제곱 오차처럼 처리하고, 큰 잔차는 가중치를 낮추어 영향력을 제한한다”는 점을 참고하였다.

본 프로젝트에서 직접 수행한 부분은 이 robust estimation 개념을 RTT 거리 기반 위치 추정 문제에 맞게 바꾼 것이다. 먼저 각 UE 후보 위치 p에 대해 anchor i와의 예측 거리 ||p - b_i||와 보정 거리 d_i^c의 차이를 잔차로 정의하였다. 이후 MAD 기반 scale을 이용해 잔차를 정규화하고, Huber weight를 계산하여 큰 잔차를 가진 anchor의 영향이 줄어들도록 하였다. 단순히 Huber loss만 적용한 것이 아니라, 기지국별 reliability와 현재 샘플에서의 거리 일관성 weight를 함께 곱해 total weight를 만들었다. 또한 residual-based reweighting을 추가하여 특정 anchor가 위치해를 과도하게 흔드는 경우 한 번 더 가중치를 낮추도록 설계하였다.

즉, [1]에서 참고한 것은 Huber loss의 robust한 잔차 처리 원리이고, 본 프로젝트에서 직접 한 일은 이를 18개 RTT anchor의 거리 잔차, MAD scale, anchor reliability, residual-based influence reweighting과 결합하여 Huber IRLS 기반 robust multilateration으로 구현한 것이다. 보고서의 Huber IRLS, 잔차 기반 재가중, tail risk 분석은 이 참고문헌의 개념을 본 과제 환경에 맞게 확장한 결과이다. :contentReference[oaicite:1]{index=1}

[2] G. Strang and K. Borre, Linear Algebra, Geodesy, and GPS, Wellesley-Cambridge Press, 1997.

Strang과 Borre의 책은 GPS와 측지학에서 위치를 추정할 때 선형대수, 최소제곱법, 거리 기반 위치 결정이 어떻게 사용되는지 설명한다. 본 프로젝트에서는 여러 기준점(anchor)과의 거리 정보를 동시에 만족하는 위치를 찾는 문제, 즉 multilateration의 기하학적 의미와 최소제곱 기반 위치 추정 관점을 이해하기 위해 이 문헌을 참고하였다. 특히 기지국 배치가 위치 추정 안정성에 영향을 준다는 점은 본 프로젝트의 geometry condition 설계와 연결된다.

본 프로젝트에서 직접 수행한 부분은 GPS 문제를 그대로 가져온 것이 아니라, 2차원 실내 RTT 환경에 맞게 바꾼 것이다. 제공된 데이터의 기지국 좌표 BS_positions 또는 p_bs를 사용하여 각 UE와 18개 anchor 사이의 실제 거리와 측정 거리 오차를 계산하였다. 그 다음 단순한 거리 원 교차 방식이 아니라, Random Forest가 예측한 p_RF를 초기 위치로 사용하고, 보정 거리 d_i^c와 anchor weight를 이용해 weighted nonlinear least squares 형태의 multilateration을 수행하였다. 또한 기지국 방향이 한쪽으로 몰리는 경우 위치 추정이 불안정해질 수 있으므로, p_RF에서 각 anchor 방향의 단위벡터를 만들고, weight가 반영된 2×2 방향 행렬 G의 조건수를 계산하였다. 이 값을 geometry_condition으로 정의하여 Router의 입력 feature로 사용하였다.

따라서 [2]에서 참고한 것은 거리 기반 위치 결정, least squares, 기하학적 안정성에 대한 이론적 배경이고, 본 프로젝트에서 직접 한 일은 이를 18개 실내 RTT anchor에 대해 2차원 weighted robust multilateration과 geometry_condition feature로 구현한 것이다. 이 과정에서 RF 초기값, Huber IRLS, anchor reliability, Router feature를 결합한 점은 본 프로젝트에서 직접 설계한 부분이다.

[3] T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning, Springer, 2009.

Hastie, Tibshirani, Friedman의 책은 supervised learning, ensemble learning, Random Forest, validation 기반 모델 평가의 기본 개념을 설명한다. 본 프로젝트에서는 거리 벡터와 좌표 사이의 비선형 관계를 학습하기 위해 Random Forest 회귀 모델을 사용하였고, 후보 선택 문제에는 Random Forest 기반 분류 모델을 사용하였다. 이때 [3]은 Random Forest가 여러 decision tree의 ensemble로 동작하고, 비선형 feature 관계를 비교적 안정적으로 학습할 수 있다는 이론적 배경으로 참고하였다.

본 프로젝트에서 직접 수행한 부분은 단순히 d_hat 18개를 RF에 넣은 것이 아니라, anchor별 feature를 새로 설계한 것이다. 각 anchor에 대해 원본 거리 d_i, 거리 구간별 bias가 제거된 보정 거리 d_i^c, std_norm, anomaly_score, reliability를 묶어 5개 feature를 만들었고, 18개 anchor에 대해 총 90차원 RF 입력을 구성하였다. 또한 RF를 최종 결과로 바로 사용하지 않고, p_RF라는 안정적인 초기 위치 후보로 사용하였다. 이후 p_RF를 기준으로 Huber IRLS를 수행해 p_ML을 만들고, Rule 후보와 Gamma Teacher 후보를 추가하였다.

Gamma Teacher와 4-Way Expert Router 역시 본 프로젝트에서 직접 구성한 부분이다. train 내부 OOF 샘플에서 p_RF와 p_ML 사이의 최적 결합 비율 gamma*를 계산하고, 이를 학습 label로 사용하여 Gamma Teacher를 만들었다. 또한 RF, ML, Rule, Teacher 네 후보 중 어느 후보가 정답에 가까운지를 train 내부 OOF 방식으로 label화하여 4-Way Expert Router를 학습하였다. 이 과정에서 residual_ratio, move_distance, geometry_condition, jackknife influence, 후보 간 spread와 같은 상태 feature를 직접 정의하였다. train.py에서는 Gamma Teacher와 Expert Router를 학습한 뒤, 최종적으로 rf_model, profile, gamma_teacher, expert_router를 model.pkl에 저장하도록 구현하였다. :contentReference[oaicite:2]{index=2}

즉, [3]에서 참고한 것은 Random Forest와 ensemble learning, supervised learning 평가 방식의 기본 개념이고, 본 프로젝트에서 직접 한 일은 RTT 측위 문제에 맞는 90차원 feature 설계, RF 초기 위치 예측, OOF 기반 Gamma Teacher label 생성, 4-Way Expert Router 학습, ablation study를 통한 성능 비교이다. 최종 main.py에서는 model.pkl을 불러와 profile, rf_model, gamma_teacher, expert_router를 사용하고, 각 UE에 대해 expert_pred를 계산한 뒤 shape (2, num_user)의 p_hat을 반환하도록 구성하였다. :contentReference[oaicite:3]{index=3}

종합적으로, 참고문헌 [1]은 이상치에 강한 잔차 처리와 Huber weight 설계의 근거로 사용하였고, [2]는 거리 기반 위치 추정과 anchor geometry 안정성 해석의 근거로 사용하였으며, [3]은 Random Forest와 ensemble 기반 지도학습 모델의 근거로 사용하였다. 본 프로젝트에서 직접 수행한 부분은 다음과 같다. 첫째, public RTT 데이터에서 기지국별 오차 profile을 만들고 평균 오차, median bias, 표준편차, 이상치 비율, reliability를 계산하였다. 둘째, 거리 구간별 median bias 보정을 적용하여 원본 RTT 거리값을 보정하였다. 셋째, 원본 거리, 보정 거리, anchor 불안정성, anomaly score, reliability를 결합한 90차원 RF feature를 설계하였다. 넷째, RF 예측값을 초기점으로 사용하는 Huber IRLS 기반 robust multilateration을 구현하였다. 다섯째, residual-based reweighting과 geometry_condition을 추가하여 불안정 anchor와 나쁜 기하 조건을 반영하였다. 여섯째, train 내부 OOF 방식으로 Gamma Teacher와 4-Way Expert Router를 학습하여 RF, ML, Rule, Teacher 후보 중 샘플별 최종 후보를 선택하도록 만들었다. 일곱째, hidden test에서 정답 p를 사용하지 않도록 train.py와 main.py를 분리하고, 제출용 main.py가 d_hat과 BS_positions, model.pkl만으로 p_hat을 반환하도록 구현하였다. :contentReference[oaicite:4]{index=4}

따라서 Router는 정답을 직접 참조하는 oracle selector가 아니라, train 내부에서 학습한 후보 선택 규칙을 hidden test의 관측 feature에 적용하는 분류기이다.

# Agent AI(e.g., ChatGPT, Claude Code, Gemini 등) 활용 방안

본 프로젝트에서는 ChatGPT를 보조 도구로 활용하였다. 활용 범위는 크게 알고리즘 아이디어 정리, 실험 결과 해석 보조, 코드 구조 점검, 보고서 문장 정리에 해당한다. 단, 최종 알고리즘의 방향 선택, 실험 실행, 결과 비교, 제출 파일 구성은 직접 수행하였다.

알고리즘 설계 단계에서는 중간 실험에서 나타난 문제를 설명하고, 가능한 개선 방향을 함께 검토하였다. 예를 들어 Random Forest 단독 방식은 안정적이지만 거리 제약 반영이 약하고, multilateration은 해석 가능하지만 이상치에 취약하다는 점을 바탕으로 두 계열을 결합하는 방향을 정리하였다. 이후 기지국별 bias 보정, Huber IRLS, residual-based reweighting, Gamma Teacher, Expert Router의 역할을 구분하는 데 ChatGPT를 사용하였다. 이 과정에서 ChatGPT는 가능한 구조와 설명 방식을 제안하였고, 실제 적용 여부는 validation 결과와 구현 가능성을 기준으로 판단하였다.

AI의 제안을 그대로 사용하지 않고, 데이터 누수 가능성과 채점 환경 적합성을 직접 검증하였다. 예를 들어 초기 아이디어 중에는 validation 정답 또는 hidden test 정답을 간접적으로 참조하는 selector처럼 해석될 위험이 있는 구조가 있었다. 이러한 방식은 성능 수치가 좋아 보여도 실제 제출 환경에서는 사용할 수 없으므로 제외하였다. 이를 막기 위해 Teacher와 Router의 label 생성은 train 내부 OOF 방식으로만 수행하도록 구조를 바꾸었다. 즉, AI가 제안한 결합 아이디어를 그대로 채택한 것이 아니라, hidden test에서 사용할 수 있는 정보와 사용할 수 없는 정보를 분리한 뒤, leakage가 발생하지 않는 형태로 재설계하였다.

코드 작성 단계에서는 오류 가능성이 높은 부분을 점검하는 용도로 ChatGPT를 사용하였다. 이때 전체 코드를 붙여넣고 과제에서 요구하는 함수 signature와 반환 형식을 함께 설명한 뒤, 제출 규격 위반 가능성을 확인하는 방식으로 질의하였다. 특히 제출 규격과 관련하여 `your_algorithm`의 인자 개수, `main()`의 반환 shape, 사용자 수를 `d_hat.shape[1]`에서 동적으로 받는지, hidden test의 정답 위치 `p`를 추론에 사용하지 않는지 등을 확인하였다. 또한 `main.py`가 `model.pkl`을 로드하여 추론만 수행하고, `train.py`가 학습을 담당하도록 역할을 분리하였다.

결과 정리 단계에서는 실험 결과를 단순히 나열하는 대신, 각 방법이 어떤 한계를 해결하기 위해 추가되었는지 설명하는 방향으로 보고서 구조를 잡는 데 도움을 받았다. ChatGPT가 제안한 문장은 최종 보고서 전체를 자동 생성하는 용도가 아니라, 알고리즘 흐름 정리, 결과 해석 문장, 표현 다듬기와 같은 부분 초안으로만 사용하였다. 본인은 프로젝트의 실제 실험 흐름과 맞지 않는 표현을 수정하고, 최종 수치와 알고리즘 설명이 코드와 일치하는지 확인하였다. 따라서 AI는 전체 과정을 자동으로 대신 수행한 것이 아니라, 아이디어 정리와 문서화, 규격 점검을 보조하는 역할로 사용되었다.

# 결과 도출 & 디스커션

평가는 public 데이터 내부를 random split으로 train 500개와 validation 200개로 나누어 수행하였다. 최종 제출 모델은 train.py에서 제공된 학습 데이터를 이용해 학습되며, main.py는 저장된 model.pkl과 hidden test의 d_hat, BS_positions만 사용하여 위치를 추론한다. 즉, hidden test의 정답 위치는 어떤 방식으로도 추론에 사용하지 않는다.

대표적인 validation 결과는 다음과 같다.

| 방법 | Mean Position Error | Median Position Error | 90% Position Error | Max Position Error |
|---|---:|---:|---:|---:|
| RF Only | 7.849851 | 6.730623 | 14.142477 | 23.781930 |
| ML Corrected Huber IRLS | 6.405236 | 4.816375 | 12.566320 | 46.830455 |
| Rule Gamma | 6.356719 | 5.542028 | 11.731842 | 23.691505 |
| Counterfactual Gamma Teacher | 6.081315 | 5.294196 | 11.281895 | 23.411367 |
| Selective Rule/Teacher | 5.892867 | 5.188806 | 10.768561 | 23.411367 |
| Final 4-Way Expert Router | 5.659821 | 4.604135 | 10.992494 | 22.825670 |

RF Only 대비 Final 4-Way Expert Router의 개선량은 다음과 같다.

| 비교 | Mean Position Error 감소량 | 개선율 |
|---|---:|---:|
| RF Only → Final 4-Way Expert Router | 2.190030 | 27.9% |

이 개선은 단순히 모델을 복잡하게 만든 것만으로 얻어진 결과가 아니라, 각 단계가 서로 다른 오차 원인을 보완했기 때문에 나타난 것으로 해석할 수 있다.

비교의 공정성을 위해 모든 방법은 같은 train/validation split과 같은 입력 거리 데이터를 사용하였다. RF Only는 최종 모델 내부에서 사용하는 Random Forest와 같은 feature profile 및 기본 하이퍼파라미터 계열을 사용한 기준 모델이며, ML Corrected Huber IRLS도 동일한 bias correction과 기지국 profile을 바탕으로 평가하였다. 따라서 단순히 약한 baseline을 세워 최종 모델이 좋아 보이도록 만든 비교가 아니라, 최종 구조의 각 구성요소를 같은 조건에서 단계적으로 추가한 ablation 성격의 비교이다.

RF Only는 전체적으로 안정적인 baseline이다. 거리 벡터와 위치 사이의 비선형 관계를 잘 학습하지만, 예측 결과가 실제 거리 방정식을 만족하는지 직접 확인하지 않는다. 반대로 ML Corrected Huber IRLS는 보정 거리와 기지국 좌표를 이용하여 기하학적 일관성을 반영한다. 이 방식은 median error가 RF보다 작아졌지만, max error가 크게 증가하였다. 이는 multilateration 계열 방법이 일부 불안정 샘플에서 큰 오차를 만들 수 있음을 보여준다. 따라서 단순히 RF보다 평균이 낮다는 이유로 multilateration만 선택하는 것은 위험하다.

이 극단 오차 원인 분석은 validation에서 max error가 커진 현상을 설명하기 위한 이론적 해석이다. 구체적으로는 다음 세 가지 상황에서 발생할 수 있다. 첫째, 여러 기지국 중 일부가 큰 거리 오차를 가지면 보정 후에도 거리 원들이 한 점에서 일관되게 만나지 않는다. 둘째, 유효 weight가 소수 기지국에 집중되면 `n_effective`가 낮아지고 특정 방향의 기하 정보가 약해져 위치해가 한쪽으로 밀릴 수 있다. 셋째, 기지국 방향 분포가 불안정하면 `geometry_condition`이 커지고, Huber IRLS가 큰 잔차를 줄이는 과정에서 실제 위치가 아니라 잔차가 작은 잘못된 교차 영역으로 수렴할 수 있다. 따라서 ML 후보는 중앙값 성능을 개선하는 장점이 있지만, tail sample에서는 반드시 RF나 Rule 후보와 함께 비교되어야 한다.

Rule Gamma와 Counterfactual Gamma Teacher는 RF와 ML의 장점을 결합하기 위한 단계이다. Rule Gamma는 사람이 정한 조건에 따라 결합 비율을 정하므로 해석 가능성이 높지만, 모든 샘플의 복잡한 상태를 충분히 반영하기 어렵다. Counterfactual Gamma Teacher는 train 내부 OOF에서 최적 결합 비율을 학습하므로 더 유연하게 동작한다. 결과적으로 Teacher는 Rule보다 평균 오차와 90% 오차를 줄였다. 이는 단순 규칙보다 데이터 기반 결합이 샘플별 상태를 더 잘 반영했음을 의미한다.

Final 4-Way Expert Router는 네 후보 중 하나를 선택한다. 이 방식은 평균 오차와 중앙값 오차를 가장 낮게 만들었다. 특히 median error가 4.604135로 가장 낮다는 점은, 전체 샘플 중 일반적인 경우에 최종 router가 가장 안정적으로 동작했음을 보여준다. 다만 90% error는 Selective Rule/Teacher보다 약간 높다. 이는 최종 Router가 다수의 정상 샘플에서 최적 후보를 선택하는 방향으로 평균과 중앙값을 줄이는 데 강점을 보였지만, 소수 극단 샘플에서는 후보 간 분류 경계가 모호해질 수 있음을 의미한다. 예를 들어 RF, ML, Rule, Teacher 후보가 서로 크게 벌어져 있고 잔차 지표도 동시에 불안정한 경우, Router가 정상 샘플에서 학습한 선택 규칙이 tail 샘플에는 완전히 맞지 않을 수 있다.

따라서 Final 4-Way Expert Router의 성능 향상은 tail risk를 완전히 제거한 결과라기보다, 평균적인 샘플과 중간 난이도 샘플에서 더 적합한 후보를 고르는 능력이 개선된 결과로 보는 것이 타당하다. 본 프로젝트의 평가는 전체 사용자에 대한 평균 위치 오차를 주요 기준으로 두는 상황을 가정했기 때문에, 90% error의 작은 증가를 감수하더라도 mean과 median이 동시에 개선된 것은 일반 성능을 높이는 관점에서 의미 있는 trade-off라고 판단하였다. 다만 안전이나 장애 대응처럼 극단적 오차가 더 중요한 실사용 환경이라면 mean보다 90% 또는 max error를 직접 최적화하는 별도 fallback이 필요하다.

baseline 비교의 fairness 측면에서, RF Only와 최종 모델은 동일한 입력 데이터와 동일한 train/validation split을 사용하였다. 또한 multilateration, Rule, Teacher, Router는 validation 정답을 직접 사용하여 조정하지 않고, train 내부 OOF 방식으로 teacher label과 router label을 생성하였다. 따라서 validation set은 최종 성능 확인용으로 남겨두었다. 딥러닝 모델과 단순 삼각측량을 직접 비교한 것이 아니라, 동일한 거리 입력과 동일한 학습 데이터 조건에서 단계별 후보를 비교했기 때문에 baseline 비교는 비교적 fair 하다고 볼 수 있다.

본 알고리즘의 장점은 세 가지이다. 첫째, 기지국별 신뢰도와 거리 구간별 bias를 반영하여 원시 거리값의 체계적 오차를 완화한다. 둘째, Random Forest와 robust multilateration을 함께 사용하여 데이터 기반 예측과 기하학적 해석을 모두 활용한다. 셋째, 모든 샘플에 같은 모델을 강제로 적용하지 않고, 4-Way Router를 통해 샘플별로 적합한 후보를 선택한다.

단점도 존재한다. 첫째, 구조가 복잡하여 단순 RF나 단순 multilateration에 비해 구현과 설명이 어렵다. 둘째, Router가 학습 데이터의 후보 선택 패턴에 의존하기 때문에 hidden test 분포가 크게 다르면 선택 성능이 떨어질 수 있다. 셋째, 90% error와 max error 관점에서는 아직 tail risk가 남아 있다. 특히 일부 샘플에서는 multilateration 후보가 크게 흔들릴 수 있고, router가 이를 항상 피하지는 못한다.

Future work로는 tail error를 줄이기 위한 conservative fallback을 추가할 수 있다. 예를 들어 후보 간 spread가 지나치게 크거나 기하 조건이 불안정한 경우에는 RF 또는 Rule처럼 더 보수적인 후보를 선택하도록 하는 방식이 가능하다. 또한 현재 평가는 random split 기반이므로, train/validation split 하나에만 의존하지 않고 여러 random split 또는 spatial split을 사용하여 위치 영역이 달라졌을 때도 성능이 안정적인지 확인할 필요가 있다. 마지막으로 현재는 샘플별 추론에서 일부 반복 계산이 존재하므로, hidden test 규모가 커질 경우 batch 처리와 벡터화를 통해 실행 시간을 더 줄일 수 있다.

# Reference

[1] P. J. Huber, “Robust Estimation of a Location Parameter,” The Annals of Mathematical Statistics, vol. 35, no. 1, pp. 73–101, 1964.

[2] G. Strang and K. Borre, Linear Algebra, Geodesy, and GPS, Wellesley-Cambridge Press, 1997.

[3] T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning, Springer, 2009.



