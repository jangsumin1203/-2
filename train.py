import numpy as np
import scipy.io as sio
import pickle
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold


# ============================================================
# 0. 설정
# ============================================================

MAT_PATH = "DH_FR1.mat"

RANDOM_SEED = 42

RF_N_ESTIMATORS_FINAL = 700
RF_N_ESTIMATORS_TEACHER = 180
GAMMA_MODEL_ESTIMATORS = 250
SELECTOR_MODEL_ESTIMATORS = 250
EXPERT_ROUTER_MODEL_ESTIMATORS = 350
OOF_SPLITS = 3

# 거리 구간별 bias 보정 설정
USE_BIN_BIAS_CORRECTION = True
BIAS_BIN_COUNT = 6
MIN_BIN_SAMPLES = 20

# Residual-based anchor influence 재가중 설정
# 아래 상수는 validation 200개를 보고 조정한 값이 아니라,
# train 내부 튜닝/보수적 기본값으로 고정한 알고리즘 설계 상수이다.
USE_JACKKNIFE_REWEIGHTING = True
JACKKNIFE_MAX_ITER = 8
JACKKNIFE_CANDIDATE_COUNT = 3
JACKKNIFE_RISK_RATIO = 0.92
JACKKNIFE_MOVE_THRESHOLD = 10.0
INFLUENCE_STRENGTH = 1.0
MIN_INFLUENCE_WEIGHT_FACTOR = 0.25

# Rule vs Teacher 선택기 설정
# selector도 validation 200개를 보지 않고 train 내부 OOF 데이터로만 학습한다.
SELECTOR_MARGIN = 0.0
SELECTOR_PROBA_THRESHOLD = 0.50
FAST_RF_UNCERTAINTY = True


# ============================================================
# 1. 데이터 불러오기
# ============================================================

def load_mat_data(mat_path):
    mat = sio.loadmat(mat_path)

    d_hat = mat["d_hat"]
    p = mat["p"]

    if "BS_positions" in mat:
        p_bs = mat["BS_positions"]
    elif "p_bs" in mat:
        p_bs = mat["p_bs"]
    else:
        raise KeyError("기지국 좌표 변수 BS_positions 또는 p_bs를 찾을 수 없습니다.")

    indices = mat["indices"].reshape(-1) if "indices" in mat else np.arange(d_hat.shape[1])

    return d_hat, p, p_bs, indices


# ============================================================
# 2. 실제 거리 계산
# ============================================================

def compute_true_distance(p, p_bs):
    d_true = np.linalg.norm(
        p_bs[:, :, None] - p[:, None, :],
        axis=0
    )
    return d_true


# ============================================================
# 3. 기지국별 오차 프로파일 생성
# ============================================================

def build_anchor_profile(d_hat_train, p_train, p_bs, eps=1e-6):
    d_true_train = compute_true_distance(p_train, p_bs)

    error = d_hat_train - d_true_train
    abs_error = np.abs(error)

    mean_error = np.mean(error, axis=1)
    median_error = np.median(error, axis=1)
    std_error = np.std(error, axis=1)
    gap_error = np.abs(mean_error - median_error)
    mae_error = np.mean(abs_error, axis=1)

    q1 = np.percentile(abs_error, 25, axis=1)
    q3 = np.percentile(abs_error, 75, axis=1)
    iqr = q3 - q1

    outlier_threshold = q3 + 1.5 * iqr
    outlier_rate = np.mean(abs_error > outlier_threshold[:, None], axis=1)

    std_norm = std_error / (np.median(std_error) + eps)

    if np.median(gap_error) < eps:
        gap_norm = gap_error
    else:
        gap_norm = gap_error / (np.median(gap_error) + eps)

    nonzero_outlier = outlier_rate[outlier_rate > 0]

    if len(nonzero_outlier) > 0:
        outlier_norm = outlier_rate / (np.median(nonzero_outlier) + eps)
    else:
        outlier_norm = outlier_rate

    anomaly_score = 0.5 * gap_norm + 0.5 * outlier_norm

    reliability = 1.0 / (eps + std_norm + anomaly_score)
    reliability = reliability / (np.max(reliability) + eps)

    # --------------------------------------------------------
    # 기지국별 거리 구간 bias 보정용 프로파일
    # 기존의 단일 median 보정은 모든 거리 구간에 같은 bias를 적용한다.
    # 여기서는 train 데이터에서 측정거리 구간별 median bias를 학습한다.
    # validation/test 정답은 사용하지 않는다.
    # --------------------------------------------------------
    bin_edges = []
    bin_bias = []

    for i in range(d_hat_train.shape[0]):
        measured_i = np.asarray(d_hat_train[i], dtype=float)
        error_i = np.asarray(error[i], dtype=float)
        valid_i = np.isfinite(measured_i) & np.isfinite(error_i)

        if np.sum(valid_i) < max(MIN_BIN_SAMPLES, BIAS_BIN_COUNT):
            edges_i = np.array([-np.inf, np.inf], dtype=float)
            bias_i = np.array([median_error[i]], dtype=float)
        else:
            q = np.linspace(0, 100, BIAS_BIN_COUNT + 1)
            edges_i = np.percentile(measured_i[valid_i], q)
            edges_i = np.unique(edges_i)

            if len(edges_i) < 3:
                edges_i = np.array([-np.inf, np.inf], dtype=float)
                bias_i = np.array([median_error[i]], dtype=float)
            else:
                edges_i[0] = -np.inf
                edges_i[-1] = np.inf
                num_bins = len(edges_i) - 1
                bias_vals = []

                for b in range(num_bins):
                    in_bin = (measured_i >= edges_i[b]) & (measured_i < edges_i[b + 1]) & valid_i
                    if np.sum(in_bin) >= MIN_BIN_SAMPLES:
                        bias_vals.append(float(np.median(error_i[in_bin])))
                    else:
                        bias_vals.append(float(median_error[i]))

                bias_i = np.array(bias_vals, dtype=float)

        bin_edges.append(edges_i)
        bin_bias.append(bias_i)

    profile = {
        "mean_error": mean_error,
        "median_error": median_error,
        "std_error": std_error,
        "gap_error": gap_error,
        "mae_error": mae_error,
        "outlier_rate": outlier_rate,
        "outlier_threshold": outlier_threshold,
        "std_norm": std_norm,
        "gap_norm": gap_norm,
        "outlier_norm": outlier_norm,
        "anomaly_score": anomaly_score,
        "reliability": reliability,
        "bin_edges": bin_edges,
        "bin_bias": bin_bias,
        "use_bin_bias_correction": USE_BIN_BIAS_CORRECTION
    }

    return profile


# ============================================================
# 4. 90차원 요약 특성 생성
# ============================================================

def _lookup_bin_bias(values, edges, biases, default_bias):
    values = np.asarray(values, dtype=float)

    if edges is None or biases is None or len(biases) == 0:
        return np.full_like(values, float(default_bias), dtype=float)

    edges = np.asarray(edges, dtype=float)
    biases = np.asarray(biases, dtype=float)

    if len(edges) < 2 or len(biases) != len(edges) - 1:
        return np.full_like(values, float(default_bias), dtype=float)

    # edges[0], edges[-1]은 -inf, inf로 설정되어 있으므로 내부 경계만 사용한다.
    bin_idx = np.searchsorted(edges[1:-1], values, side="right")
    bin_idx = np.clip(bin_idx, 0, len(biases) - 1)

    out = biases[bin_idx]
    out[~np.isfinite(values)] = float(default_bias)

    return out


def make_corrected_distance_matrix(d_hat, profile):
    d_hat = np.asarray(d_hat, dtype=float)
    corrected = np.zeros_like(d_hat, dtype=float)

    use_bins = profile.get("use_bin_bias_correction", False)
    bin_edges = profile.get("bin_edges", None)
    bin_bias = profile.get("bin_bias", None)
    median_error = profile["median_error"]

    for i in range(d_hat.shape[0]):
        if use_bins and bin_edges is not None and bin_bias is not None:
            bias_i = _lookup_bin_bias(
                values=d_hat[i],
                edges=bin_edges[i],
                biases=bin_bias[i],
                default_bias=median_error[i]
            )
        else:
            bias_i = median_error[i]

        corrected[i] = d_hat[i] - bias_i

    corrected = np.maximum(corrected, 0.0)
    return corrected


def make_profile_features(d_hat, profile):
    num_anchor, num_user = d_hat.shape

    std_norm = profile["std_norm"].reshape(-1, 1)
    anomaly_score = profile["anomaly_score"].reshape(-1, 1)
    reliability = profile["reliability"].reshape(-1, 1)

    original_distance = d_hat
    corrected_distance = make_corrected_distance_matrix(d_hat, profile)

    instability_feature = np.repeat(std_norm, num_user, axis=1)
    anomaly_feature = np.repeat(anomaly_score, num_user, axis=1)
    reliability_feature = np.repeat(reliability, num_user, axis=1)

    feature_stack = np.stack(
        [
            original_distance,
            corrected_distance,
            instability_feature,
            anomaly_feature,
            reliability_feature
        ],
        axis=2
    )

    X = feature_stack.transpose(1, 0, 2).reshape(num_user, num_anchor * 5)

    return X


# ============================================================
# 5. 보정 거리 생성
# ============================================================

def make_corrected_distance(d_single, profile):
    d_single = np.asarray(d_single, dtype=float).reshape(-1)
    d_mat = d_single.reshape(-1, 1)
    return make_corrected_distance_matrix(d_mat, profile).reshape(-1)


# ============================================================
# 6. RF 모델 및 불확실도
# ============================================================

def make_rf_model(random_seed=RANDOM_SEED, n_estimators=RF_N_ESTIMATORS_FINAL):
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=2,
                    max_features=0.8,
                    bootstrap=True,
                    random_state=random_seed,
                    n_jobs=-1
                )
            )
        ]
    )
    return model


def predict_rf_with_uncertainty(model, X_single):
    p_rf = model.predict(X_single)[0]

    # 속도 개선: tree별 분산 계산은 700개 tree를 샘플마다 반복 호출하므로 매우 느릴 수 있다.
    # 개발용 정밀 보정 실험에서는 RF 불확실도 대신 0을 사용하고,
    # 나머지 residual/geometry/weight 지표로 gate를 판단한다.
    if FAST_RF_UNCERTAINTY:
        return p_rf, 0.0

    try:
        imputer = model.named_steps.get("imputer", None)
        rf = model.named_steps["rf"]

        if imputer is not None:
            X_processed = imputer.transform(X_single)
        else:
            X_processed = X_single

        tree_predictions = np.array([
            tree.predict(X_processed)[0]
            for tree in rf.estimators_
        ])

        rf_uncertainty = float(np.mean(np.var(tree_predictions, axis=0)))

    except Exception:
        rf_uncertainty = 0.0

    return p_rf, rf_uncertainty



def predict_rf_batch_with_uncertainty(model, X_batch):
    """
    RF batch 예측을 유지하면서 tree별 예측 분산으로 샘플별 RF 불확실도를 계산한다.
    p_rf_batch shape: (n_samples, 2)
    rf_uncertainty_batch shape: (n_samples,)
    """
    p_rf_batch = model.predict(X_batch)

    try:
        imputer = model.named_steps.get("imputer", None)
        rf = model.named_steps["rf"]

        if imputer is not None:
            X_processed = imputer.transform(X_batch)
        else:
            X_processed = X_batch

        tree_predictions = np.array([
            tree.predict(X_processed)
            for tree in rf.estimators_
        ])

        # tree_predictions shape = (n_trees, n_samples, 2)
        # 각 샘플별로 x/y 좌표 예측 분산의 평균을 uncertainty로 사용
        rf_uncertainty_batch = np.mean(
            np.var(tree_predictions, axis=0),
            axis=1
        )

        rf_uncertainty_batch = np.asarray(rf_uncertainty_batch, dtype=float)
        rf_uncertainty_batch[~np.isfinite(rf_uncertainty_batch)] = 0.0

    except Exception as e:
        print(f"RF uncertainty batch 계산 실패: {e}")
        rf_uncertainty_batch = np.zeros(X_batch.shape[0], dtype=float)

    return p_rf_batch, rf_uncertainty_batch



# ============================================================
# 7. 현재 샘플 기준 기지국 신뢰도 계산
# ============================================================

def compute_current_anchor_weights(d_single, p_rf, p_bs, profile, eps=1e-6):
    base_reliability = profile["reliability"].astype(float)

    rf_distance = np.linalg.norm(
        p_rf.reshape(2, 1) - p_bs,
        axis=0
    )

    consistency_error = np.abs(rf_distance - d_single)

    valid = np.isfinite(d_single) & np.isfinite(consistency_error)

    if np.sum(valid) < 3:
        return np.ones_like(d_single) * 0.1

    scale = np.percentile(consistency_error[valid], 75) + eps
    current_reliability = 1.0 / (1.0 + consistency_error / scale)

    weights = base_reliability * current_reliability
    weights[~valid] = 0.0
    weights = np.clip(weights, 0.0, 1.0)

    if np.sum(weights > 0.05) < 4:
        best_idx = np.argsort(consistency_error)[:6]
        weights[:] = 0.0
        weights[best_idx] = base_reliability[best_idx]

    return weights


# ============================================================
# 8. MAD, Huber IRLS, Armijo 함수
# ============================================================

def mad_scale(residuals, eps=1e-6):
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    if len(residuals) == 0:
        return 1.0

    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median))

    sigma = 1.4826 * mad

    if (not np.isfinite(sigma)) or sigma < eps:
        sigma = np.percentile(np.abs(residuals), 75) + eps

    return max(float(sigma), eps)


def huber_irls_weight(normalized_residual, delta=1.345, eps=1e-6):
    abs_u = np.abs(normalized_residual)

    weights = np.where(
        abs_u <= delta,
        1.0,
        delta / (abs_u + eps)
    )

    return weights


def weighted_loss(position, anchors, distances, weights):
    pred_dist = np.linalg.norm(
        position.reshape(1, 2) - anchors,
        axis=1
    )

    residual = pred_dist - distances

    return 0.5 * np.sum(weights * residual ** 2)


def weighted_gradient(position, anchors, distances, weights, eps=1e-9):
    diff = position.reshape(1, 2) - anchors
    pred_dist = np.linalg.norm(diff, axis=1)

    safe_dist = np.maximum(pred_dist, eps)
    residual = pred_dist - distances

    jacobian = diff / safe_dist.reshape(-1, 1)

    gradient = np.sum(
        (weights * residual).reshape(-1, 1) * jacobian,
        axis=0
    )

    return gradient


def armijo_line_search(
    position,
    gradient,
    anchors,
    distances,
    weights,
    eta_init=1.0,
    beta=0.5,
    c=1e-4,
    eta_min=1e-3
):
    eta = eta_init

    current_loss = weighted_loss(position, anchors, distances, weights)
    grad_norm_sq = np.dot(gradient, gradient)

    if grad_norm_sq < 1e-12:
        return 0.0

    while eta > eta_min:
        candidate = position - eta * gradient
        candidate_loss = weighted_loss(candidate, anchors, distances, weights)

        if candidate_loss <= current_loss - c * eta * grad_norm_sq:
            return eta

        eta *= beta

    return eta_min


# ============================================================
# 9. 로버스트 멀티래터레이션: Huber IRLS
# ============================================================

def robust_multilateration(
    d_single,
    p_bs,
    p_init,
    initial_weights,
    max_iter=12,
    tol=1e-5,
    eps=1e-6
):
    valid = (
        np.isfinite(d_single)
        & np.isfinite(initial_weights)
        & (initial_weights > 0)
    )

    if np.sum(valid) < 3:
        info = {
            "mean_abs_residual": np.nan,
            "n_valid": int(np.sum(valid)),
            "n_effective": float(np.sum(valid)),
            "sigma": np.nan,
            "n_iter": 0,
            "improved": False
        }
        return p_init, info

    anchors = p_bs[:, valid].T
    distances = d_single[valid]
    base_weights = initial_weights[valid].astype(float)

    position = np.asarray(p_init, dtype=float).copy()

    if not np.all(np.isfinite(position)):
        position = np.mean(anchors, axis=0)

    initial_loss = weighted_loss(position, anchors, distances, base_weights)

    last_sigma = 1.0
    last_weights = base_weights.copy()
    n_iter_done = 0

    for it in range(max_iter):
        pred_dist = np.linalg.norm(
            position.reshape(1, 2) - anchors,
            axis=1
        )

        residual = pred_dist - distances

        sigma = mad_scale(residual, eps=eps)
        last_sigma = sigma

        normalized_residual = residual / sigma
        irls_weights = huber_irls_weight(normalized_residual)

        total_weights = base_weights * irls_weights
        total_weights = np.clip(total_weights, 1e-4, 1.0)
        last_weights = total_weights

        gradient = weighted_gradient(
            position,
            anchors,
            distances,
            total_weights
        )

        if np.linalg.norm(gradient) < tol:
            n_iter_done = it + 1
            break

        eta = armijo_line_search(
            position,
            gradient,
            anchors,
            distances,
            total_weights
        )

        new_position = position - eta * gradient
        step_size = np.linalg.norm(new_position - position)

        position = new_position
        n_iter_done = it + 1

        if step_size < tol:
            break

    final_loss_base = weighted_loss(position, anchors, distances, base_weights)

    if final_loss_base > initial_loss * 1.05:
        position = np.asarray(p_init, dtype=float).copy()
        improved = False
    else:
        improved = True

    final_pred_dist = np.linalg.norm(
        position.reshape(1, 2) - anchors,
        axis=1
    )

    final_residual = final_pred_dist - distances

    n_effective = (np.sum(last_weights) ** 2) / (
        np.sum(last_weights ** 2) + eps
    )

    info = {
        "mean_abs_residual": float(np.mean(np.abs(final_residual))),
        "n_valid": int(np.sum(valid)),
        "n_effective": float(n_effective),
        "sigma": float(last_sigma),
        "n_iter": int(n_iter_done),
        "improved": improved
    }

    return position, info


def simple_mean_abs_distance_residual(position, d_single, p_bs):
    valid = np.isfinite(d_single)
    if np.sum(valid) < 3:
        return np.inf

    dist = np.linalg.norm(position.reshape(2, 1) - p_bs, axis=0)
    return float(np.mean(np.abs(dist[valid] - d_single[valid])))


def residual_based_anchor_reweighting(
    d_single,
    p_bs,
    p_rf,
    p_ml,
    initial_weights,
    ml_info,
    eps=1e-6
):
    """
    계산량을 줄인 기지국 영향도 재가중 방식.
    엄밀한 leave-one-out을 모든 기지국에 반복하지 않고,
    현재 ML 위치에서의 잔차 크기와 기지국 weight를 결합해
    해당 기지국이 위치해를 흔들 가능성을 근사 influence로 계산한다.
    이후 influence가 큰 기지국의 weight를 낮추고 Huber IRLS를 1회 재수행한다.
    """
    if not USE_JACKKNIFE_REWEIGHTING:
        return p_ml, ml_info, initial_weights, {
            "accepted": False,
            "mean_influence": 0.0,
            "max_influence": 0.0
        }

    valid = (
        np.isfinite(d_single)
        & np.isfinite(initial_weights)
        & (initial_weights > 0)
    )

    if np.sum(valid) < 4:
        info_out = dict(ml_info)
        info_out["jackknife_mean_influence"] = 0.0
        info_out["jackknife_max_influence"] = 0.0
        info_out["jackknife_accepted"] = False
        return p_ml, info_out, initial_weights, {
            "accepted": False,
            "mean_influence": 0.0,
            "max_influence": 0.0
        }

    rf_residual0 = simple_mean_abs_distance_residual(p_rf, d_single, p_bs)
    ml_residual0 = simple_mean_abs_distance_residual(p_ml, d_single, p_bs)
    move0 = float(np.linalg.norm(p_ml - p_rf))

    ml_dist = np.linalg.norm(p_ml.reshape(2, 1) - p_bs, axis=0)
    residual_abs = np.abs(ml_dist - d_single)

    # 영향도 근사: 잔차가 크고 현재 weight도 높은 기지국일수록 위치해를 흔들 가능성이 크다고 본다.
    weight_norm = initial_weights / (np.nanmax(initial_weights) + eps)
    influence = np.zeros_like(initial_weights, dtype=float)
    influence[valid] = residual_abs[valid] * weight_norm[valid]

    active_influence = influence[valid]
    inf_scale = np.percentile(active_influence, 75) + eps

    influence_factor = 1.0 / (1.0 + INFLUENCE_STRENGTH * influence / inf_scale)
    influence_factor = np.clip(influence_factor, MIN_INFLUENCE_WEIGHT_FACTOR, 1.0)

    # ML이 이미 RF보다 충분히 좋고 이동량도 작으면, 영향도만 기록하고 재계산은 생략한다.
    if (ml_residual0 < JACKKNIFE_RISK_RATIO * rf_residual0) and (move0 < JACKKNIFE_MOVE_THRESHOLD):
        info_out = dict(ml_info)
        info_out["jackknife_mean_influence"] = float(np.mean(active_influence))
        info_out["jackknife_max_influence"] = float(np.max(active_influence))
        info_out["jackknife_accepted"] = False
        info_out["jackknife_skipped"] = True
        return p_ml, info_out, initial_weights, {
            "accepted": False,
            "mean_influence": info_out["jackknife_mean_influence"],
            "max_influence": info_out["jackknife_max_influence"]
        }

    refined_weights = initial_weights * influence_factor

    p_refined, info_refined = robust_multilateration(
        d_single=d_single,
        p_bs=p_bs,
        p_init=p_rf,
        initial_weights=refined_weights,
        max_iter=10
    )

    base_residual = simple_mean_abs_distance_residual(p_ml, d_single, p_bs)
    refined_residual = simple_mean_abs_distance_residual(p_refined, d_single, p_bs)
    base_move = float(np.linalg.norm(p_ml - p_rf))
    refined_move = float(np.linalg.norm(p_refined - p_rf))

    accepted = (refined_residual <= base_residual * 1.02) or (refined_move <= base_move * 0.95)

    info_out = dict(info_refined if accepted else ml_info)
    info_out["jackknife_mean_influence"] = float(np.mean(active_influence))
    info_out["jackknife_max_influence"] = float(np.max(active_influence))
    info_out["jackknife_accepted"] = bool(accepted)
    info_out["jackknife_base_residual"] = float(base_residual)
    info_out["jackknife_refined_residual"] = float(refined_residual)

    if accepted:
        return p_refined, info_out, refined_weights, {
            "accepted": True,
            "mean_influence": info_out["jackknife_mean_influence"],
            "max_influence": info_out["jackknife_max_influence"]
        }

    return p_ml, info_out, initial_weights, {
        "accepted": False,
        "mean_influence": info_out["jackknife_mean_influence"],
        "max_influence": info_out["jackknife_max_influence"]
    }

# ============================================================
# 10. 상태 지표 생성: Counterfactual Gamma Teacher 입력
# ============================================================

STATE_FEATURE_NAMES = [
    "rf_uncertainty",
    "rf_residual",
    "ml_residual",
    "residual_ratio",
    "residual_gain",
    "move_distance",
    "n_effective",
    "sigma",
    "n_valid",
    "weight_entropy",
    "max_weight",
    "weight_sum",
    "geometry_condition",
    "ml_improved_flag",
    "jackknife_mean_influence",
    "jackknife_max_influence",
    "jackknife_accepted_flag"
]


def calc_distance_residual(position, d_single, p_bs):
    valid = np.isfinite(d_single)

    if np.sum(valid) < 3:
        return np.nan

    dist = np.linalg.norm(
        position.reshape(2, 1) - p_bs,
        axis=0
    )

    return float(np.mean(np.abs(dist[valid] - d_single[valid])))


def weight_entropy(weights, eps=1e-12):
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w) & (w > 0)]

    if len(w) <= 1:
        return 0.0

    prob = w / (np.sum(w) + eps)
    ent = -np.sum(prob * np.log(prob + eps))
    ent_norm = ent / np.log(len(w))

    return float(ent_norm)


def geometry_condition(position, p_bs, weights, eps=1e-9):
    """
    기지국 방향 분포의 기하 안정성 지표.
    기존 condition number는 이상치가 매우 커질 수 있으므로 log1p(cond)로 반환한다.
    또한 G 행렬 계산은 np.einsum으로 벡터화한다.
    """
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(w) & (w > 0)

    if np.sum(valid) < 3:
        return float(np.log1p(1e6))

    anchors = p_bs[:, valid].T
    wv = w[valid]

    diff = position.reshape(1, 2) - anchors
    dist = np.linalg.norm(diff, axis=1)
    dist = np.maximum(dist, eps)
    unit = diff / dist.reshape(-1, 1)

    G = np.einsum("ni,nj,n->ij", unit, unit, wv)
    G = G / (np.sum(wv) + eps)

    eigvals = np.linalg.eigvalsh(G)
    min_eig = max(float(np.min(eigvals)), eps)
    max_eig = max(float(np.max(eigvals)), eps)

    cond = max_eig / min_eig
    return float(np.log1p(cond))


def make_state_features(
    p_rf,
    p_ml,
    d_single,
    p_bs,
    initial_weights,
    rf_uncertainty,
    ml_info,
    eps=1e-6
):
    rf_residual = calc_distance_residual(p_rf, d_single, p_bs)
    ml_residual = calc_distance_residual(p_ml, d_single, p_bs)

    if not np.isfinite(rf_residual):
        rf_residual = 1e6
    if not np.isfinite(ml_residual):
        ml_residual = 1e6

    residual_ratio = ml_residual / (rf_residual + eps)
    residual_gain = rf_residual - ml_residual
    move_distance = float(np.linalg.norm(p_ml - p_rf))

    n_effective = float(ml_info.get("n_effective", 0.0))
    sigma = float(ml_info.get("sigma", np.nan))
    if not np.isfinite(sigma):
        sigma = 1e6

    n_valid = float(np.sum(np.isfinite(d_single) & np.isfinite(initial_weights) & (initial_weights > 0)))
    ent = weight_entropy(initial_weights)
    max_w = float(np.nanmax(initial_weights)) if np.any(np.isfinite(initial_weights)) else 0.0
    weight_sum = float(np.nansum(initial_weights))
    geo_cond = geometry_condition(p_rf, p_bs, initial_weights)
    ml_improved_flag = 1.0 if ml_info.get("improved", False) else 0.0
    jackknife_mean_influence = float(ml_info.get("jackknife_mean_influence", 0.0))
    jackknife_max_influence = float(ml_info.get("jackknife_max_influence", 0.0))
    jackknife_accepted_flag = 1.0 if ml_info.get("jackknife_accepted", False) else 0.0

    z = np.array([
        rf_uncertainty,
        rf_residual,
        ml_residual,
        residual_ratio,
        residual_gain,
        move_distance,
        n_effective,
        sigma,
        n_valid,
        ent,
        max_w,
        weight_sum,
        geo_cond,
        ml_improved_flag,
        jackknife_mean_influence,
        jackknife_max_influence,
        jackknife_accepted_flag
    ], dtype=float)

    return z


# ============================================================
# 11. 기존 규칙 기반 gamma 결합
# ============================================================

def combine_rf_and_ml_rule(
    p_rf,
    p_ml,
    d_single,
    p_bs,
    rf_uncertainty,
    ml_info,
    eps=1e-6
):
    rf_error = calc_distance_residual(p_rf, d_single, p_bs)
    ml_error = calc_distance_residual(p_ml, d_single, p_bs)

    move_distance = np.linalg.norm(p_ml - p_rf)
    ml_improved = ml_info.get("improved", False)

    if ml_improved and ml_error < 0.85 * rf_error and move_distance < 15.0:
        gamma = 0.5
    elif ml_improved and ml_error < 0.95 * rf_error and move_distance < 25.0:
        gamma = 0.7
    else:
        gamma = 0.9

    final_position = gamma * p_rf + (1.0 - gamma) * p_ml

    return final_position, gamma


# ============================================================
# 12. Counterfactual Gamma Teacher
# ============================================================

def optimal_counterfactual_gamma(p_true, p_rf, p_ml, eps=1e-9):
    """
    p_final(gamma) = gamma*p_rf + (1-gamma)*p_ml 선분 위에서
    true 위치와 가장 가까운 gamma를 연속값으로 계산한다.
    gamma=1이면 RF만 사용, gamma=0이면 ML만 사용.
    """
    v = p_rf - p_ml
    denom = float(np.dot(v, v))

    if denom < eps:
        return 1.0

    gamma = float(np.dot(p_true - p_ml, v) / denom)
    gamma = np.clip(gamma, 0.0, 1.0)

    return gamma


def make_gamma_teacher_model(random_seed=RANDOM_SEED):
    gamma_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf_gamma",
                RandomForestRegressor(
                    n_estimators=GAMMA_MODEL_ESTIMATORS,
                    max_depth=5,
                    min_samples_split=6,
                    min_samples_leaf=8,
                    max_features=0.8,
                    bootstrap=True,
                    random_state=random_seed,
                    n_jobs=1
                )
            )
        ]
    )

    trust_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf_trust",
                RandomForestClassifier(
                    n_estimators=GAMMA_MODEL_ESTIMATORS,
                    max_depth=5,
                    min_samples_split=6,
                    min_samples_leaf=8,
                    max_features=0.8,
                    bootstrap=True,
                    class_weight="balanced",
                    random_state=random_seed,
                    n_jobs=1
                )
            )
        ]
    )

    return gamma_model, trust_model


def predict_base_sample(d_single, X_single, model, profile, p_bs, p_rf_override=None, rf_uncertainty_override=None):
    d_corrected = make_corrected_distance(d_single, profile)

    if p_rf_override is None:
        p_rf, rf_uncertainty = predict_rf_with_uncertainty(
            model,
            X_single
        )
    else:
        p_rf = np.asarray(p_rf_override, dtype=float)
        rf_uncertainty = 0.0 if rf_uncertainty_override is None else float(rf_uncertainty_override)

    initial_weights = compute_current_anchor_weights(
        d_single=d_corrected,
        p_rf=p_rf,
        p_bs=p_bs,
        profile=profile
    )

    p_ml, ml_info = robust_multilateration(
        d_single=d_corrected,
        p_bs=p_bs,
        p_init=p_rf,
        initial_weights=initial_weights
    )

    p_ml_raw = p_ml.copy()
    ml_info_raw = dict(ml_info)

    p_ml, ml_info, refined_weights, jackknife_diag = residual_based_anchor_reweighting(
        d_single=d_corrected,
        p_bs=p_bs,
        p_rf=p_rf,
        p_ml=p_ml,
        initial_weights=initial_weights,
        ml_info=ml_info
    )

    z = make_state_features(
        p_rf=p_rf,
        p_ml=p_ml,
        d_single=d_corrected,
        p_bs=p_bs,
        initial_weights=refined_weights,
        rf_uncertainty=rf_uncertainty,
        ml_info=ml_info
    )

    p_rule, gamma_rule = combine_rf_and_ml_rule(
        p_rf=p_rf,
        p_ml=p_ml,
        d_single=d_corrected,
        p_bs=p_bs,
        rf_uncertainty=rf_uncertainty,
        ml_info=ml_info
    )

    return {
        "p_rf": p_rf,
        "p_ml": p_ml,
        "p_ml_raw": p_ml_raw,
        "p_rule": p_rule,
        "gamma_rule": gamma_rule,
        "z": z,
        "d_corrected": d_corrected,
        "initial_weights": refined_weights,
        "initial_weights_raw": initial_weights,
        "jackknife_diag": jackknife_diag,
        "rf_uncertainty": rf_uncertainty,
        "ml_info": ml_info
    }


def build_oof_gamma_teacher_data(d_hat_train, p_train, p_bs):
    """
    train 데이터 내부에서 OOF 방식으로 gamma teacher 데이터를 만든다.
    각 샘플은 자기 자신을 학습하지 않은 RF/profile로부터 상태 z를 얻는다.

    중요 수정:
    1) teacher 학습 시에도 batch RF uncertainty를 계산해 최종 추론과 피처 분포를 맞춘다.
    2) inner_tune 고정 100개 대신 OOF 전체 train 데이터개를 사용해 leakage를 줄인다.
    """
    num_train = d_hat_train.shape[1]
    kf = KFold(n_splits=OOF_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    Z_list = []
    gamma_target_list = []
    use_ml_label_list = []
    rf_err_list = []
    ml_err_list = []
    rule_err_list = []
    best_err_list = []
    p_true_list = []
    p_rf_list = []
    p_ml_list = []
    p_rule_list = []
    gamma_rule_list = []

    print()
    print("Counterfactual Gamma Teacher 데이터 생성 - train 데이터 내부 OOF")
    print(f"OOF folds : {OOF_SPLITS}")
    print("teacher 학습/최종 추론 모두 RF uncertainty를 tree 분산으로 통일")

    fold_no = 1
    all_idx = np.arange(num_train)

    for fit_idx, hold_idx in kf.split(all_idx):
        d_fit = d_hat_train[:, fit_idx]
        p_fit = p_train[:, fit_idx]
        d_hold = d_hat_train[:, hold_idx]
        p_hold = p_train[:, hold_idx]

        profile_fold = build_anchor_profile(
            d_hat_train=d_fit,
            p_train=p_fit,
            p_bs=p_bs
        )

        X_fit = make_profile_features(d_fit, profile_fold)
        y_fit = p_fit.T
        X_hold = make_profile_features(d_hold, profile_fold)

        model_fold = make_rf_model(
            random_seed=RANDOM_SEED + fold_no,
            n_estimators=RF_N_ESTIMATORS_TEACHER
        )
        model_fold.fit(X_fit, y_fit)

        p_rf_hold_batch, rf_unc_hold_batch = predict_rf_batch_with_uncertainty(
            model_fold,
            X_hold
        )

        for local_idx in range(d_hold.shape[1]):
            sample = predict_base_sample(
                d_single=d_hold[:, local_idx],
                X_single=X_hold[local_idx:local_idx + 1],
                model=model_fold,
                profile=profile_fold,
                p_bs=p_bs,
                p_rf_override=p_rf_hold_batch[local_idx],
                rf_uncertainty_override=rf_unc_hold_batch[local_idx]
            )

            p_true = p_hold[:, local_idx]
            p_rf = sample["p_rf"]
            p_ml = sample["p_ml"]
            p_rule = sample["p_rule"]
            gamma_rule = sample["gamma_rule"]

            gamma_best = optimal_counterfactual_gamma(
                p_true=p_true,
                p_rf=p_rf,
                p_ml=p_ml
            )

            p_best = gamma_best * p_rf + (1.0 - gamma_best) * p_ml

            rf_err = float(np.linalg.norm(p_true - p_rf))
            ml_err = float(np.linalg.norm(p_true - p_ml))
            rule_err = float(np.linalg.norm(p_true - p_rule))
            best_err = float(np.linalg.norm(p_true - p_best))

            use_ml_label = 1 if gamma_best < 0.97 else 0

            Z_list.append(sample["z"])
            gamma_target_list.append(gamma_best)
            use_ml_label_list.append(use_ml_label)
            rf_err_list.append(rf_err)
            ml_err_list.append(ml_err)
            rule_err_list.append(rule_err)
            best_err_list.append(best_err)
            p_true_list.append(p_true)
            p_rf_list.append(p_rf)
            p_ml_list.append(p_ml)
            p_rule_list.append(p_rule)
            gamma_rule_list.append(gamma_rule)

        print(f"fold {fold_no}/{OOF_SPLITS} 완료 - holdout {len(hold_idx)} samples")
        fold_no += 1

    Z = np.vstack(Z_list)
    gamma_target = np.array(gamma_target_list, dtype=float)
    use_ml_label = np.array(use_ml_label_list, dtype=int)

    teacher_diag = {
        "rf_error": np.array(rf_err_list),
        "ml_error": np.array(ml_err_list),
        "rule_error": np.array(rule_err_list),
        "best_error": np.array(best_err_list),
        "use_ml_label": use_ml_label,
        "p_true": np.vstack(p_true_list),
        "p_rf": np.vstack(p_rf_list),
        "p_ml": np.vstack(p_ml_list),
        "p_rule": np.vstack(p_rule_list),
        "gamma_rule": np.array(gamma_rule_list, dtype=float)
    }

    return Z, gamma_target, teacher_diag


def build_inner_gamma_teacher_data(d_hat_train, p_train, p_bs, inner_train_size=400):
    """
    train 데이터를 inner_train 400 / inner_tune 100으로 나누어
    validation 200개를 보지 않고 gamma teacher 데이터를 만든다.
    """
    num_train = d_hat_train.shape[1]
    if num_train <= inner_train_size:
        inner_train_size = int(num_train * 0.8)

    d_inner_train = d_hat_train[:, :inner_train_size]
    p_inner_train = p_train[:, :inner_train_size]
    d_inner_tune = d_hat_train[:, inner_train_size:]
    p_inner_tune = p_train[:, inner_train_size:]

    print()
    print("Counterfactual Gamma Teacher 데이터 생성 - train 내부 분할")
    print(f"inner_train samples : {d_inner_train.shape[1]}")
    print(f"inner_tune samples  : {d_inner_tune.shape[1]}")

    profile_inner = build_anchor_profile(
        d_hat_train=d_inner_train,
        p_train=p_inner_train,
        p_bs=p_bs
    )

    X_inner_train = make_profile_features(d_inner_train, profile_inner)
    y_inner_train = p_inner_train.T
    X_inner_tune = make_profile_features(d_inner_tune, profile_inner)

    model_inner = make_rf_model(
        random_seed=RANDOM_SEED + 100,
        n_estimators=RF_N_ESTIMATORS_TEACHER
    )
    model_inner.fit(X_inner_train, y_inner_train)

    Z_list = []
    gamma_target_list = []
    use_ml_label_list = []
    rf_err_list = []
    ml_err_list = []
    best_err_list = []
    rule_err_list = []
    p_true_list = []
    p_rf_list = []
    p_ml_list = []
    p_rule_list = []
    gamma_rule_list = []

    for local_idx in range(d_inner_tune.shape[1]):
        sample = predict_base_sample(
            d_single=d_inner_tune[:, local_idx],
            X_single=X_inner_tune[local_idx:local_idx + 1],
            model=model_inner,
            profile=profile_inner,
            p_bs=p_bs
        )

        p_true = p_inner_tune[:, local_idx]
        p_rf = sample["p_rf"]
        p_ml = sample["p_ml"]

        gamma_best = optimal_counterfactual_gamma(
            p_true=p_true,
            p_rf=p_rf,
            p_ml=p_ml
        )

        p_best = gamma_best * p_rf + (1.0 - gamma_best) * p_ml

        rf_err = float(np.linalg.norm(p_true - p_rf))
        ml_err = float(np.linalg.norm(p_true - p_ml))
        best_err = float(np.linalg.norm(p_true - p_best))
        rule_err = float(np.linalg.norm(p_true - sample["p_rule"]))

        use_ml_label = 1 if gamma_best < 0.97 else 0

        p_true_list.append(p_true)
        p_rf_list.append(p_rf)
        p_ml_list.append(p_ml)
        p_rule_list.append(sample["p_rule"])
        gamma_rule_list.append(sample["gamma_rule"])

        Z_list.append(sample["z"])
        gamma_target_list.append(gamma_best)
        use_ml_label_list.append(use_ml_label)
        rf_err_list.append(rf_err)
        ml_err_list.append(ml_err)
        best_err_list.append(best_err)
        rule_err_list.append(rule_err)

    Z = np.vstack(Z_list)
    gamma_target = np.array(gamma_target_list, dtype=float)
    use_ml_label = np.array(use_ml_label_list, dtype=int)

    teacher_diag = {
        "rf_error": np.array(rf_err_list),
        "ml_error": np.array(ml_err_list),
        "rule_error": np.array(rule_err_list),
        "best_error": np.array(best_err_list),
        "use_ml_label": use_ml_label,
        "p_true": np.vstack(p_true_list),
        "p_rf": np.vstack(p_rf_list),
        "p_ml": np.vstack(p_ml_list),
        "p_rule": np.vstack(p_rule_list),
        "gamma_rule": np.array(gamma_rule_list, dtype=float)
    }

    return Z, gamma_target, teacher_diag


def make_selector_features(z, gamma_teacher, gamma_raw, trust_prob):
    if not np.isfinite(trust_prob):
        trust_prob = 0.0

    extra = np.array([gamma_teacher, gamma_raw, trust_prob], dtype=float)
    return np.concatenate([np.asarray(z, dtype=float), extra])


def make_selector_model(random_seed=RANDOM_SEED):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf_selector",
                RandomForestClassifier(
                    n_estimators=SELECTOR_MODEL_ESTIMATORS,
                    max_depth=5,
                    min_samples_split=6,
                    min_samples_leaf=8,
                    max_features=0.8,
                    bootstrap=True,
                    class_weight="balanced",
                    random_state=random_seed,
                    n_jobs=1
                )
            )
        ]
    )



def fit_gamma_trust_models(Z_fit, gamma_target_fit, use_ml_label_fit, random_seed):
    gamma_model, trust_model = make_gamma_teacher_model(random_seed=random_seed)
    gamma_model.fit(Z_fit, gamma_target_fit)

    if len(np.unique(use_ml_label_fit)) >= 2:
        trust_model.fit(Z_fit, use_ml_label_fit)
        has_trust_model = True
    else:
        trust_model = None
        has_trust_model = False

    return gamma_model, trust_model, has_trust_model


def apply_gamma_teacher_core(z, gamma_model, trust_model, gate):
    z2 = z.reshape(1, -1)

    gamma_raw = float(gamma_model.predict(z2)[0])
    gamma_raw = float(np.clip(gamma_raw, gate["gamma_min"], gate["gamma_max"]))

    if trust_model is not None:
        proba = trust_model.predict_proba(z2)[0]
        classes = trust_model.named_steps["rf_trust"].classes_
        if 1 in classes:
            idx_cls = int(np.where(classes == 1)[0][0])
            trust_prob = float(proba[idx_cls])
        else:
            trust_prob = 0.0
    else:
        trust_prob = float(1.0 - gamma_raw)

    gamma = 1.0 - trust_prob * (1.0 - gamma_raw)

    idx = {name: i for i, name in enumerate(STATE_FEATURE_NAMES)}

    residual_ratio = z[idx["residual_ratio"]]
    move_distance = z[idx["move_distance"]]
    n_effective = z[idx["n_effective"]]
    geo_cond = z[idx["geometry_condition"]]
    rf_uncertainty = z[idx["rf_uncertainty"]]
    ml_residual = z[idx["ml_residual"]]
    rf_residual = z[idx["rf_residual"]]

    if residual_ratio > gate["residual_ratio_limit"]:
        gamma = max(gamma, 0.90)

    if move_distance > gate["move_limit"]:
        gamma = max(gamma, 0.90)

    if n_effective < gate["n_effective_min"]:
        gamma = max(gamma, 0.95)

    if geo_cond > gate["geometry_condition_limit"]:
        gamma = max(gamma, 0.90)

    if (rf_uncertainty < gate["rf_uncertainty_low"]) and (ml_residual >= rf_residual) and (move_distance > 5.0):
        gamma = max(gamma, 0.95)

    gamma = float(np.clip(gamma, gate["gamma_min"], gate["gamma_max"]))

    return gamma, gamma_raw, trust_prob



def train_gamma_teacher(Z, gamma_target, teacher_diag):
    """
    Gamma model/trust model은 최종적으로 train OOF 데이터 전체로 학습한다.
    Selector는 같은 데이터에 즉시 재적합하지 않고, train 내부 OOF 예측으로 selector 학습 데이터를 만든다.
    """
    use_ml_label = teacher_diag["use_ml_label"]

    # train 내부에서 ML을 반영해도 좋았던 샘플들의 안전 범위를 학습한다.
    useful = gamma_target < 0.97
    if np.sum(useful) < 20:
        useful = np.ones_like(gamma_target, dtype=bool)

    idx_residual_ratio = STATE_FEATURE_NAMES.index("residual_ratio")
    idx_move = STATE_FEATURE_NAMES.index("move_distance")
    idx_neff = STATE_FEATURE_NAMES.index("n_effective")
    idx_geo = STATE_FEATURE_NAMES.index("geometry_condition")
    idx_rf_unc = STATE_FEATURE_NAMES.index("rf_uncertainty")

    gate = {
        "residual_ratio_limit": float(np.percentile(Z[useful, idx_residual_ratio], 90)),
        "move_limit": float(np.percentile(Z[useful, idx_move], 90)),
        "n_effective_min": float(np.percentile(Z[useful, idx_neff], 10)),
        "geometry_condition_limit": float(np.percentile(Z[useful, idx_geo], 90)),
        "rf_uncertainty_low": float(np.percentile(Z[:, idx_rf_unc], 25)),
        "gamma_min": float(max(0.35, np.percentile(gamma_target, 5))),
        "gamma_ref": float(np.median(gamma_target)),
        "gamma_max": 1.0
    }

    # 1) 최종 gamma/trust 모델은 전체 OOF teacher 데이터로 학습
    gamma_model, trust_model, has_trust_model = fit_gamma_trust_models(
        Z,
        gamma_target,
        use_ml_label,
        random_seed=RANDOM_SEED
    )

    gate["has_trust_model"] = has_trust_model

    teacher = {
        "gamma_model": gamma_model,
        "trust_model": trust_model,
        "gate": gate,
        "selector_model": None,
        "has_selector_model": False,
        "selector_teacher_rate": 0.0,
        "selector_train_rule_mean_error": np.nan,
        "selector_train_teacher_mean_error": np.nan
    }

    # 2) Selector는 train 내부 OOF로 만든 teacher 예측에 대해 학습한다.
    required_keys = {"p_true", "p_rf", "p_ml", "p_rule", "rule_error"}

    if required_keys.issubset(set(teacher_diag.keys())):
        p_true_all = teacher_diag["p_true"]
        p_rf_all = teacher_diag["p_rf"]
        p_ml_all = teacher_diag["p_ml"]
        rule_err_all = teacher_diag["rule_error"]

        selector_X = []
        selector_y = []
        teacher_err_list = []

        kf_sel = KFold(n_splits=OOF_SPLITS, shuffle=True, random_state=RANDOM_SEED + 777)
        all_idx = np.arange(Z.shape[0])

        for fold_no, (fit_idx, hold_idx) in enumerate(kf_sel.split(all_idx), start=1):
            gamma_fold, trust_fold, _ = fit_gamma_trust_models(
                Z[fit_idx],
                gamma_target[fit_idx],
                use_ml_label[fit_idx],
                random_seed=RANDOM_SEED + 500 + fold_no
            )

            for i in hold_idx:
                gamma_t, gamma_raw, trust_prob = apply_gamma_teacher_core(
                    Z[i],
                    gamma_fold,
                    trust_fold,
                    gate
                )

                p_teacher = gamma_t * p_rf_all[i] + (1.0 - gamma_t) * p_ml_all[i]
                teacher_err = float(np.linalg.norm(p_true_all[i] - p_teacher))
                teacher_err_list.append(teacher_err)

                selector_X.append(make_selector_features(Z[i], gamma_t, gamma_raw, trust_prob))
                selector_y.append(1 if teacher_err < rule_err_all[i] - SELECTOR_MARGIN else 0)

        selector_X = np.vstack(selector_X)
        selector_y = np.array(selector_y, dtype=int)
        teacher_err_arr = np.array(teacher_err_list, dtype=float)

        if len(np.unique(selector_y)) >= 2:
            selector_model = make_selector_model(random_seed=RANDOM_SEED)
            selector_model.fit(selector_X, selector_y)
            teacher["selector_model"] = selector_model
            teacher["has_selector_model"] = True

        teacher["selector_teacher_rate"] = float(np.mean(selector_y))
        teacher["selector_train_rule_mean_error"] = float(np.mean(rule_err_all))
        teacher["selector_train_teacher_mean_error"] = float(np.mean(teacher_err_arr))

    return teacher


def apply_gamma_teacher(z, teacher):
    return apply_gamma_teacher_core(
        z,
        teacher["gamma_model"],
        teacher.get("trust_model", None),
        teacher["gate"]
    )


def select_rule_or_teacher(z, gamma_t, gamma_raw, trust_prob, teacher):
    selector_model = teacher.get("selector_model", None)

    if selector_model is None:
        return False, 0.0

    x = make_selector_features(z, gamma_t, gamma_raw, trust_prob).reshape(1, -1)

    try:
        proba = selector_model.predict_proba(x)[0]
        classes = selector_model.named_steps["rf_selector"].classes_
        if 1 in classes:
            idx = int(np.where(classes == 1)[0][0])
            p_use_teacher = float(proba[idx])
        else:
            p_use_teacher = 0.0
    except Exception:
        p_use_teacher = float(selector_model.predict(x)[0])

    return p_use_teacher >= SELECTOR_PROBA_THRESHOLD, p_use_teacher


# ============================================================
# 13. OOF Counterfactual 4-Way Expert Router
# ============================================================

EXPERT_NAMES = np.array(["RF", "ML", "Rule", "Teacher"], dtype=object)


def _safe_float(x, default=0.0):
    try:
        x = float(x)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return float(default)


def make_expert_router_features(
    z,
    p_rf,
    p_ml,
    p_rule,
    p_teacher,
    gamma_rule,
    gamma_teacher,
    gamma_raw,
    trust_prob
):
    """
    4개 후보(RF/ML/Rule/Teacher) 중 어떤 전문가를 선택할지 판단하는 feature.
    validation 정답은 절대 사용하지 않고, 현재 샘플에서 계산 가능한 상태량만 사용한다.
    """
    z = np.asarray(z, dtype=float)
    p_rf = np.asarray(p_rf, dtype=float)
    p_ml = np.asarray(p_ml, dtype=float)
    p_rule = np.asarray(p_rule, dtype=float)
    p_teacher = np.asarray(p_teacher, dtype=float)

    candidates = [p_rf, p_ml, p_rule, p_teacher]
    pairwise = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            pairwise.append(float(np.linalg.norm(candidates[i] - candidates[j])))
    pairwise = np.array(pairwise, dtype=float)

    # 후보들이 서로 많이 흩어져 있으면 선택 위험이 크다는 신호로 사용한다.
    spread_mean = _safe_float(np.mean(pairwise))
    spread_max = _safe_float(np.max(pairwise))
    spread_std = _safe_float(np.std(pairwise))

    extra = np.array([
        _safe_float(gamma_rule),
        _safe_float(gamma_teacher),
        _safe_float(gamma_raw),
        _safe_float(trust_prob),
        float(np.linalg.norm(p_rf - p_ml)),
        float(np.linalg.norm(p_rf - p_rule)),
        float(np.linalg.norm(p_rf - p_teacher)),
        float(np.linalg.norm(p_ml - p_rule)),
        float(np.linalg.norm(p_ml - p_teacher)),
        float(np.linalg.norm(p_rule - p_teacher)),
        spread_mean,
        spread_max,
        spread_std
    ], dtype=float)

    feat = np.concatenate([z, extra])
    feat[~np.isfinite(feat)] = 0.0
    return feat


def make_expert_router_model(random_seed=RANDOM_SEED):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf_router",
                RandomForestClassifier(
                    n_estimators=EXPERT_ROUTER_MODEL_ESTIMATORS,
                    max_depth=7,
                    min_samples_split=8,
                    min_samples_leaf=6,
                    max_features=0.8,
                    bootstrap=True,
                    class_weight="balanced",
                    random_state=random_seed,
                    n_jobs=1
                )
            )
        ]
    )


def train_four_way_expert_router(Z, gamma_target, teacher_diag, gate):
    """
    train 데이터 내부 OOF 결과로 RF/ML/Rule/Teacher 중 최적 후보 라벨을 만든다.
    라벨은 정답 좌표를 사용하는 반사실적 진단 라벨이지만, train 내부에서만 생성된다.
    validation 200개는 라벨 생성/튜닝에 사용하지 않는다.
    """
    required_keys = {"p_true", "p_rf", "p_ml", "p_rule", "gamma_rule", "rule_error"}
    if not required_keys.issubset(set(teacher_diag.keys())):
        return None, {
            "has_router": False,
            "reason": "teacher_diag required keys missing"
        }

    p_true_all = teacher_diag["p_true"]
    p_rf_all = teacher_diag["p_rf"]
    p_ml_all = teacher_diag["p_ml"]
    p_rule_all = teacher_diag["p_rule"]
    gamma_rule_all = teacher_diag["gamma_rule"]
    use_ml_label = teacher_diag["use_ml_label"]

    router_X = []
    router_y = []
    oracle_err = []
    candidate_err_accum = []

    kf_router = KFold(n_splits=OOF_SPLITS, shuffle=True, random_state=RANDOM_SEED + 999)
    all_idx = np.arange(Z.shape[0])

    for fold_no, (fit_idx, hold_idx) in enumerate(kf_router.split(all_idx), start=1):
        gamma_fold, trust_fold, _ = fit_gamma_trust_models(
            Z[fit_idx],
            gamma_target[fit_idx],
            use_ml_label[fit_idx],
            random_seed=RANDOM_SEED + 900 + fold_no
        )

        for i in hold_idx:
            gamma_t, gamma_raw, trust_prob = apply_gamma_teacher_core(
                Z[i],
                gamma_fold,
                trust_fold,
                gate
            )

            p_teacher = gamma_t * p_rf_all[i] + (1.0 - gamma_t) * p_ml_all[i]

            candidate_points = [
                p_rf_all[i],
                p_ml_all[i],
                p_rule_all[i],
                p_teacher
            ]
            candidate_err = np.array([
                np.linalg.norm(p_true_all[i] - cand)
                for cand in candidate_points
            ], dtype=float)

            best_label = int(np.argmin(candidate_err))

            router_X.append(
                make_expert_router_features(
                    z=Z[i],
                    p_rf=p_rf_all[i],
                    p_ml=p_ml_all[i],
                    p_rule=p_rule_all[i],
                    p_teacher=p_teacher,
                    gamma_rule=gamma_rule_all[i],
                    gamma_teacher=gamma_t,
                    gamma_raw=gamma_raw,
                    trust_prob=trust_prob
                )
            )
            router_y.append(best_label)
            oracle_err.append(float(candidate_err[best_label]))
            candidate_err_accum.append(candidate_err)

    router_X = np.vstack(router_X)
    router_y = np.array(router_y, dtype=int)
    oracle_err = np.array(oracle_err, dtype=float)
    candidate_err_accum = np.vstack(candidate_err_accum)

    if len(np.unique(router_y)) < 2:
        return None, {
            "has_router": False,
            "reason": "only one expert label generated",
            "label_distribution": np.bincount(router_y, minlength=4)
        }

    router_model = make_expert_router_model(random_seed=RANDOM_SEED)
    router_model.fit(router_X, router_y)

    label_counts = np.bincount(router_y, minlength=4)
    label_rates = label_counts / max(1, np.sum(label_counts))

    diag = {
        "has_router": True,
        "label_distribution": label_counts,
        "label_rate": label_rates,
        "oracle_mean_error": float(np.mean(oracle_err)),
        "oracle_median_error": float(np.median(oracle_err)),
        "oracle_90_error": float(np.percentile(oracle_err, 90)),
        "candidate_mean_errors": np.mean(candidate_err_accum, axis=0)
    }

    return router_model, diag


def apply_four_way_expert_router(
    z,
    p_rf,
    p_ml,
    p_rule,
    p_teacher,
    gamma_rule,
    gamma_teacher,
    gamma_raw,
    trust_prob,
    router_model
):
    if router_model is None:
        return 3, 0.0, p_teacher

    x = make_expert_router_features(
        z=z,
        p_rf=p_rf,
        p_ml=p_ml,
        p_rule=p_rule,
        p_teacher=p_teacher,
        gamma_rule=gamma_rule,
        gamma_teacher=gamma_teacher,
        gamma_raw=gamma_raw,
        trust_prob=trust_prob
    ).reshape(1, -1)

    try:
        pred_label = int(router_model.predict(x)[0])
        proba = router_model.predict_proba(x)[0]
        classes = router_model.named_steps["rf_router"].classes_
        if pred_label in classes:
            p_conf = float(proba[int(np.where(classes == pred_label)[0][0])])
        else:
            p_conf = 0.0
    except Exception:
        pred_label = 3
        p_conf = 0.0

    candidates = [p_rf, p_ml, p_rule, p_teacher]
    pred_label = int(np.clip(pred_label, 0, 3))
    return pred_label, p_conf, candidates[pred_label]


# ============================================================
# 14. 전체 추론 함수
# ============================================================

def predict_with_algorithm(d_hat_eval, X_eval, model, profile, p_bs, gamma_teacher=None, expert_router=None):
    num_eval = d_hat_eval.shape[1]

    rf_pred = np.zeros((num_eval, 2))
    ml_pred = np.zeros((num_eval, 2))
    rule_pred = np.zeros((num_eval, 2))
    teacher_pred = np.zeros((num_eval, 2))
    selective_pred = np.zeros((num_eval, 2))
    expert_pred = np.zeros((num_eval, 2))

    gamma_rule_list = []
    gamma_teacher_list = []
    gamma_raw_list = []
    trust_prob_list = []
    selector_used_list = []
    selector_prob_list = []
    expert_label_list = []
    expert_confidence_list = []
    rf_uncertainty_list = []
    ml_residual_list = []
    ml_improved_list = []
    n_effective_list = []
    jackknife_accepted_list = []
    jackknife_max_influence_list = []
    state_list = []

    # RF 예측은 batch로 한 번에 계산하되,
    # tree별 예측 분산을 같이 계산해서 RF 불확실도도 유지한다.
    p_rf_batch, rf_uncertainty_batch = predict_rf_batch_with_uncertainty(
        model,
        X_eval
    )
    print(f"RF batch prediction + uncertainty 계산 완료: {num_eval} samples")

    for local_idx in range(num_eval):
        if (local_idx + 1) % 50 == 0:
            print(f"validation 진행: {local_idx + 1}/{num_eval}")
        sample = predict_base_sample(
            d_single=d_hat_eval[:, local_idx],
            X_single=X_eval[local_idx:local_idx + 1],
            model=model,
            profile=profile,
            p_bs=p_bs,
            p_rf_override=p_rf_batch[local_idx],
            rf_uncertainty_override=rf_uncertainty_batch[local_idx]
        )

        p_rf = sample["p_rf"]
        p_ml = sample["p_ml"]
        p_rule = sample["p_rule"]
        gamma_rule = sample["gamma_rule"]
        z = sample["z"]

        if gamma_teacher is not None:
            gamma_t, gamma_raw, trust_prob = apply_gamma_teacher(z, gamma_teacher)
            use_teacher, selector_prob = select_rule_or_teacher(
                z=z,
                gamma_t=gamma_t,
                gamma_raw=gamma_raw,
                trust_prob=trust_prob,
                teacher=gamma_teacher
            )
        else:
            gamma_t = gamma_rule
            gamma_raw = gamma_rule
            trust_prob = np.nan
            use_teacher = False
            selector_prob = 0.0

        p_teacher = gamma_t * p_rf + (1.0 - gamma_t) * p_ml
        p_selective = p_teacher if use_teacher else p_rule

        if expert_router is not None:
            expert_label, expert_confidence, p_expert = apply_four_way_expert_router(
                z=z,
                p_rf=p_rf,
                p_ml=p_ml,
                p_rule=p_rule,
                p_teacher=p_teacher,
                gamma_rule=gamma_rule,
                gamma_teacher=gamma_t,
                gamma_raw=gamma_raw,
                trust_prob=trust_prob,
                router_model=expert_router
            )
        else:
            expert_label = 3 if use_teacher else 2
            expert_confidence = selector_prob
            p_expert = p_selective

        rf_pred[local_idx, :] = p_rf
        ml_pred[local_idx, :] = p_ml
        rule_pred[local_idx, :] = p_rule
        teacher_pred[local_idx, :] = p_teacher
        selective_pred[local_idx, :] = p_selective
        expert_pred[local_idx, :] = p_expert

        gamma_rule_list.append(gamma_rule)
        gamma_teacher_list.append(gamma_t)
        gamma_raw_list.append(gamma_raw)
        trust_prob_list.append(trust_prob)
        selector_used_list.append(use_teacher)
        selector_prob_list.append(selector_prob)
        expert_label_list.append(expert_label)
        expert_confidence_list.append(expert_confidence)
        rf_uncertainty_list.append(sample["rf_uncertainty"])
        ml_residual_list.append(sample["ml_info"]["mean_abs_residual"])
        ml_improved_list.append(sample["ml_info"]["improved"])
        n_effective_list.append(sample["ml_info"]["n_effective"])
        jackknife_accepted_list.append(sample["ml_info"].get("jackknife_accepted", False))
        jackknife_max_influence_list.append(sample["ml_info"].get("jackknife_max_influence", 0.0))
        state_list.append(z)

    diagnostic = {
        "gamma_rule": np.array(gamma_rule_list),
        "gamma_teacher": np.array(gamma_teacher_list),
        "gamma_raw": np.array(gamma_raw_list),
        "trust_prob": np.array(trust_prob_list),
        "selector_used": np.array(selector_used_list, dtype=bool),
        "selector_prob": np.array(selector_prob_list, dtype=float),
        "expert_label": np.array(expert_label_list, dtype=int),
        "expert_confidence": np.array(expert_confidence_list, dtype=float),
        "rf_uncertainty": np.array(rf_uncertainty_list),
        "ml_residual": np.array(ml_residual_list),
        "ml_improved": np.array(ml_improved_list),
        "n_effective": np.array(n_effective_list),
        "jackknife_accepted": np.array(jackknife_accepted_list, dtype=bool),
        "jackknife_max_influence": np.array(jackknife_max_influence_list, dtype=float),
        "state": np.vstack(state_list)
    }

    return rf_pred, ml_pred, rule_pred, teacher_pred, selective_pred, expert_pred, diagnostic


# ============================================================
# 14. 위치 오차 평가
# ============================================================

def get_position_error(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred, axis=1)


def evaluate_position_error(y_true, y_pred, title):
    coord_mae = mean_absolute_error(y_true, y_pred)
    coord_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    position_error = get_position_error(y_true, y_pred)

    print()
    print("============================================================")
    print(title)
    print("============================================================")
    print(f"Coordinate MAE        : {coord_mae:.6f}")
    print(f"Coordinate RMSE       : {coord_rmse:.6f}")
    print(f"Mean Position Error   : {np.mean(position_error):.6f}")
    print(f"Median Position Error : {np.median(position_error):.6f}")
    print(f"90% Position Error    : {np.percentile(position_error, 90):.6f}")
    print(f"Max Position Error    : {np.max(position_error):.6f}")

    return position_error


# ============================================================
# 15. 제출용 학습 실행
# ============================================================

MODEL_PATH = "model.pkl"


def resolve_train_mat_path():
    """
    제출 환경에서는 DH_FR1.mat를 사용한다.
    로컬 실험 파일명이 InF_DH_FR1.mat인 경우만 fallback으로 허용한다.
    """
    candidates = [MAT_PATH, "InF_DH_FR1.mat"]
    for path in candidates:
        if Path(path).exists():
            return path
    raise FileNotFoundError("DH_FR1.mat 파일을 찾을 수 없습니다. train.py와 같은 폴더에 배치하세요.")


def build_model_package(d_hat, p, p_bs):
    """
    제공된 학습 데이터 전체를 사용해 최종 제출용 모델 패키지를 만든다.
    hidden test의 정답 p는 main.py에서 절대 사용하지 않고, 여기 train.py에서만 p를 사용한다.
    """
    num_user = d_hat.shape[1]

    print()
    print("제출용 최종 모델 학습 시작")
    print(f"학습 샘플 수 : {num_user}")
    print("사용 알고리즘: v4 OOF Counterfactual 4-Way Expert Router")

    # 1. 전체 train 데이터 내부 OOF로 Gamma Teacher / Router 학습 데이터 생성
    Z_teacher, gamma_target, teacher_diag = build_oof_gamma_teacher_data(
        d_hat_train=d_hat,
        p_train=p,
        p_bs=p_bs
    )

    print()
    print("Gamma Teacher 내부 진단")
    print(f"teacher samples           : {len(gamma_target)}")
    print(f"평균 gamma_best           : {np.mean(gamma_target):.6f}")
    print(f"중앙 gamma_best           : {np.median(gamma_target):.6f}")
    print(f"ML 반영 필요 샘플 비율    : {np.mean(teacher_diag['use_ml_label']) * 100:.2f}%")
    print(f"OOF RF 평균 위치 오차      : {np.mean(teacher_diag['rf_error']):.6f}")
    print(f"OOF ML 평균 위치 오차      : {np.mean(teacher_diag['ml_error']):.6f}")
    print(f"OOF 최적 gamma 평균 오차   : {np.mean(teacher_diag['best_error']):.6f}")

    gamma_teacher = train_gamma_teacher(
        Z=Z_teacher,
        gamma_target=gamma_target,
        teacher_diag=teacher_diag
    )

    print()
    print("Gamma Teacher 학습 완료")
    gate = gamma_teacher["gate"]
    print(f"gamma_min                 : {gate['gamma_min']:.6f}")
    print(f"gamma_ref                 : {gate['gamma_ref']:.6f}")
    print(f"residual_ratio_limit      : {gate['residual_ratio_limit']:.6f}")
    print(f"move_limit                : {gate['move_limit']:.6f}")
    print(f"n_effective_min           : {gate['n_effective_min']:.6f}")
    print(f"geometry_condition_limit  : {gate['geometry_condition_limit']:.6f}")
    print(f"selector teacher 사용 학습비율 : {gamma_teacher.get('selector_teacher_rate', 0.0) * 100:.2f}%")
    print(f"selector train rule 평균오차   : {gamma_teacher.get('selector_train_rule_mean_error', np.nan):.6f}")
    print(f"selector train teacher 평균오차: {gamma_teacher.get('selector_train_teacher_mean_error', np.nan):.6f}")

    # 2. 전체 train 데이터 내부 OOF 반사실적 라벨로 4-Way Expert Router 학습
    expert_router, expert_router_diag = train_four_way_expert_router(
        Z=Z_teacher,
        gamma_target=gamma_target,
        teacher_diag=teacher_diag,
        gate=gamma_teacher["gate"]
    )

    print()
    print("4-Way Expert Router 학습 완료")
    if expert_router_diag.get("has_router", False):
        print(f"router oracle 평균오차       : {expert_router_diag['oracle_mean_error']:.6f}")
        print(f"router oracle 중앙오차       : {expert_router_diag['oracle_median_error']:.6f}")
        print(f"router oracle 90%오차        : {expert_router_diag['oracle_90_error']:.6f}")
        print("router 학습 라벨 비율 RF/ML/Rule/Teacher")
        for name, rate in zip(EXPERT_NAMES, expert_router_diag["label_rate"]):
            print(f"{str(name):>8s}: {rate * 100:.2f}%")
    else:
        print(f"router 비활성화 사유: {expert_router_diag.get('reason', 'unknown')}")

    # 3. 전체 train 데이터로 최종 profile + RF 모델 학습
    profile = build_anchor_profile(
        d_hat_train=d_hat,
        p_train=p,
        p_bs=p_bs
    )

    X_train = make_profile_features(d_hat, profile)
    y_train = p.T

    print()
    print("최종 RF 입력 확인")
    print(f"X_train shape : {X_train.shape}")
    print(f"y_train shape : {y_train.shape}")

    rf_model = make_rf_model(
        random_seed=RANDOM_SEED,
        n_estimators=RF_N_ESTIMATORS_FINAL
    )

    print()
    print("최종 RF 학습 시작")
    rf_model.fit(X_train, y_train)
    print("최종 RF 학습 완료")

    model_package = {
        "algorithm_name": "v4_oof_counterfactual_4way_expert_router",
        "rf_model": rf_model,
        "profile": profile,
        "gamma_teacher": gamma_teacher,
        "expert_router": expert_router,
        "expert_router_diag": expert_router_diag,
        "p_bs_train": np.asarray(p_bs, dtype=float),
        "state_feature_names": list(STATE_FEATURE_NAMES),
        "expert_names": [str(x) for x in EXPERT_NAMES],
        "settings": {
            "random_seed": RANDOM_SEED,
            "rf_n_estimators_final": RF_N_ESTIMATORS_FINAL,
            "rf_n_estimators_teacher": RF_N_ESTIMATORS_TEACHER,
            "gamma_model_estimators": GAMMA_MODEL_ESTIMATORS,
            "selector_model_estimators": SELECTOR_MODEL_ESTIMATORS,
            "expert_router_model_estimators": EXPERT_ROUTER_MODEL_ESTIMATORS,
            "oof_splits": OOF_SPLITS,
            "use_bin_bias_correction": USE_BIN_BIAS_CORRECTION,
            "bias_bin_count": BIAS_BIN_COUNT,
            "min_bin_samples": MIN_BIN_SAMPLES,
            "use_jackknife_reweighting": USE_JACKKNIFE_REWEIGHTING,
            "jackknife_max_iter": JACKKNIFE_MAX_ITER,
            "jackknife_candidate_count": JACKKNIFE_CANDIDATE_COUNT,
            "jackknife_risk_ratio": JACKKNIFE_RISK_RATIO,
            "jackknife_move_threshold": JACKKNIFE_MOVE_THRESHOLD,
            "influence_strength": INFLUENCE_STRENGTH,
            "min_influence_weight_factor": MIN_INFLUENCE_WEIGHT_FACTOR,
            "selector_margin": SELECTOR_MARGIN,
            "selector_proba_threshold": SELECTOR_PROBA_THRESHOLD
        }
    }

    return model_package


def main():
    mat_path = resolve_train_mat_path()

    print("제출용 train.py 실행")
    print(f"데이터 파일 : {mat_path}")

    d_hat, p, p_bs, indices = load_mat_data(mat_path)

    print()
    print("데이터 확인")
    print(f"d_hat shape   : {d_hat.shape}")
    print(f"p shape       : {p.shape}")
    print(f"p_bs shape    : {p_bs.shape}")
    print(f"indices shape : {indices.shape}")

    model_package = build_model_package(
        d_hat=d_hat,
        p=p,
        p_bs=p_bs
    )

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_package, f)

    print()
    print("model.pkl 저장 완료")
    print(f"저장 파일 : {MODEL_PATH}")
    print("1단계 완료: train.py는 public 데이터로 모델을 학습하고 model.pkl을 생성한다.")


if __name__ == "__main__":
    main()

