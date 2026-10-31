import numpy as np

def linear_spline_basis(x, knots):
    """C = [1, x, (x-κ1)+, ..., (x-κK)+]"""
    x = np.asarray(x, dtype=float)
    cols = [np.ones_like(x), x]
    for k in knots:
        cols.append(np.maximum(x - k, 0.0))
    return np.column_stack(cols)

def penalty_matrix(p, K):
    """D = diag(0,...,0, 1,...,1)，前 p+1 个不惩罚"""
    d = np.zeros(p + 1 + K)
    d[p + 1:] = 1.0
    return np.diag(d)

def fit_pspline(C, y, D, lam, p=1):
    """返回 beta_hat, y_hat, S_lambda；lam=0 用 lstsq 防秩亏"""
    if lam == 0:
        beta, *_ = np.linalg.lstsq(C, y, rcond=None)
        S = C @ np.linalg.pinv(C.T @ C) @ C.T
    else:
        A = C.T @ C + (lam ** (2 * p)) * D
        beta = np.linalg.solve(A, C.T @ y)
        S = C @ np.linalg.solve(A, C.T)
    return beta, C @ beta, S

def df_fit(S):
    return float(np.trace(S))

def df_res(S):
    n = S.shape[0]
    return float(n - 2 * np.trace(S) + np.trace(S @ S.T))

def knots_quantile(x, K):
    """按书中 (5.8) 生成样本分位结点"""
    u = np.unique(x)
    qs = np.linspace(0, 1, K + 2)[1:-1]
    return np.quantile(u, qs)

# ---- 二次样条基 ----
def quadratic_spline_basis(x, knots):
    x = np.asarray(x, dtype=float)
    cols = [np.ones_like(x), x, x ** 2]
    for k in knots:
        cols.append(np.maximum(x - k, 0.0) ** 2)
    return np.column_stack(cols)

# ---- CV / GCV / RSS ----
def cv_score(S, y):
    e = y - S @ y
    d = 1.0 - np.diag(S)
    return float(np.sum((e / d) ** 2))

def gcv_score(S, y):
    n = len(y)
    rss = float(np.sum((y - S @ y) ** 2))
    return rss / (1.0 - np.trace(S) / n) ** 2

def rss_score(S, y):
    return float(np.sum((y - S @ y) ** 2))

# ---- REML / ML 一维优化求 lambda ----
def reml_ml_lambda(X_poly, Z, y, method="REML"):
    """
    X_poly: n×p 多项式部分；Z: n×K 结点部分
    返回最优 lambda = sqrt(sigma_eps^2 / sigma_u^2)
    用 Woodbury 避免 n×n 求逆。
    """
    from scipy.optimize import minimize_scalar
    n = len(y)
    p = X_poly.shape[1]
    K = Z.shape[1]
    Ik = np.eye(K)

    def neg2loglik(log_rho):
        rho = np.exp(log_rho)
        M = rho * Ik + Z.T @ Z
        Minv = np.linalg.inv(M)
        V_inv = (1.0 / rho) * (np.eye(n) - Z @ Minv @ Z.T)
        logdetV = (n - K) * np.log(rho) + np.linalg.slogdet(M)[1]
        XVX = X_poly.T @ V_inv @ X_poly
        _, logdetXVX = np.linalg.slogdet(XVX)
        beta = np.linalg.solve(XVX, X_poly.T @ V_inv @ y)
        resid = y - X_poly @ beta
        Q = float(resid @ V_inv @ resid)
        if method == "REML":
            return (n - p) * np.log(Q) + logdetV + logdetXVX
        else:
            return n * np.log(Q / n) + logdetV

    res = minimize_scalar(neg2loglik,
                          bounds=(np.log(1e-4), np.log(1e4)),
                          method="bounded")
    rho_opt = np.exp(res.x)
    return float(np.sqrt(rho_opt))

# ---- 二分搜 log 尺度参数，使 df 达到目标 ----
def bisect_log_df(df_fn, target, lo, hi, tol=1e-2, maxiter=80):
    for _ in range(maxiter):
        mid = np.exp((np.log(lo) + np.log(hi)) / 2)
        d = df_fn(mid)
        if abs(d - target) < tol:
            return mid
        if d > target:
            lo = mid
        else:
            hi = mid
    return mid

# ---- 安全求逆 ----
def safe_inv(M):
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)

# ---- 三次样条基 & 一阶导基 ----
def cubic_spline_basis(x, knots):
    x = np.asarray(x, dtype=float)
    cols = [np.ones_like(x), x, x**2, x**3]
    for k in knots:
        cols.append(np.maximum(x - k, 0.0) ** 3)
    return np.column_stack(cols)

def cubic_deriv_basis(x, knots):
    x = np.asarray(x, dtype=float)
    cols = [np.zeros_like(x), np.ones_like(x), 2*x, 3*x**2]
    for k in knots:
        cols.append(3 * np.maximum(x - k, 0.0) ** 2)
    return np.column_stack(cols)

# ---- 三次样条拟合（惩罚 λ^6 D，p=3） ----
def fit_cubic_penalty(x, y, knots, lam):
    n = len(x)
    K = len(knots)
    C = cubic_spline_basis(x, knots)
    D = np.zeros((4 + K, 4 + K))
    for i in range(K):
        D[4 + i, 4 + i] = 1.0
    A = C.T @ C + (lam ** 6) * D
    Ainv = np.linalg.inv(A)
    beta = Ainv @ C.T @ y
    yhat = C @ beta
    S = C @ Ainv @ C.T
    df = float(np.trace(S))
    rss = float(np.sum((y - yhat) ** 2))
    sigma2 = rss / (n - df)
    gcv = rss / (1 - df / n) ** 2
    return {
        "beta": beta, "df": df, "rss": rss, "gcv": gcv,
        "sigma2": sigma2, "Ainv": Ainv, "C": C,
    }

# ---- 带逐结点不同 λ 的三次样条（空间自适应） ----
def fit_cubic_local(x, y, knots, lam_per_knot):
    n = len(x)
    K = len(knots)
    C = cubic_spline_basis(x, knots)
    D = np.zeros((4 + K, 4 + K))
    for i in range(K):
        D[4 + i, 4 + i] = lam_per_knot[i] ** 6
    A = C.T @ C + D
    Ainv = np.linalg.inv(A)
    beta = Ainv @ C.T @ y
    yhat = C @ beta
    S = C @ Ainv @ C.T
    df = float(np.trace(S))
    rss = float(np.sum((y - yhat) ** 2))
    sigma2 = rss / (n - df)
    gcv = rss / (1 - df / n) ** 2
    return {
        "beta": beta, "df": df, "rss": rss, "gcv": gcv,
        "sigma2": sigma2, "Ainv": Ainv, "C": C,
    }

def fit_weighted(C, y, D, lam, w, p=1):
    """加权惩罚样条：min ||sqrt(W)(y-Cb)||² + lam^(2p) b'Db"""
    W = np.diag(w)
    A = C.T @ W @ C + (lam ** (2 * p)) * D
    Ainv = np.linalg.inv(A)
    beta = Ainv @ C.T @ W @ y
    yhat = C @ beta
    return beta, yhat, Ainv

def ig_sample(shape, rate, rng):
    """从 IG(shape, rate) 抽样：1 / Gamma(shape, scale=1/rate)"""
    z = rng.gamma(shape=shape, scale=1.0 / rate)
    return 1.0 / z
