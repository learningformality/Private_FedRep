from numpy.linalg import qr
import numpy.linalg as la
import numpy as np
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import csv

###############################################################################
# 1. Synthetic Data Generation
###############################################################################


def generate_synthetic_data(
    num_users=200,           # N
    num_train_samples=200,  # n_i for training
    num_test_samples=100,    # n_i for testing
    d=50,                    # Input dimension
    k=5,                     # Representation dimension
    noise_std=0.01,          # Std of label noise (R)
    subG_v=1.0               # Sub-Gaussian norm (Lambda) for v_i
):
    """
    Generate synthetic training, testing, and final data for num_users users.
    Each user's data is generated as:
      X_train_i, X_test_i, X_final_i ~ N(0, 1)
      U ~ Orthonormal columns (d x k matrix)
      v_i ~ N(0, 1) (k-dimensional vector)
      y_{train/test/final} = X_{train/test/final} @ U @ v_i + noise

    Returns:
      X_train_list, y_train_list, X_test_list, y_test_list,
      X_final_list, y_final_list, U, vs
    """
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    X_final_list = []
    y_final_list = []

    # Initialize U with random values and orthonormalize its columns
    U = torch.randn(d, k)
    U, _ = torch.linalg.qr(U)  # Orthonormalize U

    vs = [torch.randn(k) for _ in range(num_users)]

    vs = [torch.nn.functional.normalize(
        vs[i], p=2, dim=0) for i in range(len(vs))]

    for i in range(num_users):
        # Generate training data
        X_train_i = torch.randn(num_train_samples, d)
        y_train_i = X_train_i.mm(U).mv(vs[i])
        y_train_i += noise_std * torch.randn(num_train_samples)

        # Generate testing data
        X_test_i = torch.randn(num_test_samples, d)
        y_test_i = X_test_i.mm(U).mv(vs[i])
        y_test_i += noise_std * torch.randn(num_test_samples)

        # Generate final data
        X_final_i = torch.randn(num_train_samples, d)
        y_final_i = X_final_i.mm(U).mv(vs[i])
        y_final_i += noise_std * torch.randn(num_train_samples)

        # Append to lists
        X_train_list.append(X_train_i)
        y_train_list.append(y_train_i)
        X_test_list.append(X_test_i)
        y_test_list.append(y_test_i)
        X_final_list.append(X_final_i)
        y_final_list.append(y_final_i)

    return X_train_list, y_train_list, X_test_list, y_test_list, X_final_list, y_final_list, U, vs


def gaussian_noise_std(max_norm, epsilon, delta, T):
    """
    Computes the standard deviation σ for a Gaussian mechanism whose variance
    is given by:
        var = max_norm * sqrt(2 * log(1.25/delta)) / (num_users * epsilon).

    Returns:
        sigma (float): standard deviation for the Gaussian noise.
    """
    var = (max_norm ** 4) * 4 * T * np.log(1.25 / delta) / ((epsilon)**2)
    return np.sqrt(var)


def clip_vector(v, clip_norm):
    """
    Clips vector v to have L2 norm at most clip_norm.
    If ||v|| <= clip_norm, returns v unchanged; otherwise scales it.

    Used for per-sample clipping, if needed.
    """
    norm_v = np.linalg.norm(v)
    if norm_v > clip_norm:
        return (clip_norm / norm_v) * v
    return v


def ols_personal_model_update(U_t, X_user, y_user, portion='first'):
    """
    Computes the OLS solution for the local user model v_j using a subset of data:
    - portion='first' uses the first m/4 samples
    - portion='last'  uses the last m/4 samples
    or any other indexing logic as needed.

    Args:
        U_t:     Current embedding (d x k).
        X_user:  (m x d) array of user features.
        y_user:  (m,) array of user labels.
        portion: which subset of data to use for OLS (default: 'first').

    Returns:
        v_j: a (k,)-dim vector (the OLS solution in the embedded space).
    """
    m = X_user.shape[0]
    if portion == 'first':
        idx_start, idx_end = 0, m // 2
    elif portion == 'last':
        idx_start, idx_end = (3*m)//4, m
    else:
        # Custom logic or fallback
        idx_start, idx_end = 0, m // 4

    X_sub = X_user[idx_start:idx_end]  # shape (m/4, d)
    y_sub = y_user[idx_start:idx_end]  # shape (m/4,)

    # Embedded features Z = X_sub @ U_t  => (m/4, k)
    Z = X_sub @ U_t

    # OLS solution: v_j = (Z^T Z + ridge)^{-1} (Z^T y_sub)
    ridge = 1e-6 * np.eye(Z.shape[1])
    ZtZ = Z.T @ Z + ridge
    Zty = Z.T @ y_sub
    v_j = np.linalg.solve(ZtZ, Zty)  # shape (k,)

    return v_j


def dp_embedding_update(
    U_t,
    v_dict,
    X_train_list,
    y_train_list,
    user_subset,
    epsilon,
    delta,
    max_norm,
    clip_val_feature=1.0,
    clip_val_response=1.0,
    rng=np.random.default_rng(),
    k=1,
    m=1,
    T=1
):
    """
    Instantiates the differentially private update step so that:

      1) For each user i in user_subset and each sample j in [m]:
         - Compute the (d x k) outer product M_{i,j} = x_{i,j} v_{i,j}^T.
         - Vectorize it to get a (d*k,) vector, then clip its Euclidean norm
           at clip_val_feature to get W_{i,j}.
         - Clip y_{i,j} at clip_val_response.
         - Accumulate W_acc += W_{i,j} W_{i,j}^T and b_acc += y_{i,j}^tilde * W_{i,j}.

      2) Add Gaussian noise to (W_acc, b_acc) => (W_priv, b_priv).

      3) Solve the (d*k)-dimensional OLS problem:
            min_u  u^T W_priv u - 2 b_priv^T u,
         which has solution u = (W_priv + ridge)^{-1} b_priv.

      4) Reshape u into Z in R^{d x k}, compute QR to get an orthonormal
         embedding U_next in R^{d x k}.

      5) Return U_next and the noise scale sigma.

    Args:
        U_t:               Unused in this DP step (kept for API consistency).
        v_dict:            Dictionary {user_index: v_user}, each v_user in R^k.
        X_train_list:      List (or dict) of user feature arrays. For user i,
                           X_train_list[i] has shape (m_i, d).
        y_train_list:      List (or dict) of user label arrays. For user i,
                           y_train_list[i] has shape (m_i,).
        user_subset:       Subset of user indices to include in the DP update.
        epsilon, delta:    DP parameters for the Gaussian mechanism.
        max_norm:          Overall sensitivity scale used in computing sigma.
                           (User must ensure max_norm aligns with the L2
                            clipping strategy so that the final statistic
                            has at most 'max_norm' sensitivity.)
        clip_val_feature:  Clip threshold for each W_{i,j} in L2 norm.
        clip_val_response: Clip threshold for each y_{i,j}.
        rng:               NumPy random generator for reproducibility.
        k:                 Dimension of each v_user and rank of final embedding.

    Returns:
        U_next: (d x k) updated embedding with orthonormal columns.
        sigma:  The noise standard deviation used in the Gaussian mechanism.
    """
    ##################################################################
    # 1) Gather dimensions and initialize accumulators
    ##################################################################
    # Pick the first user in user_subset to infer dimension d and check v-dim.
    first_user = user_subset[0]
    d = X_train_list[first_user].shape[1]        # dimension of x_i
    # Optionally verify that all v_dict[u] have dimension k
    if v_dict[first_user].shape[0] != k:
        raise ValueError(
            "Mismatch between the 'k' parameter and dimension of v_dict entries.")

    # We'll accumulate in (d*k)-dim vector space
    W_acc = np.zeros((d*k, d*k))
    b_acc = np.zeros(d*k)
    clip_val_feature = max_norm
    clip_val_response = max_norm

    ##################################################################
    # 2) Define a helper for L2 clipping
    ##################################################################
    def l2_clip(vec, clip_norm):
        norm = np.linalg.norm(vec)
        if norm > clip_norm:
            vec = vec * (clip_norm / norm)
        return vec

    ##################################################################
    # 3) Accumulate W_acc and b_acc over all chosen users and samples
    ##################################################################
    for user_id in user_subset:
        X_user = X_train_list[user_id]   # shape: (m_user, d)
        y_user = y_train_list[user_id]   # shape: (m_user,)
        v_user = v_dict[user_id]         # shape: (k,)

        m_user = X_user.shape[0]
        for j in range(m_user):
            x_ij = X_user[j]             # shape (d,)
            y_ij = y_user[j]
            # (d x k) outer product, then vectorize => shape (d*k,)
            M_ij = np.outer(x_ij, v_user).reshape(d*k)
            # L2 clip W_{i,j}
            W_ij = l2_clip(M_ij, clip_val_feature)

            # Clip label
            y_tilde_ij = np.clip(y_ij, -clip_val_response, clip_val_response)

            # Accumulate
            W_acc += np.outer(W_ij, W_ij)   # shape (d*k, d*k)
            b_acc += y_tilde_ij * W_ij      # shape (d*k,)

    ##################################################################
    # 4) Add Gaussian noise to preserve DP: W_priv, b_priv
    ##################################################################
    # Compute noise scale. The user must ensure 'max_norm' is set so that
    # the sensitivity matches the sum of W_ij W_ij^T and sum of W_ij terms.
    sigma = gaussian_noise_std(max_norm, epsilon, delta, T)
    sigma = m*sigma
    # W_noise is (d*k x d*k)
    W_noise = rng.normal(loc=0.0, scale=sigma, size=(d*k, d*k))
    # To keep the noise matrix symmetric on average:
    W_noise = 0.5 * (W_noise + W_noise.T)

    # b_noise is (d*k,)
    b_noise = rng.normal(loc=0.0, scale=m*sigma, size=(d*k,))

    W_priv = (2/(m*len(user_subset))) * (W_acc + W_noise)
    b_priv = (2/(m*len(user_subset))) * (b_acc + b_noise)

    ##################################################################
    # 5) Solve the OLS system:  min_u u^T W_priv u - 2 b_priv^T u
    ##################################################################
    # Minimizer is (W_priv + λI)^{-1} b_priv
    ridge = 1e-8 * np.eye(d*k)  # tiny regularizer to avoid singularities
    try:
        u_sol = np.linalg.solve(W_priv + ridge, b_priv)
    except np.linalg.LinAlgError:
        u_sol = np.linalg.lstsq(W_priv + ridge, b_priv, rcond=None)[0]

    ##################################################################
    # 6) Reshape u_sol => Z in R^{d x k}, then QR => U_next in R^{d x k}
    ##################################################################
    Z_mat = u_sol.reshape(d, k)  # shape (d, k)

    # QR decomposition
    Q, _ = qr(Z_mat, mode='reduced')  # Q is (d, d)
    U_next = Q[:, :k]                 # keep first k columns => (d, k)

    return U_next, (2/(m*len(user_subset)))*(sigma)


def apriv_altmin_ols_dp(
    X_train_list,
    y_train_list,
    U_init,
    T,
    epsilon,
    delta,
    max_norm,
    subset_list=None,
    rng=np.random.default_rng(),
    clip_val_feature=1.0,
    clip_val_response=1.0,
    k=2
):
    """
    APriv-AltMin meta-algorithm using:
      - Ordinary Least Squares for personal updates
      - Gaussian Mechanism for DP embedding update

    For each iteration t, we use a *disjoint* subset of users. If 'subset_list'
    is not provided, we construct T disjoint subsets, each of ~ n/T users.

    Args:
        X_train_list:     List of length n, each is (m, d) features for user i
        y_train_list:     List of length n, each is (m,) labels for user i
        U_init:           Initial (d x k) embedding matrix
        T:                Number of outer iterations
        epsilon, delta:   DP parameters
        max_norm:         Factor controlling the noise scale (variance)
        subset_list:      Optional list [S_1, ..., S_T]; each S_t is a subset of user indices
                          If None, we default to T disjoint subsets each of size ~ n/T.
        rng:              RNG for reproducibility
        clip_val_feature: Clipping threshold for the W_{ij} "features"
        clip_val_response:Clipping threshold for y_{ij} "responses"
        k:                The embedding dimension

    Returns:
        U_priv: (d x k) final DP embedding
        v_dict: dictionary {user_i: (k,)} of final personal models
    """
    n = len(X_train_list)
    m, d = X_train_list[0].shape
    k = U_init.shape[1]

    # Generate disjoint subsets if not provided:
    if subset_list is None:
        subset_list = []
        # Simple partition of the range(n) into T chunks
        # chunk_size = int(np.ceil(n / T))
        chunk_size = T
        start = 0
        for t in range(T):
            end = min(start + chunk_size, n)
            # Construct subset for iteration t
            subset_list.append(range(start, end))
            start = end
            if start >= n:
                break
        # If T > n, we'll just get up to n subsets. If T < n but doesn't divide n
        # evenly, the last subset might be slightly larger or smaller.

    U_current = U_init.copy()

    # Initialize personal models: dictionary user i -> R^k
    v_dict = [np.zeros(k) for _ in range(n)]

    # Main loop
    for t in range(T):
        if t >= len(subset_list):
            # If we ended up with fewer subsets than T (e.g., T>n),
            # just break or reuse an empty subset
            print(f"No more subsets for iteration {t}, stopping early.")
            break

        # Personal OLS update on the subset for iteration t
        # for j in subset_list[t]:
        for j in range(n):
            X_j = X_train_list[j]
            y_j = y_train_list[j]
            v_dict[j] = ols_personal_model_update(
                U_current, X_j, y_j, portion='first')

        # DP embedding update on that same subset
        U_next, sigma = dp_embedding_update(
            U_current,
            v_dict,
            X_train_list,
            y_train_list,
            user_subset=range(n),
            # user_subset=subset_list[t]
            epsilon=epsilon,
            delta=delta,
            max_norm=max_norm,
            clip_val_feature=clip_val_feature,
            clip_val_response=clip_val_response,
            rng=rng,
            k=k,
            m=m,
            T=T
        )
        U_current = U_next

    print("Final noise std:", sigma)

    # Optionally, do a final personal OLS step with the final embedding
    for j in range(n):
        X_j = X_train_list[j]
        y_j = y_train_list[j]
        v_dict[j] = ols_personal_model_update(
            U_current, X_j, y_j, portion='first')

    return U_current, v_dict


###############################################################################
# 2. Helper functions for FedRep
###############################################################################


def forward_Uv(X, U, v):
    """
    Forward pass: (X @ U) @ v
    X: [n x d], U: [d x k], v: [k]
    returns: predictions [n]
    """
    return (X.mm(U)).mv(v)


def solve_ols_v(X, y, U):
    """
    Solve for v using Ordinary Least Squares (OLS):
      v = (A^T A)^(-1) A^T y
    where A = X U
    """
    A = X.mm(U)
    A_pinv = torch.pinverse(A)
    v = A_pinv.mv(y)
    return v


def compute_user_grad_U(X, y, U, v):
    """
    Compute gradient of user's loss w.r.t. the global representation U,
    with local v fixed. Quadratic loss is used.

    Returns: gradient dL/dU (torch.Tensor of shape [d x k]).
    """
    U.requires_grad_(True)
    pred = forward_Uv(X, U, v)
    loss = torch.mean((pred - y)**2)
    loss.backward()

    grad_U = U.grad.data.clone()
    U.grad = None
    U.requires_grad_(False)
    return grad_U


def clip(grad, max_norm):
    """
    Clip the gradient to L2 norm max_norm.
    """
    norm = grad.norm(2)
    if norm > max_norm:
        grad = grad * (max_norm / norm)
    return grad


###############################################################################
# 3. FedRep Training
###############################################################################


def train_fedrep_dp(
    X_train_fedrep_list, y_train_fedrep_list,
    d, k,
    num_rounds=100,         # T
    global_lr=0.1,          # LR for global U update
    max_grad_norm=1.0,      # clipping norm
    noise_std=1.0,          # user-level DP noise standard deviation
    fedrep_batches_list=None,
    U_init=torch.zeros(1, 1)
):
    """
    FedRep approach with user-level DP:
      - Initialize global U
      - For each round:
        1. Local update: fix U, solve for each v_u using OLS on FedRep batch
        2. Global update: fix v_u, get grad from each user -> clip & add noise -> update U
        3. Orthonormalize U

    Parameters:
        fedrep_batches_list: precomputed FedRep batches for each user
        noise_std: stdev of added Gaussian noise
    """
    device = torch.device("cpu")
    num_users = len(X_train_fedrep_list)

    # Initialize global representation U
    U = U_init
    U_param = nn.Parameter(U, requires_grad=False)

    # Initialize local vectors v_i for each user
    v_list = [torch.zeros(k, requires_grad=True, device=device)
              for _ in range(num_users)]

    for rd in range(num_rounds):
        # 1. Local updates for each user
        for u in range(num_users):
            '''
            batch_idx = rd % len(fedrep_batches_list[u])
            X_batch, y_batch = fedrep_batches_list[u][batch_idx]
            '''
            X_batch, y_batch = X_train_fedrep_list[u], y_train_fedrep_list[u]
            v_i = solve_ols_v(X_batch, y_batch, U_param)
            v_list[u].data = v_i  # store OLS solution

        # 2. Global update with DP
        grad_sum = torch.zeros_like(U_param.data)
        for u in range(num_users):
            '''
            batch_idx = rd % len(fedrep_batches_list[u])
            X_batch, y_batch = fedrep_batches_list[u][batch_idx]
            '''
            X_batch, y_batch = X_train_fedrep_list[u], y_train_fedrep_list[u]
            grad_u = compute_user_grad_U(X_batch, y_batch, U_param, v_list[u])
            grad_u = clip(grad_u, max_grad_norm)
            grad_sum += grad_u / num_users

        # Add user-level noise once (not per-user, but we treat the sum as 1 user-level query)
        grad_sum += torch.randn_like(grad_sum) * noise_std

        # Gradient descent step
        U_param.data = U_param.data - global_lr * grad_sum

        # 3. Orthonormalize U
        Q, _ = torch.linalg.qr(U_param.data)
        U_param.data = Q

        if (rd + 1) % 10 == 0 or rd == 0:
            print(f"FedRep Round {rd+1}/{num_rounds} completed.")

    return U_param.data, [v.detach() for v in v_list]


###############################################################################
# 4. Evaluation: MSE
###############################################################################


def compute_test_mse(U, v_list, X_test_list, y_test_list):
    """
    Compute average MSE across all users' test data.
    """
    mse_list = []
    loss_fn = nn.MSELoss()
    for u in range(len(X_test_list)):
        with torch.no_grad():
            pred = forward_Uv(X_test_list[u], U, v_list[u])
            mse_val = loss_fn(pred, y_test_list[u])
        mse_list.append(mse_val.item())
    return np.mean(mse_list)


###############################################################################
# 5. Local DP GD (per-user)
###############################################################################


def local_dp_gradient_descent(X_train, y_train, num_steps, noise_std, max_norm, learning_rate, gd_batches=None):
    """
    Perform Local Differential Privacy Gradient Descent for a single user.
    For demonstration, noise_std is set to 0 (since you might only do local
    clipping, or local noise). Adjust as needed.
    """
    d = X_train.size(1)
    w = torch.zeros(d, requires_grad=False)

    num_batches = len(gd_batches)

    for step in range(num_steps):
        # Select batch in a cyclic manner
        batch_idx = step % num_batches
        X_batch, y_batch = gd_batches[batch_idx]

        # Forward pass
        pred = X_batch.mv(w)
        loss = torch.mean((pred - y_batch) ** 2)
        # Compute gradient
        grad = (2.0 / X_batch.size(0)) * X_batch.t().mv(pred - y_batch)

        # Clip gradient
        grad = clip(grad, max_norm)

        # Add noise if we want local DP
        noise = torch.randn_like(grad) * noise_std
        grad_noisy = grad + noise

        # Update w
        w = w - learning_rate * grad_noisy

    return w


###############################################################################
# 6. Model Initialization via SVD
###############################################################################


def model_init(X_train_list, y_train_list, d, k, m):
    """
    Computes the top-k spectral initializer from the second moment matrix of
    (x_i * y_i). This is for demonstration only.
    Returns a (d x k) PyTorch tensor with orthonormal columns.
    """
    # Convert lists into NumPy arrays
    X = np.stack([x.numpy() for x in X_train_list], axis=0)  # (n, m, d)
    y = np.stack([y.numpy() for y in y_train_list], axis=0)  # (n, m)

    n = X.shape[0]
    if n == 0:
        raise ValueError("Empty training data.")

    # 1) XY[i, j, :] = y[i, j] * X[i, j, :]
    XY = X * y[:, :, None]

    # 2) A = sum of outer-products
    A = np.einsum('nmd,nme->de', XY, XY)  # shape (d, d)

    # 3) B = sum of y^2_i * x_i x_i^T
    B = np.einsum('nm,nmd,nme->de', y*y, X, X)

    # 4) M_hat
    factor = (1.0 / (m * (m - 1))) * (1.0 / n)
    M_hat = factor * (A - B)

    # 5) SVD
    U_svd, S_svd, V_svd = la.svd(M_hat, full_matrices=True)
    U_init = U_svd[:, :k]  # top-k

    return torch.from_numpy(U_init.astype(np.float32))


###############################################################################
# 7. NEW: DP FedAvg Implementation
###############################################################################


def train_fedavg_dp(
    X_train_list, y_train_list,
    num_rounds=5,             # Number of global rounds (same as FedRep T)
    local_epochs=1,           # Number of local epochs each user does
    global_lr=0.1,            # Global learning rate when aggregating
    max_grad_norm=1.0,        # Clipping norm
    noise_std=1.0,            # user-level DP noise
    init_w=None,               # Optional: initial global model in R^d
    batch_size=0
):
    """
    Standard FedAvg with user-level DP:
      - We have a global model w in R^d
      - For each round:
          * Each user does 'local_epochs' steps of gradient descent from w
          * The local update delta_i = w_i - w is clipped to L2 <= max_grad_norm
          * We average the deltas (1/N sum), add noise, and update the global w
    """

    num_users = len(X_train_list)
    d = X_train_list[0].shape[1]

    if init_w is None:
        w_global = torch.zeros(d)
    else:
        w_global = init_w.clone()

    for rnd in range(num_rounds):
        # Sum of local updates
        delta_sum = torch.zeros(d)

        for u in range(num_users):
            w_local = w_global.clone()

            # Perform local_epochs of (full-batch) gradient descent
            for _ in range(local_epochs):
                '''
                X_u = X_train_list[u][rnd*batch_size:(rnd+1)*batch_size]
                y_u = y_train_list[u][rnd*batch_size:(rnd+1)*batch_size]
                '''
                X_u, y_u = X_train_list[u], y_train_list[u]
                pred = X_u.mv(w_local)
                grad = (2.0 / X_u.shape[0]) * \
                    X_u.t().mv(pred - y_u)  # MSE gradient
                # Update local
                w_local = w_local - global_lr * grad

            # local update
            delta = w_local - w_global
            delta = clip(delta, max_grad_norm)  # user-level clipping
            delta_sum += delta

        # Average
        delta_avg = delta_sum / num_users

        # Add DP noise once at the aggregator
        delta_avg += torch.randn_like(delta_avg) * noise_std

        # Update global
        w_global = w_global + delta_avg
        '''
        if (rnd + 1) % 1 == 0:
            print(f"FedAvg Round {rnd+1}/{num_rounds} completed.")'''

    return w_global


def compute_test_mse_fedavg(
    w_global,
    X_test_list,
    y_test_list
):
    """Compute average MSE for a global model w_global across all users."""
    mse_list = []
    for u in range(len(X_test_list)):
        pred = X_test_list[u].mv(w_global)
        mse_val = torch.mean((pred - y_test_list[u]) ** 2).item()
        mse_list.append(mse_val)
    return np.mean(mse_list)


###############################################################################
# 8. Main experiment: vary epsilon
###############################################################################


def run_experiment_vary_epsilon():
    # -----------------
    # Hyperparameters
    # -----------------
    num_users = 20000
    num_train_samples = 5
    num_test_samples = 200
    d = 50                      # Input dimension
    k = 2                        # Representation dimension
    noise_std_data = 0.01
    subG_v = math.sqrt(k)

    # FedRep hyperparameters
    fedrep_num_rounds = 5
    fedrep_global_lr = 2.5
    fedrep_batch_size = int(num_train_samples / fedrep_num_rounds)

    # Local GD hyperparameters
    gd_num_steps = 5
    gd_learning_rate = 0.065
    gd_batch_size = int(num_train_samples / gd_num_steps)

    # APriv-AltMin hyperparameters
    apriv_T = 5          # number of alt-min iterations
    apriv_max_norm = 0.0001  # noise-sensitivity scale in dp_embedding_update

    # Privacy parameters
    delta = 1e-6
    eps_list = [1, 2, 4, 6, 8]

    # Gradient clipping
    max_grad_norm = 10.0

    # --------------------------------------------------
    # 1) Generate synthetic data
    # --------------------------------------------------
    print("Generating synthetic data...")
    (X_train_list, y_train_list,
     X_test_list, y_test_list,
     X_final_list, y_final_list,
     U_true, vs_true) = generate_synthetic_data(
        num_users=num_users,
        num_train_samples=num_train_samples,
        num_test_samples=num_test_samples,
        d=d,
        k=k,
        noise_std=noise_std_data,
        subG_v=subG_v
    )

    # --------------------------------------------------
    # 2) Compute initialization U_init
    # --------------------------------------------------
    print("Generating spectral initialization for FedRep...")
    U_init = model_init(X_train_list, y_train_list, d, k, num_train_samples)

    # --------------------------------------------------
    # 3) Create FedRep and Local GD batches
    #    (unchanged from your code)
    # --------------------------------------------------
    fedrep_batches_list = []
    gd_batches_list = []
    for u in range(num_users):
        indices = torch.randperm(num_train_samples)
        fedrep_indices = indices[:num_train_samples//2]
        gd_indices = indices[num_train_samples//2:]

        # FedRep batches
        fedrep_user_batches = [
            (X_train_list[u][i*fedrep_batch_size:(i+1)*fedrep_batch_size],
             y_train_list[u][i*fedrep_batch_size:(i+1)*fedrep_batch_size])
            for i in range(fedrep_num_rounds)
        ]
        fedrep_batches_list.append(fedrep_user_batches)

        # Local GD batches
        gd_user_batches = [
            (X_train_list[u][i*gd_batch_size:(i+1)*gd_batch_size],
             y_train_list[u][i*gd_batch_size:(i+1)*gd_batch_size])
            for i in range(gd_num_steps)
        ]
        gd_batches_list.append(gd_user_batches)

    # True MSE for reference
    def compute_test_mse_true():
        mse_list = []
        loss_fn = nn.MSELoss()
        for u in range(num_users):
            pred = forward_Uv(X_test_list[u], U_true, vs_true[u])
            mse_val = loss_fn(pred, y_test_list[u]).item()
            mse_list.append(mse_val)
        return np.mean(mse_list)

    mse_true = compute_test_mse_true()
    print(f"True MSE: {mse_true}")

    # --------------------------------------------------
    # 4) Train Non-Private FedRep (baseline)
    # --------------------------------------------------
    print("\nTraining Non-Private FedRep (noise_std=0)...")
    U_final_np, v_final_list_np = train_fedrep_dp(
        X_train_fedrep_list=X_train_list,
        y_train_fedrep_list=y_train_list,
        d=d, k=k,
        num_rounds=fedrep_num_rounds,
        global_lr=fedrep_global_lr,
        max_grad_norm=max_grad_norm,
        noise_std=0.0,  # no DP noise
        fedrep_batches_list=fedrep_batches_list,
        U_init=U_init
    )
    # Re-fit local v_i on X_final_list
    v_final_new_np = []
    for u in range(num_users):
        v_new = solve_ols_v(X_final_list[u], y_final_list[u], U_final_np)
        v_final_new_np.append(v_new.detach())

    mse_trained_np = compute_test_mse(
        U_final_np, v_final_new_np, X_test_list, y_test_list
    )
    print(f"Non-Private FedRep final MSE: {mse_trained_np}")

    # (e) APriv-AltMin (NEW)
    # ======================
    #  1) Convert Torch to NumPy
    X_np_list = [X_train_list[u].numpy() for u in range(num_users)]
    y_np_list = [y_train_list[u].numpy() for u in range(num_users)]
    U_init_np = U_init.detach().numpy()  # (d, k)

    #  2) Run the DP alt-min approach on training data
    U_priv, v_dict = apriv_altmin_ols_dp(
        X_train_list=X_np_list,
        y_train_list=y_np_list,
        U_init=U_init_np,
        T=apriv_T,
        epsilon=1,
        delta=1.25,
        max_norm=apriv_max_norm,
        subset_list=None,
        k=k
    )

    #  3) Final OLS with X_final_list, y_final_list
    #     to produce new personal vectors for each user
    v_final_apriv = []
    for u in range(num_users):
        X_final_np = X_final_list[u].numpy()
        y_final_np = y_final_list[u].numpy()

        Z_final = X_final_np @ U_priv  # shape (n_final, k)
        ridge = 1e-8 * np.eye(Z_final.shape[1])
        ZtZ = Z_final.T @ Z_final + ridge
        Zty = Z_final.T @ y_final_np
        try:
            v_sol = np.linalg.solve(ZtZ, Zty)
        except np.linalg.LinAlgError:
            v_sol, _, _, _ = np.linalg.lstsq(ZtZ, Zty, rcond=None)
        v_final_apriv.append(v_sol)

    #  4) Compute MSE on X_test_list, y_test_list using U_priv, v_final_apriv
    #     We replicate the "compute_test_mse" logic but in NumPy
    mse_list = []
    for u in range(num_users):
        X_test_np = X_test_list[u].numpy()    # (m_test, d)
        y_test_np = y_test_list[u].numpy()    # (m_test,)
        # shape (m_test,)
        pred_test = (X_test_np @ U_priv) @ v_final_apriv[u]
        mse_user = np.mean((pred_test - y_test_np)**2)
        mse_list.append(mse_user)

    mse_apriv_nonpriv = np.mean(mse_list)
    print(
        f"Non-private APriv-AltMin => MSE on test data: {mse_apriv_nonpriv}")

    # Prepare containers
    mse_results_fedrep = []
    mse_results_gd = []
    mse_results_agg = []
    mse_results_fedavg = []
    mse_results_apriv = []

    # --------------------------------------------------
    # 5) Vary epsilon and compare methods
    # --------------------------------------------------
    for eps in eps_list:
        # (a) DP FedRep
        noise_std_dp_fedrep = (
            max_grad_norm *
            np.sqrt(48 * fedrep_num_rounds * np.log(1.25 / delta)) /
            (num_users * eps)
        )
        print(f"\n[EPS={eps}] => DP FedRep, noise_std={
              noise_std_dp_fedrep:.4f}")
        U_final, v_final_list = train_fedrep_dp(
            X_train_fedrep_list=X_train_list,
            y_train_fedrep_list=y_train_list,
            d=d, k=k,
            num_rounds=fedrep_num_rounds,
            global_lr=fedrep_global_lr,
            max_grad_norm=max_grad_norm,
            noise_std=noise_std_dp_fedrep,
            fedrep_batches_list=fedrep_batches_list,
            U_init=U_init
        )
        # Final local OLS on X_final_list
        v_final_new = []
        for u in range(num_users):
            v_new = solve_ols_v(X_final_list[u], y_final_list[u], U_final)
            v_final_new.append(v_new.detach())

        mse_fedrep = compute_test_mse(
            U_final, v_final_new, X_test_list, y_test_list)
        mse_results_fedrep.append(mse_fedrep)
        print(f"DP FedRep MSE: {mse_fedrep}")

        noise_std_dp_fedrep = (
            max_grad_norm *
            np.sqrt(8 * fedrep_num_rounds * np.log(1.25 / delta)) /
            (num_users * eps)
        )

        # (b) Local DP GD
        noise_std_gd = (
            max_grad_norm *
            math.sqrt(8.0 * gd_num_steps * math.log(1.25 / delta)) /
            eps
        )
        print(f"Local DP GD, noise_std={noise_std_gd:.4f}")
        user_mse_list = []
        for u in range(num_users):
            gd_user_batches = gd_batches_list[u]
            X_train_gd = torch.cat([batch[0] for batch in gd_user_batches])
            y_train_gd = torch.cat([batch[1] for batch in gd_user_batches])

            w_local = local_dp_gradient_descent(
                X_train=X_train_gd,
                y_train=y_train_gd,
                num_steps=gd_num_steps,
                noise_std=0.0,  # or noise_std_gd if fully local DP
                max_norm=max_grad_norm,
                learning_rate=gd_learning_rate,
                gd_batches=gd_user_batches
            )
            # test
            pred_test = X_test_list[u].mv(w_local)
            mse_test = torch.mean((pred_test - y_test_list[u]) ** 2).item()
            user_mse_list.append(mse_test)

        mse_local_gd = np.mean(user_mse_list)
        mse_results_gd.append(mse_local_gd)
        print(f"Local DP GD MSE: {mse_local_gd}")

        # (e) APriv-AltMin (NEW)
        # ======================
        #  1) Convert Torch to NumPy
        X_np_list = [X_train_list[u].numpy() for u in range(num_users)]
        y_np_list = [y_train_list[u].numpy() for u in range(num_users)]
        U_init_np = U_init.detach().numpy()  # (d, k)
        
        #  2) Run the DP alt-min approach on training data
        U_priv, v_dict = apriv_altmin_ols_dp(
            X_train_list=X_np_list,
            y_train_list=y_np_list,
            U_init=U_init_np,
            T=apriv_T,
            epsilon=eps,
            delta=delta,
            max_norm=apriv_max_norm,
            subset_list=None,
            k=k
        )

        #  3) Final OLS with X_final_list, y_final_list
        #     to produce new personal vectors for each user
        v_final_apriv = []
        for u in range(num_users):
            X_final_np = X_final_list[u].numpy()
            y_final_np = y_final_list[u].numpy()

            Z_final = X_final_np @ U_priv  # shape (n_final, k)
            ridge = 1e-8 * np.eye(Z_final.shape[1])
            ZtZ = Z_final.T @ Z_final + ridge
            Zty = Z_final.T @ y_final_np
            try:
                v_sol = np.linalg.solve(ZtZ, Zty)
            except np.linalg.LinAlgError:
                v_sol, _, _, _ = np.linalg.lstsq(ZtZ, Zty, rcond=None)
            v_final_apriv.append(v_sol)

        #  4) Compute MSE on X_test_list, y_test_list using U_priv, v_final_apriv
        #     We replicate the "compute_test_mse" logic but in NumPy
        mse_list = []
        for u in range(num_users):
            X_test_np = X_test_list[u].numpy()    # (m_test, d)
            y_test_np = y_test_list[u].numpy()    # (m_test,)
            # shape (m_test,)
            pred_test = (X_test_np @ U_priv) @ v_final_apriv[u]
            mse_user = np.mean((pred_test - y_test_np)**2)
            mse_list.append(mse_user)

        mse_apriv = np.mean(mse_list)
        mse_results_apriv.append(mse_apriv)
        print(
            f"APriv-AltMin (final OLS on X_final) => MSE on test data: {mse_apriv}")

    with open("rundata.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epsilon", "LocalDPGD", "DPFedRep", "APrivAltMin"])
        for i, eps in enumerate(eps_list):
            writer.writerow([
                eps,
                mse_results_gd[i],
                mse_results_fedrep[i],
                mse_results_apriv[i]
            ])

    # --------------------------------------------------
    # 6) Plot
    # --------------------------------------------------
    # Local DP GD
    plt.plot(eps_list, mse_results_gd, marker='x',
             linestyle='-', color='g', label='Local Optimization')

    # Non-private FedRep (horizontal line)
    plt.axhline(y=mse_trained_np, color='r',
                linestyle='--', label='Non-private FedRep')

    # DP FedRep
    plt.plot(eps_list, mse_results_fedrep, marker='o',
             linestyle='-', color='b', label='Private FedRep')

    # APriv-AltMin
    plt.plot(eps_list, mse_results_apriv, marker='d',
             linestyle='-', color='orange', label='Priv-AltMin')

    # Set x-axis limits to range from 1 to 10
    plt.xlim(0.5, 8.5)

    # Adjust x-ticks to include only values from 1 to 10
    plt.xticks([eps for eps in eps_list if 1 <= eps <= 10])

    # Labels and Title
    plt.xlabel("User-level Epsilon")
    plt.ylabel("Population MSE")

    # Grid for better readability
    plt.grid(True, linestyle=':', color='gray', linewidth=0.5)

    # Place the legend above the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
               ncol=2, fancybox=True, shadow=True)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Display the plot
    plt.show()


if __name__ == "__main__":
    run_experiment_vary_epsilon()
