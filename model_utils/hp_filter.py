import torch


def construct_A(m, lam=1.0, device="cpu"):
    diag = torch.eye(m, device=device)
    d = diag[:-2, :] - 2 * diag[1:-1, :] + diag[2:, :]

    A = diag + 2 * lam * d.T @ d

    return A

def construct_A2(m, lam1=1.0, lam2=1.0, device="cpu"):
    diag = torch.eye(m, device=device)
    d1 = diag[:-1, :] - diag[1:, :]
    d2 = diag[:-2, :] - 2 * diag[1:-1, :] + diag[2:, :]

    A = diag + 2 * lam1 * d1.T @ d1 + lam2 * d2.T @ d2

    return A

def symmetric_df(A):
    m = A.shape[0]
    L = torch.zeros((m, m)).to(A)
    D = torch.zeros(m).to(A)
    for k in range(0, m):
        L[k, k] = 1
        D[k] = A[k, k] - D[:k] @ (L[k, :k] ** 2)
        for j in range(k+1, m):
            L[j, k] = (A[j, k] - torch.sum(L[j, :k] * D[:k] * L[k, :k])) / D[k]

    return L, D

def gaussian_eliminate_L(L, y):
    m = L.shape[0] - 1
    z = y.clone()
    L = L.unsqueeze(0)
    for i in range(m):
        start = i+1
        end = min(m+1, i+3)
        z[:, start:end] -= L[:, start:end, i] * z[:, i].unsqueeze(-1)
        # for j in range(i+1, i+3):
        #     z[:, j] -= L[:, j, i] * z[:, i]
    return L, z

def gaussian_eliminate_U(U, b):
    m = U.shape[0] - 1
    x = b.clone()
    U = U.unsqueeze(0)
    for i in range(m+1):
        # coeff = U[m-i, m-i]
        # U[m-i] /= coeff
        # b[m-i] /= coeff
        start = max(m-i-2, 0)
        end = m-i
        x[:, start:end] -= U[:, start:end, m-i] * x[:, m - i].unsqueeze(-1)

    return U, x


def step_LD_part(A_prev, L_prev, D_prev, sub_A=None):
    # input params
    # A_prev = A[-3, -5:]
    A_append = torch.nn.functional.pad(A_prev, (0, 1, 0, 1))
    # TODO: construct -3 -3 A
    if sub_A is None:
        A_append[-3:, -3:] = A_prev[-3:, -3:]
    else:
        A_append[-3:, -3:] = sub_A
    A_append = A_append[1:, 1:]
    L_append = torch.nn.functional.pad(L_prev, (0, 1, 0, 1))
    D_append = torch.nn.functional.pad(D_prev, (0, 1))

    for k in range(3):
        L_append[k, k+2] = 1
        D_append[k+2] = A_append[k, k+2] - D_append[k:k+2] @ (L_append[k, k:k+2] ** 2)
        for j in range(k+1, 3):
            L_append[j, k+2] = (A_append[j, k+2] - L_append[j, k+1] * D_append[k+1] * L_append[k, k+1]) / D_append[k+2]

    return A_append, L_append, D_append


def step_zx_sym_part(L, D, z):
    L_append = L.unsqueeze(0)
    z_append = z.clone()
    # forward
    # for i in range(1, 4):
    #     z_append[:, 2:4] -= L_append[:, 1:3, i] * z_append[:, i-1].unsqueeze(-1)
    for i in range(1, 4):
        for j in range(1, 3):
            if j <= i - 2:
                continue
            z_append[:, j+1] -= L_append[:, j, i] * z_append[:, i-1]

    x_update = z_append[:, -1] / D

    return x_update, L_append, z_append


# In-memory cache of the LDL^T factorization of A. A depends only on (warm_up_step, lam)
# and not on the data, so it is factorized once instead of on every batch.
# The output is bit-identical to recomputing it.
_LD_CACHE = {}


def _get_ALD(m, lam1, lam2, device):
    key = (m, float(lam1), float(lam2), str(device))
    if key not in _LD_CACHE:
        A = construct_A2(m, lam1=lam1, lam2=lam2, device=device)
        L, D = symmetric_df(A)
        _LD_CACHE[key] = (A, L, D)
    return _LD_CACHE[key]


def hp_filter(y, lam=1.0, warm_up_step=5, hidden=None):
    # y -> (B, T)
    bsz, T = y.shape
    device = y.device
    if hidden is None:
        assert 5 <= warm_up_step <= T
        # warm up
        A, L, D = _get_ALD(warm_up_step, lam, lam, device)
        I1, z = gaussian_eliminate_L(L, y[:, :warm_up_step])
        zs_invd = 1 / D * z
        I2, x = gaussian_eliminate_U(L.T, zs_invd)
        xs = x
        y_prev = y[:, warm_up_step-1].unsqueeze(-1)
    else:
        y_prev, z, A, L, D = hidden
        xs = torch.tensor([]).to(y)
    # iteration
    for t in range(warm_up_step, T):
        z = z[:, -4:]
        A = A[-3:, -5:]
        L = L[-2:, -4:]
        D = D[-4:]
        # step L and D (3, 5)
        A, L, D = step_LD_part(A, L, D)
        # step Gaussian elminate
        z = torch.cat([z[:, 1:-1], y_prev, y[:, t].unsqueeze(-1)], dim=-1)
        # forward
        x_update, _, z = step_zx_sym_part(L, D[-1], z)

        xs = torch.cat([xs, x_update.unsqueeze(-1)], dim=-1)

        y_prev = y[:, t].unsqueeze(-1)
    # y_prev = y[:, -1]
    z = z[:, -4:]
    A = A[-3:, -5:]
    L = L[-2:, -4:]
    D = D[-4:]

    return xs, (y_prev, z, A, L, D)
