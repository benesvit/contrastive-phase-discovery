import numpy as np
from pathlib import Path
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.metrics import adjusted_rand_score

def get_total_order(P: np.ndarray) -> float:
    """
    Calculate total ordering parameter for 2D vector field P.
    """

    return float(np.linalg.norm(P.mean(axis=(0,1))))

def get_structure_factor(P: np.ndarray, Q: np.ndarray = np.array([np.pi, np.pi]), normalize: bool = True) -> np.ndarray:
    """
    Vector structure factor S(Q) for a 2D polarization field.

    Parameters
    ----------
    P : array_like, shape (M, N, 2)
        Polarization vectors with P[...,0]=u and P[...,1]=v.
    Q : (float, float)
        Wave vector (Qx, Qy) in radians; lattice spacing a=1.
    normalize : bool, default True
        If True, divide by the number of lattice sites (M*N).

    Returns
    -------
    S : ndarray, shape (2,), complex128
        Complex 2D vector S(Q) = [Sx(Q), Sy(Q)].
        |S(Q)| is an origin-independent order-parameter amplitude.
    """
    
    # Check input
    P = np.asarray(P)
    assert P.ndim == 3 and P.shape[2] == 2, "P must have shape (M, N, 2)"

    # Set variables
    M, N, _ = P.shape
    num_sites = M * N
    Qx, Qy = float(Q[0]), float(Q[1])

    x = np.arange(M)[:, None]  # (M,1)
    y = np.arange(N)[None, :]  # (1,N)
    
    # Calculate phase factor for each site
    phase = np.exp(1.0j * (Qx * x + Qy * y))

    # Structure factor components
    Sx = np.sum(P[..., 0] * phase, dtype=np.complex128)
    Sy = np.sum(P[..., 1] * phase, dtype=np.complex128)

    if normalize:
        Sx /= num_sites
        Sy /= num_sites

    return np.array([Sx, Sy], dtype=np.complex128)

import numpy as np

def get_order_parameter(P: np.ndarray, Q: np.ndarray, normalize: bool = True) -> float:
    """
    Root-sum-of-squares order parameter using the four sign variants (±Qx, ±Qy).

    Parameters
    ----------
    P : ndarray, shape (Nx, Ny, 2)
        Polarization field with P[...,0]=u and P[...,1]=v.
    Q : tuple(float, float)
        Target wave vector (Qx, Qy) in radians.
    normalize : bool, default True
        If True, divide each structure factor by Nx*Ny.

    Returns
    -------
    float
        sqrt( sum_{sx=±1, sy=±1} |S(P; sx*Qx, sy*Qy)|^2 ),
        where S is the complex 2D structure factor of (u,v).
    """
    
    P = np.asarray(P)
    Nx, Ny, _ = P.shape
    x = np.arange(Nx)[:, None]
    y = np.arange(Ny)[None, :]
    Qx, Qy = float(Q[0]), float(Q[1])

    total = 0.0
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            phase = np.exp(1j * (sx*Qx * x + sy*Qy * y))  # (Nx,Ny)
            Su = np.sum(P[..., 0] * phase)
            Sv = np.sum(P[..., 1] * phase)
            if normalize:
                Su /= (Nx * Ny)
                Sv /= (Nx * Ny)
            total += (abs(Su)**2 + abs(Sv)**2).real
    
    return float(np.sqrt(total))

def get_cross_coherence(P, A, Q, aggregate="power_weighted", normalize=True, eps=1e-12):
    """
    Cross-coherence between P and A at the four sign variants (±Qx, ±Qy).

    Parameters
    ----------
    P, A : ndarray, shape (Nx, Ny, 2)
        Vector fields with [...,0]=u and [...,1]=v.
    Q : tuple(float, float)
        Target wave vector (Qx, Qy) in radians.
    aggregate : {'power_weighted','mean','max'}
        How to combine the four coherence values.
    normalize : bool
        If True, divide each structure factor by Nx*Ny.
    eps : float
        Threshold to avoid division by zero.

    Returns
    -------
    float
        Cross-coherence in [0,1], aggregated over (±Qx, ±Qy).
    """
    
    P = np.asarray(P); A = np.asarray(A)
    Nx, Ny, _ = P.shape
    x = np.arange(Nx)[:, None]
    y = np.arange(Ny)[None, :]
    Qx, Qy = float(Q[0]), float(Q[1])

    cohs, weights = [], []

    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            phase = np.exp(1j * (sx*Qx * x + sy*Qy * y))  # (Nx,Ny)

            # complex 2D structure factors at this (sx,sy)
            SP = np.array([
                np.sum(P[..., 0] * phase),
                np.sum(P[..., 1] * phase)
            ], dtype=np.complex128)

            SA = np.array([
                np.sum(A[..., 0] * phase),
                np.sum(A[..., 1] * phase)
            ], dtype=np.complex128)

            if normalize:
                SP /= (Nx * Ny)
                SA /= (Nx * Ny)

            nP = np.linalg.norm(SP)
            nA = np.linalg.norm(SA)
            if nP < eps or nA < eps:
                continue  # skip near-zero signal

            # coherence in [0,1]
            coh = float(np.abs(np.vdot(SP, SA)) / (nP * nA))
            w = float(nP * nA)  # power weight

            cohs.append(coh)
            weights.append(w)

    if not cohs:
        return 0.0

    cohs = np.array(cohs); weights = np.array(weights)

    if aggregate == "power_weighted":
        return float((cohs * weights).sum() / (weights.sum() + eps))
    elif aggregate == "mean":
        return float(cohs.mean())
    elif aggregate == "max":
        return float(cohs.max())
    else:
        raise ValueError("aggregate must be 'power_weighted', 'mean', or 'max'")


# ============================================================================
# ================== REDEFINED WITH ROTATIONAL SYMMETRY ======================

rot90 = np.array([
        [0, 1],
        [-1, 0]
    ]
)
def get_order_parameter_rot(P: np.ndarray, Q: np.ndarray, normalize: bool = True) -> float:
    """
    Root-sum-of-squares order parameter using the four sign variants (±Qx, ±Qy).

    Parameters
    ----------
    P : ndarray, shape (Nx, Ny, 2)
        Polarization field with P[...,0]=u and P[...,1]=v.
    Q : tuple(float, float)
        Target wave vector (Qx, Qy) in radians.
    normalize : bool, default True
        If True, divide each structure factor by Nx*Ny.

    Returns
    -------
    float
        sqrt( sum_{sx=±1, sy=±1} |S(P; sx*Qx, sy*Qy)|^2 ),
        where S is the complex 2D structure factor of (u,v).
    """
    
    P = np.asarray(P)
    Nx, Ny, _ = P.shape
    x = np.arange(Nx)[:, None]
    y = np.arange(Ny)[None, :]

    total = 0.0
    # for sx in (1.0, -1.0):
    #     for sy in (1.0, -1.0):
    Q = Q.copy()
    for _ in range(4):
            Qx, Qy = float(Q[0]), float(Q[1])
            phase = np.exp(1j * (Qx * x + Qy * y))  # (Nx,Ny)
            Su = np.sum(P[..., 0] * phase)
            Sv = np.sum(P[..., 1] * phase)
            if normalize:
                Su /= (Nx * Ny)
                Sv /= (Nx * Ny)
            total += (abs(Su)**2 + abs(Sv)**2).real
            Q = Q @ rot90
    
    return float(np.sqrt(total))

def get_cross_coherence_rot(P, A, Q, aggregate="power_weighted", normalize=True, eps=1e-12):
    """
    Cross-coherence between P and A at the four sign variants (±Qx, ±Qy).

    Parameters
    ----------
    P, A : ndarray, shape (Nx, Ny, 2)
        Vector fields with [...,0]=u and [...,1]=v.
    Q : tuple(float, float)
        Target wave vector (Qx, Qy) in radians.
    aggregate : {'power_weighted','mean','max'}
        How to combine the four coherence values.
    normalize : bool
        If True, divide each structure factor by Nx*Ny.
    eps : float
        Threshold to avoid division by zero.

    Returns
    -------
    float
        Cross-coherence in [0,1], aggregated over (±Qx, ±Qy).
    """
    
    P = np.asarray(P); A = np.asarray(A)
    Nx, Ny, _ = P.shape
    x = np.arange(Nx)[:, None]
    y = np.arange(Ny)[None, :]

    Q = Q.copy()

    cohs, weights = [], []

    for _ in range(4):
        Qx, Qy = float(Q[0]), float(Q[1])
        phase = np.exp(1j * (Qx * x + Qy * y))  # (Nx,Ny)

        # complex 2D structure factors at this (sx,sy)
        SP = np.array([
            np.sum(P[..., 0] * phase),
            np.sum(P[..., 1] * phase)
        ], dtype=np.complex128)

        SA = np.array([
            np.sum(A[..., 0] * phase),
            np.sum(A[..., 1] * phase)
        ], dtype=np.complex128)

        if normalize:
            SP /= (Nx * Ny)
            SA /= (Nx * Ny)

        nP = np.linalg.norm(SP)
        nA = np.linalg.norm(SA)
        if nP < eps or nA < eps:
            continue  # skip near-zero signal

        # coherence in [0,1]
        coh = float(np.abs(np.vdot(SP, SA)) / (nP * nA))
        w = float(nP * nA)  # power weight

        cohs.append(coh)
        weights.append(w)
        Q = Q @ rot90

    if not cohs:
        return 0.0

    cohs = np.array(cohs); weights = np.array(weights)

    if aggregate == "power_weighted":
        return float((cohs * weights).sum() / (weights.sum() + eps))
    elif aggregate == "mean":
        return float(cohs.mean())
    elif aggregate == "max":
        return float(cohs.max())
    else:
        raise ValueError("aggregate must be 'power_weighted', 'mean', or 'max'")


Q00 = np.array([0, 0])
Q01 = np.array([0, np.pi])
Q10 = np.array([np.pi, 0])
Q11 = np.array([np.pi, np.pi])
Q02 = np.array([0, np.pi / 2])
Q20 = np.array([np.pi / 2, 0])
Q22 = np.array([np.pi / 2, np.pi / 2])

def get_order_parameters(V: np.ndarray) -> list:
    """
    Calculate ordering parameters for a 2D polarization field from the structure factor S(Q).
    
    Parameters:
    V : np.ndarray
        Polarization field in pseudo-cubic coordinates.
        
    Returns:
    list
        Array of ordering parameters.
    """
       
    v_tot = get_total_order(V)
    v_01  = get_order_parameter(V, Q=Q01)
    v_10  = get_order_parameter(V, Q=Q10)
    v_11  = get_order_parameter(V, Q=Q11)
    v_02  = get_order_parameter(V, Q=Q02)
    v_20  = get_order_parameter(V, Q=Q20)
    v_22  = get_order_parameter(V, Q=Q22)

    return [v_tot, v_01, v_10, v_11, v_02, v_20, v_22]

def get_cross_coherence_parameters(P: np.ndarray, A: np.ndarray) -> list:
    """
    Calculate the cross-coherence between polarization and antiferroelectric order parameters.
    
    Parameters:
    P : np.ndarray
        Polarization field in pseudo-cubic coordinates.
    A : np.ndarray
        Antiferroelectric order parameter field in pseudo-cubic coordinates.
    Q : tuple
        Q vector for the cross-coherence calculation.
        
    Returns:
    float
        Cross-coherence value.
    """
    c_00 = get_cross_coherence(P, A, Q=Q00)
    c_01 = get_cross_coherence(P, A, Q=Q01)
    c_10 = get_cross_coherence(P, A, Q=Q10)
    c_11 = get_cross_coherence(P, A, Q=Q11)
    c_02 = get_cross_coherence(P, A, Q=Q02)
    c_20 = get_cross_coherence(P, A, Q=Q20)  
    c_22 = get_cross_coherence(P, A, Q=Q22)

    return [c_00, c_01, c_10, c_11, c_02, c_20, c_22]

def get_order_parameters_rot(V: np.ndarray)->list:
    """
    Calculate ordering parameters for a 2D polarization field from the structure factor S(Q) using rotationally invariant version.
    
    Parameters:
    V : np.ndarray
        Polarization field in pseudo-cubic coordinates.
        
    Returns:
    list
        Array of ordering parameters.
    """
       
    v_tot = get_total_order(V)
    v_01  = get_order_parameter_rot(V, Q=Q01)
    v_11  = get_order_parameter_rot(V, Q=Q11)
    v_02  = get_order_parameter_rot(V, Q=Q02)
    v_22  = get_order_parameter_rot(V, Q=Q22)
    return [v_tot, v_01, v_11, v_02, v_22]

def get_cross_coherence_parameters_rot(P: np.ndarray, A: np.ndarray) -> list:
    """
    Calculate the cross-coherence between polarization and antiferroelectric order parameters using rotationally invariant version.
    
    Parameters:
    P : np.ndarray
        Polarization field in pseudo-cubic coordinates.
    A : np.ndarray
        Antiferroelectric order parameter field in pseudo-cubic coordinates.
    Q : tuple
        Q vector for the cross-coherence calculation.
        
    Returns:
    float
        Cross-coherence value.
    """
    c_00 = get_cross_coherence_rot(P, A, Q=Q00)
    c_01 = get_cross_coherence_rot(P, A, Q=Q01)
    c_11 = get_cross_coherence_rot(P, A, Q=Q11)
    c_02 = get_cross_coherence_rot(P, A, Q=Q02)
    c_22 = get_cross_coherence_rot(P, A, Q=Q22)
    return [c_00, c_01, c_11, c_02, c_22]

# Example usage
if __name__ == '__main__':
    data_pth = Path('data')
    data = np.load(data_pth / 'monte_carlo_configs.npz')

    V = data['Configurations']

    X = []
    for i, v in enumerate(V):
        p = v[..., :2]
        a = v[..., 2:]
        p_S = get_order_parameters_rot(p)
        a_S = get_order_parameters_rot(a)
        cross_c = get_cross_coherence_parameters_rot(p, a)
        X.append(np.concatenate([p_S, a_S, cross_c]))

    # Check that the saved descriptor values match
    X = np.array(X)
    assert np.allclose(X, data['X'])

    # The clustering can be performed as:
    X_umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42).fit_transform(X)
    labels = HDBSCAN(min_cluster_size=20, cluster_selection_epsilon=2.0, cluster_selection_method='eom').fit(X_umap).labels_

    # Check similarity of the clustering
    score = adjusted_rand_score(data['Labels'], labels)
    print(f'ARI vs stored Labels: {score:.3f}')

