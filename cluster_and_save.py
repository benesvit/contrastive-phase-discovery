import sys


from umap import UMAP
from hdbscan import HDBSCAN
from tqdm import tqdm
import numpy as np
import argparse

from pathlib import Path
from pzo.pzo_utils import get_total_order, get_order_parameter, get_order_parameter_rot
from pzo.pzo_utils import get_cross_coherence, get_cross_coherence_rot
from metric_learning.dataset import DataWithLabels

import numpy as np

# Define the Q vectors for the different order parameters

    
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
    v_tot = get_total_order(V)
    v_01  = get_order_parameter_rot(V, Q=Q01)
    v_11  = get_order_parameter_rot(V, Q=Q11)
    v_02  = get_order_parameter_rot(V, Q=Q02)
    v_22  = get_order_parameter_rot(V, Q=Q22)
    return [v_tot, v_01, v_11, v_02, v_22]

def get_cross_coherence_parameters_rot(P: np.ndarray, A: np.ndarray) -> list:
    c_00 = get_cross_coherence_rot(P, A, Q=Q00)
    c_01 = get_cross_coherence_rot(P, A, Q=Q01)
    c_11 = get_cross_coherence_rot(P, A, Q=Q11)
    c_02 = get_cross_coherence_rot(P, A, Q=Q02)
    c_22 = get_cross_coherence_rot(P, A, Q=Q22)
    return [c_00, c_01, c_11, c_02, c_22]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--o', type=str, default='', help='Filename to save results')
    parser.add_argument('--s', type=str, default='../data_pzo_8x8_sim_anneal',
                        help='Folder with source files for calculation and clustering')
    parser.add_argument('--sym', type=str, default='lower', help='Choose order param symmetries')

    args = parser.parse_args()

    folder = Path(args.s)

    if args.sym == 'lower':
        order_param_f = get_order_parameters
        coherence_f = get_cross_coherence_parameters
    elif args.sym == 'higher':
        order_param_f = get_order_parameters_rot
        coherence_f = get_cross_coherence_parameters_rot
    else:
        raise ValueError(f'Unsupported argument: {args.sym}')

    print(f'Calculating order parameters from:  {folder.absolute()}')
    Values, X = [], []
    Configurations, Energies = [], []
    files = list(folder.iterdir())
    for file in tqdm(files):
        if file.is_file():
            # Load data
            data = np.load(file)
            
            # Extract alpha and kappa from the data
            alpha = data["alpha"]
            kappa = data["kappa"]

            # Extract polarization and hidden field from the data
            P0, A0 = data["P"], data["A"]
            M, N, D = P0.shape
            num_sites = M * N
            
            # Energy per site
            ener = data["energy"] / num_sites

            # Calculate pseudo-cubic coordinates
            sqrt2 = np.sqrt(2.)
            R, S = P0[..., 0], P0[..., 1]
            V, U = (S + R) / sqrt2, (S - R) / sqrt2
            Phi, Psi = A0[..., 0], A0[..., 1]
            Xi, Eta = (Psi + Phi) / sqrt2, (Psi - Phi) / sqrt2
            
            # Polarization and hidden field in pseudo-cubic coordinates
            P = np.stack([U, V], axis=-1)
            A = np.stack([Eta, Xi], axis=-1)
            
            # Compute ordering parameters
            p_order_params = order_param_f(P)
            a_order_params = order_param_f(A)
            cross_coherence = coherence_f(P, A)

            # Append values to lists
            Values.append(np.hstack([alpha, kappa]))
            X.append([*p_order_params, *a_order_params, *cross_coherence])
            # X.append([*p_order_params, *a_order_params, ])
            Configurations.append(np.dstack([P, A]))
            Energies.append(ener)

    # Convert lists to numpy arrays
    X              = np.array(X)
    Values         = np.array(Values)
    Configurations = np.array(Configurations)
    Energies       = np.array(Energies)

    print('Clustering...')


    umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42)
    X_umap = umap.fit_transform(X)
    # Clustering
    hd = HDBSCAN(min_cluster_size=20, cluster_selection_epsilon=2.0, cluster_selection_method='eom').fit(X_umap)

    Labels = hd.labels_

    data = DataWithLabels(
        Configurations,
        Values,
        X,
        Energies,
        Labels,
    )

    output_folder = Path('./clustered_data')
    output_folder.mkdir(parents=True, exist_ok=True)
    output_filename = args.o
    if output_filename == '':
        output_filename = 'data_' + args.sym
    output_file = output_folder / output_filename
    data.to_npz(output_file)
    print('Saved results as:', output_file.absolute())