import numpy as np
import numpy.linalg as nlg
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from pathlib import Path
import pickle as pkl
from tqdm import tqdm
from itertools import product
from joblib import Parallel, delayed

import src.utils as utils

# File structure
project_src = Path(__file__).parent
project_root = project_src.parent
styles_dir = project_root / 'matplotlib_styles'
data_dir = project_root / 'data'
figure_dir = project_root / 'figures'

# Define Pauli and Gamma matrices for convenience
sigma_0 = np.array([[1, 0], [0, 1]], dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

gamma_1 = np.kron(sigma_x, sigma_z)
gamma_2 = np.kron(-sigma_y, sigma_0)
gamma_3 = np.kron(sigma_z, sigma_0)
gamma_4 = np.kron(sigma_x, sigma_x)
gamma_5 = np.kron(sigma_x, sigma_y)


# Bloch Hamiltonians and plotting
def dsm_bloch_ham(k, model_params: dict):
    """
    Bloch Hamiltonian of a simple Dirac semi-metal.

    Parameters
    ----------
    k : tuple
        The crystal momentum three-vector
    params : dict
        A dictionary of the system parameters m_0, b_z, b_xy, g1, and g2
    q : float
        The location of the Dirac nodes on the kz axis (one at q, one at -q)

    Returns
    -------
    np.ndarray
        The 4x4 Bloch Hamiltonian matrix at the given momentum with the given parameters

    """
    kx, ky, kz = k
    
    q = model_params['q']
    b_xy = model_params['b_xy']

    h = sin(kx) * gamma_1 + sin(ky) * gamma_2 + (cos(kz) - cos(q) - b_xy * (2 - cos(kx) - cos(ky))) * gamma_3
            
    if 'g1' in model_params:
        h += model_params['g1'] * (cos(kx) - cos(ky)) * sin(kz) * gamma_4
    if 'g2' in model_params:
        h += model_params['g2'] * sin(kx) * sin(ky) * sin(kz) * gamma_5

    return h


def dsm_cdw_bloch_ham(k, model_params: dict, cdw_params: dict):
    """
    Bloch Hamiltonian of a simple Dirac semi-metal with a CDW along the z-axis

    Parameters
    ----------
    k : tuple
        The crystal momentum three-vector (the third component is in the reduced BZ of the system with the CDW)
    model_params : dict
        A dictionary of the system parameters m_0, b_z, b_xy, g1, and g2
    cdw_params : dict
        A dictionary of the cdw parameters n, delta, phi, matrix, and bond_centered

    Returns
    -------
    np.ndarray
        The 4nx4n Bloch Hamiltonian matrix of the system with a CDW at the given momentum with the given parameters

    """

    n_cdw = cdw_params['n']
    delta_cdw = cdw_params['delta']
    phi_cdw = cdw_params['phi']
    cdw_matrix = cdw_params['matrix']
    bond_centered = cdw_params['bond_centered']

    if delta_cdw.imag != 0:
        raise ValueError('The coefficient of the CDW must be a real number.')
    
    if not bond_centered and not np.isclose(cdw_matrix - cdw_matrix.conj().T, np.zeros_like(cdw_matrix)).all:
        raise ValueError('The CDW matrix must be Hermitian if onsite.')

    kx, ky, kz = k

    # Rescale kz to the folded BZ
    kz = kz / n_cdw

    h = np.zeros((n_cdw, n_cdw, 4, 4), dtype=complex)

    for ii in range(n_cdw):
        h[ii, ii] = dsm_bloch_ham((kx, ky, kz + 2 * pi * ii / n_cdw), model_params)

    if bond_centered:
        for ii in range(n_cdw - 1):
            h_cdw = delta_cdw / 2 * np.exp(-1j * phi_cdw) * (np.exp(-1j * (kz + 2 * pi * ii / n_cdw)) * cdw_matrix + np.exp(1j * (kz + 2 * pi * (ii + 1) / n_cdw)) * cdw_matrix.conj().T)

            h[ii + 1, ii] = h_cdw
            h[ii, ii + 1] = h_cdw.conj().T
        
        h_cdw = delta_cdw / 2 * np.exp(-1j * phi_cdw) * (np.exp(-1j * (kz + 2 * pi * (n_cdw - 1) / n_cdw)) * cdw_matrix + np.exp(1j * kz) * cdw_matrix.conj().T)

        h[0, -1] += h_cdw
        h[-1, 0] += h_cdw.conj().T

    else:
        h_cdw = delta_cdw * cdw_matrix * np.exp(-1j * phi_cdw)

        for ii in range(n_cdw - 1):
            h[ii + 1, ii] = h_cdw
            h[ii, ii + 1] = h_cdw.conj().T
   
        h[0, -1] += h_cdw
        h[-1, 0] += h_cdw.conj().T

    return np.reshape(np.transpose(h, (0, 2, 1, 3)), (4 * n_cdw, 4 * n_cdw))


def plot_band_structure(nk_tot, model_params: dict, cdw_params: dict, save=True, fig_fname=None):

    if cdw_params is None:
        if fig_fname is None:
            fig_fname='dsm_bands'

        def ham(k_vec) -> np.ndarray:
            return dsm_bloch_ham(k_vec, model_params)
    else:
        if fig_fname is None:
            fig_fname='dsm_cdw_bands'

        def ham(k_vec) -> np.ndarray:
            return dsm_cdw_bloch_ham(k_vec, model_params, cdw_params)

    k_ax_3d, k_nodes_3d, labels_3d = utils.high_symmetry_bz_path_3d(nk_tot)

    evals = []
    for k in k_ax_3d:
        evals.append(np.linalg.eigvalsh(ham(k)))

    plt.style.use(styles_dir / 'bands.mplstyle')

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(evals, 'b-', linewidth=2)

    ax.set_xticks(k_nodes_3d)
    ax.set_xticklabels(labels_3d)
    ax.set_xmargin(0)

    for k in k_nodes_3d[1:]:
        ax.axvline(k, color='k')

    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    return fig, ax, evals


# Z coordinate space Hamiltonians and plotting
def z_coord_hamiltonian_blocks(k, model_params: dict) -> np.ndarray:
    """
    Produces the onsite and z-hopping matrices for the DSM with the given parameters and kx, ky momentum

    Parameters
    ----------
    k: tuple
        The two-dimensional in-place crystal mometnum (kx, ky)
    model_params : dict
        A dictionary of system parameters q, b_xy, g1, and g2

    Returns
    -------
    h0 : np.ndarray
        The onsite Hamiltonian
    hz : np.ndarray
        The hopping matrix for the z-direction   

    """
    kx, ky = k

    b_xy = model_params['b_xy']
    q = model_params['q']

    h0  = sin(kx) * gamma_1 + sin(ky) * gamma_2 \
        - (cos(q) + b_xy * (2 - cos(kx) - cos(ky))) * gamma_3
    
    hz  = 1 / 2 * gamma_3

    if 'g1' in model_params:
        hz -= 1j / 2 * model_params['g1'] * (cos(kx) - cos(ky)) * gamma_4
    if 'g2' in model_params:
        hz -= 1j / 2 * model_params['g2'] * sin(kx) * sin(ky) * gamma_5

    return h0, hz


def z_coord_hamiltonian(k, n_z: int, pbc: bool, model_params: dict, cdw_params: dict):
    """
    The mixed coordinate-momentum space Hamiltonian with a CDW in the z-direction

    Parameters
    ----------
    k: tuple
        The two-dimensional in-place crystal mometnum (kx, ky)
    n_z : int
        The number of sites in the z-direction
    pbc: bool
        A boolean indicating the presence or absence of periodic boundary conditions
    model_params : dict
        A dictionary of system parameters m_0, b_z, b_xy, g1, and g2
    cdw_params : dict
        A dictionary of the cdw parameters n, delta, phi, matrix, and bond_centered
    surface_params: dict
        A dictionary of parameters for the PHS-breaking surface mass including the mass and mirror_sym

    Returns
    -------
    ham : np.ndarray
        The Hamiltonian matrix  

    """
    h0, hz = z_coord_hamiltonian_blocks(k, model_params)
    norb = h0.shape[0]

    ham = np.zeros((n_z, n_z, norb, norb), dtype=complex)

    for ii in range(n_z):
        ham[ii, ii] = h0

    for ii in range(n_z - 1):
        ham[ii + 1, ii] = hz
        ham[ii, ii + 1] = hz.conj().T

    if pbc and n_z != 2:
        ham[0, -1] = hz
        ham[-1, 0] = hz.conj().T

    if cdw_params is not None:
        n_cdw = cdw_params['n']
        delta_cdw = cdw_params['delta']
        phi_cdw = cdw_params['phi']
        cdw_matrix = cdw_params['matrix']
        bond_centered = cdw_params['bond_centered']

        if bond_centered:
            for ii in range(n_z - 1):
                h_cdw = delta_cdw * cdw_matrix * cos(2 * pi * ii / n_cdw + phi_cdw)
                
                ham[ii, ii + 1] += h_cdw
                ham[ii + 1, ii] += h_cdw.conj().T

            if pbc and n_z != 2:
                h_cdw = delta_cdw * cdw_matrix * cos(2 * pi * (n_z - 1) / n_cdw + phi_cdw)

                ham[-1, 0] += h_cdw
                ham[0, -1] += h_cdw.conj().T

        else:
            for ii in range(n_z):
                h_cdw = 2 * delta_cdw * cdw_matrix * cos(2 * pi * ii / n_cdw + phi_cdw)
                
                ham[ii, ii] += h_cdw

    ham = np.reshape(np.transpose(ham, (0, 2, 1, 3)), (norb * n_z, norb * n_z))

    return ham


def plot_z_coord_band_structure(nk_tot, n_z: int, pbc: bool, model_params: dict, cdw_params: dict, save=True,fig_fname=None):

    if fig_fname is None:
        if cdw_params is None:
            fig_fname='dsm_z_coord_bands'
        else:
            fig_fname='dsm_cdw_z_coord_bands'
        
    def ham(k_vec) -> np.ndarray:
        return z_coord_hamiltonian(k_vec, n_z, pbc, model_params, cdw_params)
    
    k_ax_2d, k_nodes_2d, labels_2d = utils.high_symmetry_bz_path_2d(nk_tot)

    evals = []
    for k in tqdm(k_ax_2d):
        evals.append(np.linalg.eigvalsh(ham(k)))

    plt.style.use(styles_dir / 'bands.mplstyle')

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(evals, 'b-', linewidth=2)

    ax.set_xticks(k_nodes_2d)
    ax.set_xticklabels(labels_2d)
    ax.set_xmargin(0)

    for k in k_nodes_2d[1:]:
        ax.axvline(k, color='k')

    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    return fig, ax, evals


# Not debugged
def calculate_bloch_dos(energy_axis: np.ndarray, eta: float, n_ks: tuple, model_params: dict, cdw_params: dict, fname=None):
    (kx_ax, ky_ax, kz_ax) = (np.linspace(0, 2 * pi, n_ks[ii] + 1)[:-1] for ii in range(3))
    dks = (kx_ax[1] - kx_ax[0], ky_ax[1] - ky_ax[0], kz_ax[1] - kz_ax[0])

    if cdw_params is None:
        if fname is None:
            fname='dsm_dos'

        def ham(k_vec) -> np.ndarray:
            return dsm_bloch_ham(k_vec, model_params)
    else:
        if fname is None:
            fname='dsm_cdw_bands'

        def ham(k_vec) -> np.ndarray:
            return dsm_cdw_bloch_ham(k_vec, model_params)

    def ldos_func(energy: float):
        temp = np.zeros(n_ks)

        for ii in range(n_ks[0]):
            for jj in range(n_ks[1]):
                for kk in range(n_ks[2]):
                    k = (kx_ax[ii], ky_ax[jj], kz_ax[kk])
                    h = ham(k)

                    g0 = np.linalg.inv((energy + 1j * eta) * np.identity(h.shape[0]) - h)            

                    temp[ii, jj, kk] = -np.sum(np.diag(g0).imag) / pi

        return temp

    ldos = np.stack(Parallel(n_jobs=-2)(delayed(ldos_func)(e) for e in energy_axis), 0)
    dos = np.sum(ldos, axis=(1, 2, 3)) * dks[0] * dks[1] * dks[2]

    results = (ldos, dos)
    params = (energy_axis, eta, model_params)
    data = (results, params)

    with open(data_dir / (fname + '.pickle'), 'wb') as handle:
        pkl.dump(data, handle)

    return results


# Not debugged
def open_z_dos(energy_axis, eta, n_k: int, n_z: int, model_params: dict, cdw_params: dict, fname='open_z_dos'):
    ks, k_nodes, k_labels = utils.high_symmetry_bz_path_2d(n_k)
    
    def ldos_func(energy: float):
        temp = np.zeros((len(ks), n_z))

        for ii, k in enumerate(ks):
            h = z_coord_hamiltonian(k, n_z, False, model_params, cdw_params)

            g0 = np.linalg.inv((energy + 1j * eta) * np.identity(h.shape[0]) - h)            
            temp[ii] = -np.sum(np.reshape(np.diag(g0), (-1, 4)), axis=-1).imag / pi

        return temp

    ldos = np.stack(Parallel(n_jobs=-2)(delayed(ldos_func)(e) for e in tqdm(energy_axis)), 0)

    results = ldos
    k_data = (ks, k_nodes, k_labels)
    params = (energy_axis, eta, n_k, n_z, model_params, cdw_params)
    data = (results, k_data, params)

    with open(data_dir / (fname + '.pickle'), 'wb') as handle:
        pkl.dump(data, handle)

    return results
