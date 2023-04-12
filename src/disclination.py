import numpy as np
import numpy.linalg as nlg
from numpy import sin, cos, pi
from scipy import linalg as slg
from tqdm import tqdm

from pathlib import Path
import os
import pickle as pkl

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


# Disclination Functions
def disc_dimensions(n_x: int):
    """

    Return the dimensions of the square lattice with the C4 section remove to form a disclination

    Parameters
    ----------
    n_x : int
        The side length of the disclinated lattice

    Returns
    -------
    bottom_width, top_width, left_height, right_height : ints
        The dimensions

    """
    if n_x % 2 == 0:
        bottom_width = n_x
        top_width = n_x // 2
        left_height = n_x
        right_height = n_x // 2

    else:
        bottom_width = n_x
        top_width = n_x // 2
        left_height = n_x
        right_height = n_x // 2 + 1

    return bottom_width, top_width, left_height, right_height


def disc_core_ind(n_x: int):
    """

    Returns the matrix index of the disclination core site

    Parameters
    ----------
    n_x : int
        The side length of the disclinated lattice

    """
    if n_x % 2 == 0:
        raise ValueError('Plaquette-centered disclinations have no core site (nx is even).')

    bottom_width, top_width, left_height, right_height = disc_dimensions(n_x)

    return (right_height - 1) * bottom_width + top_width


def ind_to_coord(n_x: int, ind: int):
    """

    Return the lattice coordinates of the provided matrix index

    Parameters
    ----------
    n_x : int
        The side length of the disclinated lattice
    
    ind : int
        The matrix index to be converted

    Returns
    -------
    x, y : ints
        The lattice coordinates

    """
    bottom_width, top_width, left_height, right_height = disc_dimensions(n_x)

    midpoint_ind = bottom_width * right_height

    if ind < midpoint_ind:
        x = ind % bottom_width
        y = ind // bottom_width
    elif ind > midpoint_ind:
        temp_ind = ind - midpoint_ind
        x = temp_ind % top_width
        y = (temp_ind // top_width) + right_height
    else:
        x = 0
        y = right_height

    return x, y


def bulk_ind_check(n_x: int, threshold: int, ind: int):
    """

    A boolean check of whether the provided matrix index is that of a bulk lattice site

    Parameters
    ----------
    n_x : int
        The side length of the disclinated lattice
    
    threshold : int
        The radial distance from the disclination boundary that is considered the surface

    ind : int
        The matrix index to be checked
        
    """
    if threshold > n_x // 2:
        raise ValueError(f"Parameter 'threshold'={threshold} must be less than half the width 'nx'={n_x}.")

    x, y = ind_to_coord(n_x, ind)

    if x < threshold or y < threshold:
        return False
    elif y < n_x // 2 and n_x - x <= threshold:
        return False
    elif x < n_x // 2 and n_x - y <= threshold:
        return False
    else:
        return True


def disc_num_sites(nx: int):
    """

    Return the number of sites in a disclinated lattice cross-section

    Parameters
    ----------
    n_x : int
        The side length of the disclinated lattice

    Returns
    -------
    n_xy : int
        The number of sites in the disclinated lattice
        
    """
    bottom_width, top_width, left_height, right_height = disc_dimensions(nx)
    return bottom_width * right_height + top_width * (left_height - right_height)


def x_hopping_matrix(n_x, core_hopping=False):
    bottom_width, top_width, left_height, right_height = disc_dimensions(n_x)

    x_hopping_sites = np.zeros(0)

    for ii in range(left_height - 1):
        if ii < right_height:
            x_hopping_sites = np.concatenate((x_hopping_sites, np.ones(bottom_width - 1, dtype=complex), (0,)))
        else:
            x_hopping_sites = np.concatenate((x_hopping_sites, np.ones(top_width - 1, dtype=complex), (0,)))

    x_hopping_sites = np.concatenate((x_hopping_sites, np.ones(top_width - 1, dtype=complex)))

    if not core_hopping and n_x % 2 == 1:
        x_hopping_sites[bottom_width * (right_height - 1) + top_width - 1] = 0
        x_hopping_sites[bottom_width * (right_height - 1) + top_width] = 0

    hop_mat = np.diag(x_hopping_sites, k=1)
    return hop_mat.T


def y_hopping_matrix(n_x, core_hopping=False):
    bottom_width, top_width, left_height, right_height = disc_dimensions(n_x)

    y_hopping_sites_1 = np.concatenate((np.ones(bottom_width * (right_height - 1) + top_width, dtype=complex),
                                        np.zeros(top_width * (left_height - right_height - 1), dtype=complex)))
    y_hopping_sites_2 = np.concatenate((np.zeros(bottom_width * right_height, dtype=complex),
                                        np.ones(top_width * (left_height - right_height - 1), dtype=complex)))

    if not core_hopping and n_x % 2 == 1:
        y_hopping_sites_1[bottom_width * (right_height - 2) + top_width] = 0

    hop_mat = np.diag(y_hopping_sites_1, k=n_x) + np.diag(y_hopping_sites_2, k=n_x // 2)
    return hop_mat.T


def xy_hopping_matrix(n_x):
    bottom_width, top_width, left_height, right_height = disc_dimensions(n_x)

    n_tot = disc_num_sites(n_x)

    hopping_matrix = np.zeros((n_tot, n_tot))

    for ii in range(right_height - 1):
        for jj in range(bottom_width - 1):
            hopping_matrix[(ii + 1) * bottom_width + jj + 1, ii * bottom_width + jj] = 1
            hopping_matrix[(ii + 1) * bottom_width + jj, ii * bottom_width + jj + 1] = -1

    offset = bottom_width * (right_height - 1)
    for ii in range(top_width - 1):
        hopping_matrix[offset + bottom_width + ii + 1, offset + ii] = 1
        hopping_matrix[offset + bottom_width + ii, offset + ii + 1] = -1

    offset = bottom_width * right_height
    for ii in range(left_height - right_height - 1):
        for jj in range(top_width - 1):
            hopping_matrix[offset + (ii + 1) * top_width + jj + 1, offset + ii * top_width + jj] = 1
            hopping_matrix[offset + (ii + 1) * top_width + jj, offset + ii * top_width + jj + 1] = -1

    return hopping_matrix


def disc_hopping_matrix(n_x, nnn=False):
    bottom_width, top_width, left_height, right_height = disc_dimensions(n_x)
    n_tot = disc_num_sites(n_x)

    hopping_matrix = np.zeros((n_tot, n_tot))

    num_disc_sites = min(bottom_width - top_width, left_height - right_height)

    if nnn:
        # x+y
        ind_1 = [bottom_width * right_height - ii - 2 for ii in range(num_disc_sites)]
        ind_2 = [n_tot - 1 - top_width * ii for ii in range(num_disc_sites)]

        for (ii, jj) in zip(ind_1, ind_2):
            hopping_matrix[ii, jj] = 1

        # x-y
        ind_1 = [bottom_width * right_height - ii - 1 for ii in range(num_disc_sites)]
        ind_2 = [n_tot - 1 - top_width * (ii + 1) for ii in range(num_disc_sites)]

        ind_2[-1] -= bottom_width - top_width

        for (ii, jj) in zip(ind_2, ind_1):
            hopping_matrix[ii, jj] = -1
    else:
        ind_1 = [bottom_width * right_height - ii - 1 for ii in range(num_disc_sites)]
        ind_2 = [n_tot - 1 - top_width * ii for ii in range(num_disc_sites)]

        for (ii, jj) in zip(ind_2, ind_1):
            hopping_matrix[ii, jj] = 1

    return hopping_matrix


def disc_bloch_hamiltonian(k_z: float, n_x: int, model_params: dict, core_hopping=False):
    """

    Return the Bloch Hamiltonian of the DSM model on a disclinated lattice that is periodic in z

    Parameters
    ----------
    k_z : float
        The crystal momentum along z
    n_x : int
        The side length of the disclinated lattice
    model_params : dict
        The parameters of the DSM model, including q, b_xy, g1, g2, c4_masses, and core_mu
    core_hoppings : bool
        A boolean flag indicating whether or not to include hopping terms to the core site

    Returns
    -------
    ham : np.ndarray
        The Hamiltonian matrix
        
    """
    # Build Hamiltonian blocks
    u_4 = slg.expm(1j * pi / 4 * np.identity(4)) @ slg.expm(-1j * pi / 4 * (np.kron(2 * sigma_0 - sigma_z, sigma_z)))
    u_4 = u_4.conj().T

    q = model_params['q']
    b_xy = model_params['b_xy']

    h_onsite = (cos(q) - 2 * b_xy - cos(k_z)) * gamma_3

    if 'c4_masses' in model_params:
        h_onsite += sin(k_z) * (model_params['c4_masses'][0] * gamma_4 + model_params['c4_masses'][1] * gamma_5)

    h_x = -1j / 2 * gamma_1 + 1 / 2 * b_xy * gamma_3
    h_y = -1j / 2 * gamma_2 + 1 / 2 * b_xy * gamma_3

    if 'g1' in model_params:
        h_x += 1 / 2 * model_params['g1'] * sin(k_z) * gamma_4
        h_y -= 1 / 2 * model_params['g1'] * sin(k_z) * gamma_4

    if 'g2' in model_params:
        h_xy = -1 / 4 * model_params['g2'] * sin(k_z) * gamma_5
    else:
        h_xy = 0 * gamma_5

    norb = 4

    # Arrange blocks into full Hamiltonian
    n_xy = disc_num_sites(n_x)

    ham = np.zeros((n_xy * norb, n_xy * norb), dtype=complex)

    # Onsite Hamiltonian
    ham += np.kron(np.identity(n_xy, dtype=complex), h_onsite)

    if 'core_mu' in model_params:
        core_inds = np.zeros(n_xy)
        core_inds[disc_core_ind(n_x)] = 1
        ham += np.kron(np.diag(core_inds), model_params['core_mu'] * np.identity(4))

    # X-Hopping
    x_hopping = np.kron(x_hopping_matrix(n_x, core_hopping=core_hopping), h_x)
    ham += x_hopping + x_hopping.conj().T

    # Y-Hopping
    y_hopping = np.kron(y_hopping_matrix(n_x, core_hopping=core_hopping), h_y)
    ham += y_hopping + y_hopping.conj().T

    # XY-Hopping
    xy_hopping = np.kron(xy_hopping_matrix(n_x), h_xy)
    ham += xy_hopping + xy_hopping.conj().T

    # Disclination Hopping
    disc_hopping = np.kron(disc_hopping_matrix(n_x, nnn=False), np.dot(nlg.inv(u_4), h_y))
    ham += disc_hopping + disc_hopping.conj().T
    #
    disc_hopping = np.kron(disc_hopping_matrix(n_x, nnn=True), np.dot(nlg.inv(u_4), h_xy))
    ham += disc_hopping + disc_hopping.conj().T

    return ham


def cdw_bloch_ham(k_z, n_x: int, model_params: dict, cdw_params: dict, core_hopping=False):
    norb = 4
    n_xy = disc_num_sites(n_x)
    
    n_cdw = cdw_params['n']
    delta_cdw = cdw_params['delta']
    phi_cdw = cdw_params['phi']
    cdw_matrix = np.kron(np.identity(n_xy), gamma_3)
    bond_centered = cdw_params['bond_centered']

    if delta_cdw.imag != 0:
        raise ValueError('The coefficient of the CDW must be a real number.')
    
    if not bond_centered and not np.isclose(cdw_matrix - cdw_matrix.conj().T, np.zeros_like(cdw_matrix)).all:
        raise ValueError('The CDW matrix must be Hermitian if onsite.')

    # Rescale kz to the folded BZ
    k_z = k_z / n_cdw
    
    h = np.zeros((n_cdw, n_cdw, norb * n_xy, norb * n_xy), dtype=complex)

    for ii in range(n_cdw):
        h[ii, ii] = disc_bloch_hamiltonian(k_z + 2 * pi * ii / n_cdw, n_x, model_params, core_hopping=core_hopping)

    if bond_centered:
        for ii in range(n_cdw - 1):
            h_cdw = delta_cdw / 2 * np.exp(-1j * phi_cdw) * (np.exp(-1j * (k_z + 2 * pi * ii / n_cdw)) * cdw_matrix + np.exp(1j * (k_z + 2 * pi * (ii + 1) / n_cdw)) * cdw_matrix.conj().T)

            h[ii + 1, ii] = h_cdw
            h[ii, ii + 1] = h_cdw.conj().T
        
        h_cdw = delta_cdw / 2 * np.exp(-1j * phi_cdw) * (np.exp(-1j * (k_z + 2 * pi * (n_cdw - 1) / n_cdw)) * cdw_matrix + np.exp(1j * k_z) * cdw_matrix.conj().T)

        h[0, -1] = h_cdw
        h[-1, 0] = h_cdw.conj().T
    else:
        h_cdw = delta_cdw * cdw_matrix * np.exp(-1j * phi_cdw)

        for ii in range(n_cdw - 1):
            h[ii + 1, ii] = h_cdw
            h[ii, ii + 1] = h_cdw.conj().T
   
        h[0, -1] += h_cdw
        h[-1, 0] += h_cdw.conj().T

    return np.reshape(np.transpose(h, (0, 2, 1, 3)), (norb * n_xy * n_cdw, norb * n_xy * n_cdw))


def disclination_hamiltonian_z_blocks(nx: int, params: dict, core_mu=None, core_hopping=False):
    # Build Hamiltonian blocks
    u_4 = slg.expm(1j * pi / 4 * np.identity(4)) @ slg.expm(-1j * pi / 4 * (np.kron(2 * sigma_0 - sigma_z, sigma_z)))
    u_4 = u_4.conj().T

    h_onsite = (params['m0'] - 2 * params['bxy'] - params['bz']) * gamma_3

    h_x = -1j / 2 * gamma_1 + 1 / 2 * params['bxy'] * gamma_3
    h_y = -1j / 2 * gamma_2 + 1 / 2 * params['bxy'] * gamma_3
    h_z = 1 / 2 * params['bz'] * gamma_3

    if 'c4_masses' in params:
        h_z += -1j / 2 * (params['c4_masses'][0] * gamma_4 + params['c4_masses'][1] * gamma_5)

    h_xz = -1j / 4 * params['g1'] * gamma_4
    h_yz = 1j / 4 * params['g1'] * gamma_4
    
    h_xyz = 1j / 8 * params['g2'] * gamma_5

    norb = 4

    # Arrange blocks into full Hamiltonian
    n_sites = disc_num_sites(nx)

    ham_0 = np.zeros((n_sites * norb, n_sites * norb), dtype=complex)
    ham_z = np.zeros((n_sites * norb, n_sites * norb), dtype=complex)

    # Onsite Hamiltonian
    ham_0 += np.kron(np.identity(n_sites, dtype=complex), h_onsite)

    if core_mu is not None:
        core_inds = np.zeros(n_sites)
        core_inds[disc_core_ind(nx)] = 1
        ham_0 += np.kron(np.diag(core_inds), core_mu * np.identity(4))

    # X-Hopping
    x_hopping = np.kron(x_hopping_matrix(nx, core_hopping=core_hopping), h_x)
    ham_0 += x_hopping + x_hopping.conj().T

    # Y-Hopping
    y_hopping = np.kron(y_hopping_matrix(nx, core_hopping=core_hopping), h_y)
    ham_0 += y_hopping + y_hopping.conj().T

    # XZ-Hopping
    xz_hopping = np.kron(x_hopping_matrix(nx, core_hopping=core_hopping), h_xz)
    ham_z += xz_hopping + xz_hopping.conj().T

    # YZ-Hopping
    yz_hopping = np.kron(y_hopping_matrix(nx, core_hopping=core_hopping), h_yz)
    ham_z += yz_hopping + yz_hopping.conj().T

    # XYz-Hopping
    xyz_hopping = np.kron(xy_hopping_matrix(nx), h_xyz)
    ham_z += xyz_hopping + xyz_hopping.conj().T

    # Disclination Hopping
    disc_hopping = np.kron(disc_hopping_matrix(nx, nnn=False), np.dot(nlg.inv(u_4), h_y))
    ham_0 += disc_hopping + disc_hopping.conj().T
    #
    disc_hopping = np.kron(disc_hopping_matrix(nx, nnn=True), np.dot(nlg.inv(u_4), h_xyz))
    ham_z += disc_hopping + disc_hopping.conj().T

    return ham_0, ham_z


def z_coord_disclination_hamiltonian(n_z: int, pbc: bool, n_cdw: int, delta_phi: float, phi_cdw: float, n_x: int, params: dict, core_mu=None, core_hopping=False, bond_centered=True):
    h0, hz = disclination_hamiltonian_z_blocks(n_x, params, core_mu, core_hopping)
    
    n_xy = disc_num_sites(n_x)

    ham = np.zeros(n_z, n_z, 4 * n_xy, 4 * n_xy)

    for ii in range(n_z):
        ham[ii, ii] = h0

    for ii in range(n_z - 1):
        ham[ii + 1, ii] = hz
        ham[ii, ii + 1] = hz.conj().T
    
    if pbc:
        ham[0, -1] = hz
        ham[-1, 0] = hz.conj().T

    if bond_centered:
        for ii in range(n_z - 1):
            h_cdw = np.abs(delta_phi) * cos( 2 * pi * ii / n_cdw + phi_cdw) * np.kron(np.identity(n_xy), gamma_3)

            ham[ii, ii + 1] += h_cdw
            ham[ii + 1, ii] += h_cdw.conj().T
        
        if pbc:
            h_cdw = np.abs(delta_phi) * cos( 2 * pi * (n_z - 1) / n_cdw + phi_cdw) * np.kron(np.identity(n_xy), gamma_3)

            ham[0, -1] = h_cdw
            ham[-1, 0] = h_cdw.conj().T
    else:
        for ii in range(n_z):
            h_cdw = 2 * np.abs(delta_phi) * cos( 2 * pi * ii / n_cdw + phi_cdw) * np.kron(np.identity(n_xy), gamma_3)

            ham[ii, ii] += h_cdw

    return np.reshape(np.transpose(ham, (0, 2, 1, 3)), (4 * n_xy * n_z, 4 * n_xy * n_z))


def calculate_disc_rho(n_kz: int, n_x: int, model_params: dict, cdw_params=None, core_mu=None, core_hopping=False, use_gpu=True, fname='ed_disclination_rho'):
    if use_gpu:
        import cupy as cp
        import cupy.linalg as clg

    if cdw_params is None:
        h_func = lambda k : disc_bloch_hamiltonian(k, n_x, model_params, core_mu=core_mu, core_hopping=core_hopping)
    else:
        h_func = lambda k : cdw_bloch_ham(k, n_x, model_params, cdw_params['n_cdw'], cdw_params['delta_cdw'], cdw_params['phi_cdw'], cdw_params['bond_centered'], core_mu=core_mu, core_hopping=core_hopping)
    
    norb = 4
    n_xy = disc_num_sites(n_x)

    if use_gpu:
        def h_to_rho(hamiltonian):
            rho = cp.zeros_like(n_xy)
            
            evals, evecs = clg.eigh(hamiltonian)
            
            for jj, energy in enumerate(evals):
                if energy <= 0:
                    wf = evecs[:, jj]
                    temp_rho = cp.reshape(cp.multiply(cp.conj(wf), wf), (-1, norb))
                    rho += cp.sum(temp_rho, axis=-1).real

            return rho.get()
    else:  
        def h_to_rho(hamiltonian):
            rho = np.zeros_like(n_xy)
            
            evals, evecs = nlg.eigh(hamiltonian)
            
            for jj, energy in enumerate(evals):
                if energy <= 0:
                    wf = evecs[:, jj]
                    temp_rho = np.reshape(np.multiply(np.conj(wf), wf), (-1, norb))
                    rho += np.sum(temp_rho, axis=-1).real
                    
            return rho
        
    kz_ax = np.linspace(0, 2 * pi, n_kz + 1)[:-1]
    dk = kz_ax[1] - kz_ax[0]

    rho = np.zeros(n_kz, n_xy)

    for ii, k_z in tqdm(enumerate(kz_ax)):
        if use_gpu:
            h = cp.asarray(h_func(k_z))
        else:
            h = h_func(k_z)

        rho[ii] = h_to_rho(h)

    results = rho
    params = (n_kz, n_x, model_params, cdw_params)
    data = (results, params)

    with open(data_dir / (fname + '.pickle'), 'wb') as handle:
        pkl.dump(data, handle)

    return rho


def calculate_disc_ldos(energy_axis: np.ndarray, eta: float, nkz: int, nx: int, model_params: dict, core_mu=None, core_hopping=False, fname='ed_disclination_ldos'):
    import cupy as cp
    import cupy.linalg as clg

    norb = 4

    kz_ax = np.linspace(0, 2 * pi, nkz + 1)[:-1]

    dk = kz_ax[1] - kz_ax[0]

    ldos = cp.zeros((disc_num_sites(nx), len(energy_axis)))

    for kz in tqdm(kz_ax):
        h = cp.asarray(disc_bloch_hamiltonian(kz, nx, model_params, core_mu=core_mu,
                                                core_hopping=core_hopping))
        for ii, energy in enumerate(energy_axis):
            g0 = clg.inv((energy + 1j * eta) * cp.identity(h.shape[0]) - h)            
            ldos[:, ii] += cp.sum(cp.reshape(-cp.diag(g0).imag, (-1, norb)), axis=-1)

    results = ldos.get() / pi * dk
    params = (energy_axis, eta, nkz, nx, model_params)
    data = (results, params)

    with open(data_dir / (fname + '.pickle'), 'wb') as handle:
        pkl.dump(data, handle)

    return results


def calculate_bound_charge(nx: int, threshold: int, rho, subtract_avg=True, exclude_core=False):
    if subtract_avg:
        rho = rho - np.mean(rho)

    core_ind = disc_core_ind(nx)

    q_bound = 0

    for ii, q in enumerate(rho):
        if ii == core_ind and exclude_core:
            continue
        elif bulk_ind_check(nx, threshold, ii):
            q_bound += q

    return q_bound


def calculate_bound_ldos(nx: int, threshold: int, ldos, exclude_core=False):
    core_ind = disc_core_ind(nx)

    ldos_bound = np.zeros(ldos.shape[-1])

    for ii in range(ldos.shape[0]):
        if ii == core_ind and exclude_core:
            continue
        elif bulk_ind_check(nx, threshold, ii):
            ldos_bound += ldos[ii]

    return ldos_bound


def response_coef(m0: float, bz: float):
    arg = 1 - m0 / bz
    if np.abs(arg) > 1:
        raise ValueError("There are no Dirac nodes for these values of 'm0' and 'bz'.")
    else:
        return np.arccos(arg) / pi


def calculate_bound_charge_vs_nu(nkz: int, nx: int, m0: float, bxy: float, g1: float, g2: float,
                                 coef_min: float, coef_max: float, bz_pts: int,
                                 c4_masses=(0.0, 0.0), core_mu=None, core_hopping=False,
                                 use_gpu=True, data_folder_name=None):

    if data_folder_name is not None:
        os.makedirs(data_dir / data_folder_name, exist_ok=True)

    if coef_min < 0 or coef_min > 1 or coef_max < 0 or coef_max > 1:
        raise ValueError("The response coefficient must be between 0 and 1.")

    coef_ax = np.linspace(coef_max, coef_min, bz_pts)

    bz_ax = []
    for coef in coef_ax:
        bz_ax.append(m0 / (1 - cos(pi * coef)))

    print('Starting calculation...')
    for ii in range(len(bz_ax)):
        bz = bz_ax[ii]
        calculate_disc_rho(nkz, nx, m0, bxy, bz, g1, g2, c4_masses=c4_masses, core_mu=core_mu,
                                   core_hopping=core_hopping, use_gpu=use_gpu,
                                   fname=data_folder_name + f'/data_run_{ii}')
        print(f'Finished run {ii+1}/{len(bz_ax)}.\n')


def calculate_dq_dnu(nkz: int, nx: int, m0: float, bxy: float, g1: float, g2: float, coef_min: float, coef_max: float,
                     bz_pts: int, dbz: float, c4_masses=None, core_mu=None, use_gpu=True, data_folder_name=None):

    if data_folder_name is not None:
        os.makedirs(data_dir / data_folder_name, exist_ok=True)

    if coef_min < 0 or coef_min > 1 or coef_max < 0 or coef_max > 1:
        raise ValueError("The response coefficient must be between 0 and 1.")

    coef_ax = np.linspace(coef_max, coef_min, bz_pts)

    bz_ax = []
    for coef in coef_ax:
        bz_ax.append(m0 / (1 - cos(pi * coef)))

    print('Starting calculation...')
    for ii in range(len(bz_ax)):
        bz = bz_ax[ii]
        calculate_disc_rho(nkz, nx, m0, bxy, bz, g1, g2, c4_masses=c4_masses, core_mu=core_mu, use_gpu=use_gpu,
                                   fname=data_folder_name + f'/run_0_{ii}')
        calculate_disc_rho(nkz, nx, m0, bxy, bz + dbz, g1, g2, c4_masses=c4_masses, core_mu=core_mu,
                                   use_gpu=use_gpu, fname=data_folder_name + f'/run_1_{ii}')

        print(f'\n\nFinished run {ii+1}/{len(bz_ax)}.\n')
        # print(f'{response_coef(m0, bz)=}')
        # print(f'{response_coef(m0, bz+dbz)=}')
