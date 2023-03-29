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
def disclination_dimensions(nx: int):

    if nx % 2 == 0:
        bottom_width = nx
        top_width = nx // 2
        left_height = nx
        right_height = nx // 2

    else:
        bottom_width = nx
        top_width = nx // 2
        left_height = nx
        right_height = nx // 2 + 1

    return bottom_width, top_width, left_height, right_height


def disc_core_ind(nx: int):
    if nx % 2 == 0:
        raise ValueError('Plaquette-centered disclinations have no core site (nx is even).')

    bottom_width, top_width, left_height, right_height = disclination_dimensions(nx)

    return (right_height - 1) * bottom_width + top_width


def ind_to_coord(nx: int, ind: int):
    bottom_width, top_width, left_height, right_height = disclination_dimensions(nx)

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


def bulk_ind(nx: int, threshold: int, ind: int):
    if threshold > nx // 2:
        raise ValueError(f"Parameter 'threshold'={threshold} must be less than half the width 'nx'={nx}.")

    x, y = ind_to_coord(nx, ind)

    if x < threshold or y < threshold:
        return False
    elif y < nx // 2 and nx - x <= threshold:
        return False
    elif x < nx // 2 and nx - y <= threshold:
        return False
    else:
        return True


def number_of_sites(nx: int):
    bottom_width, top_width, left_height, right_height = disclination_dimensions(nx)
    return bottom_width * right_height + top_width * (left_height - right_height)


def x_hopping_matrix(nx, core_hopping=False):
    bottom_width, top_width, left_height, right_height = disclination_dimensions(nx)

    x_hopping_sites = np.zeros(0)

    for ii in range(left_height - 1):
        if ii < right_height:
            x_hopping_sites = np.concatenate((x_hopping_sites, np.ones(bottom_width - 1, dtype=complex), (0,)))
        else:
            x_hopping_sites = np.concatenate((x_hopping_sites, np.ones(top_width - 1, dtype=complex), (0,)))

    x_hopping_sites = np.concatenate((x_hopping_sites, np.ones(top_width - 1, dtype=complex)))

    if not core_hopping and nx % 2 == 1:
        x_hopping_sites[bottom_width * (right_height - 1) + top_width - 1] = 0
        x_hopping_sites[bottom_width * (right_height - 1) + top_width] = 0

    hop_mat = np.diag(x_hopping_sites, k=1)
    return hop_mat.T


def y_hopping_matrix(nx, core_hopping=False):
    bottom_width, top_width, left_height, right_height = disclination_dimensions(nx)

    y_hopping_sites_1 = np.concatenate((np.ones(bottom_width * (right_height - 1) + top_width, dtype=complex),
                                        np.zeros(top_width * (left_height - right_height - 1), dtype=complex)))
    y_hopping_sites_2 = np.concatenate((np.zeros(bottom_width * right_height, dtype=complex),
                                        np.ones(top_width * (left_height - right_height - 1), dtype=complex)))

    if not core_hopping and nx % 2 == 1:
        y_hopping_sites_1[bottom_width * (right_height - 2) + top_width] = 0

    hop_mat = np.diag(y_hopping_sites_1, k=nx) + np.diag(y_hopping_sites_2, k=nx // 2)
    return hop_mat.T


def xy_hopping_matrix(nx):
    bottom_width, top_width, left_height, right_height = disclination_dimensions(nx)

    n_tot = number_of_sites(nx)

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


def disclination_hopping_matrix(nx, nnn=False):

    bottom_width, top_width, left_height, right_height = disclination_dimensions(nx)
    n_tot = number_of_sites(nx)

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


def disclination_hamiltonian(kz: float, nx: int, params: dict, core_mu=None, core_hopping=False):
    # Build Hamiltonian blocks
    u_4 = slg.expm(1j * pi / 4 * np.identity(4)) @ slg.expm(-1j * pi / 4 * (np.kron(2 * sigma_0 - sigma_z, sigma_z)))
    u_4 = u_4.conj().T

    h_onsite = (params['m0'] - 2 * params['bxy'] - params['bz'] * (1 - cos(kz))) * gamma_3

    if 'c4_masses' in params:
        h_onsite += sin(kz) * (params['c4_masses'][0] * gamma_4 + params['c4_masses'][1] * gamma_5)

    h_x = -1j / 2 * gamma_1 + 1 / 2 * params['bxy'] * gamma_3 + 1 / 2 * params['g1'] * sin(kz) * gamma_4
    h_y = -1j / 2 * gamma_2 + 1 / 2 * params['bxy'] * gamma_3 - 1 / 2 * params['g1'] * sin(kz) * gamma_4
    h_xy = -1 / 4 * params['g2'] * sin(kz) * gamma_5

    norb = 4

    # Arrange blocks into full Hamiltonian
    n_sites = number_of_sites(nx)

    ham = np.zeros((n_sites * norb, n_sites * norb), dtype=complex)

    # Onsite Hamiltonian
    ham += np.kron(np.identity(n_sites, dtype=complex), h_onsite)

    if core_mu is not None:
        core_inds = np.zeros(n_sites)
        core_inds[disc_core_ind(nx)] = 1
        ham += np.kron(np.diag(core_inds), core_mu * np.identity(4))

    # X-Hopping
    x_hopping = np.kron(x_hopping_matrix(nx, core_hopping=core_hopping), h_x)

    ham += x_hopping + x_hopping.conj().T

    # Y-Hopping
    y_hopping = np.kron(y_hopping_matrix(nx, core_hopping=core_hopping), h_y)
    ham += y_hopping + y_hopping.conj().T

    # XY-Hopping
    xy_hopping = np.kron(xy_hopping_matrix(nx), h_xy)
    ham += xy_hopping + xy_hopping.conj().T

    # Disclination Hopping
    disc_hopping = np.kron(disclination_hopping_matrix(nx, nnn=False), np.dot(nlg.inv(u_4), h_y))
    ham += disc_hopping + disc_hopping.conj().T
    #
    disc_hopping = np.kron(disclination_hopping_matrix(nx, nnn=True), np.dot(nlg.inv(u_4), h_xy))
    ham += disc_hopping + disc_hopping.conj().T

    return ham


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
    n_sites = number_of_sites(nx)

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
    disc_hopping = np.kron(disclination_hopping_matrix(nx, nnn=False), np.dot(nlg.inv(u_4), h_y))
    ham_0 += disc_hopping + disc_hopping.conj().T
    #
    disc_hopping = np.kron(disclination_hopping_matrix(nx, nnn=True), np.dot(nlg.inv(u_4), h_xyz))
    ham_z += disc_hopping + disc_hopping.conj().T

    return ham_0, ham_z


def z_coord_disclination_hamiltonian(nz: int, pbc: bool, n_cdw: int, delta_phi: float, phi_cdw: float, nx: int, params: dict, core_mu=None, core_hopping=False):
    h0, hz = disclination_hamiltonian_z_blocks(nx, params, core_mu, core_hopping)

    ham = np.zeros(nz, nz, 4 * number_of_sites(nx), 4 * number_of_sites(nx))

    for ii in range(nz):
        ham[ii, ii] = h0 + delta_phi * cos( 2 * pi * ii / n_cdw + phi_cdw) * np.kron(np.identity(number_of_sites(nx)), gamma_3)

    for ii in range(nz - 1):
        ham[ii + 1, ii] = hz
        ham[ii, ii + 1] = hz.conj().T
    
    if pbc:
        ham[0, -1] = hz
        ham[-1, 0] = hz.conj().T


def calculate_disclination_rho(nkz: int, nx: int, model_params: dict, core_mu=None, core_hopping=False,
                               use_gpu=True, fname='ed_disclination_rho'):
    norb = 4

    kz_ax = np.linspace(0, 2 * pi, nkz + 1)[:-1]

    dk = kz_ax[1] - kz_ax[0]

    rho = np.zeros(number_of_sites(nx))

    for kz in tqdm(kz_ax):
        if use_gpu:
            import cupy as cp
            import cupy.linalg as clg

            h = cp.asarray(disclination_hamiltonian(kz, nx, model_params, core_mu=core_mu,
                                                    core_hopping=core_hopping))

            evals, evecs = clg.eigh(h)
            evals = evals.get()
            evecs = evecs.get()
        else:
            h = disclination_hamiltonian(kz, nx, model_params, core_mu=core_mu,
                                         core_hopping=core_hopping)

            evals, evecs = nlg.eigh(h)

        rho_kz = np.zeros(number_of_sites(nx))

        for ii, energy in enumerate(evals):
            if energy <= 0:
                wf = evecs[:, ii]
                temp_rho = np.reshape(np.multiply(np.conj(wf), wf), (-1, norb))
                rho_kz += np.sum(temp_rho, axis=-1).real

        rho += rho_kz * dk

    rho = rho / (2 * np.pi)

    results = rho
    params = (nkz, nx, model_params)
    data = (results, params)

    with open(data_dir / (fname + '.pickle'), 'wb') as handle:
        pkl.dump(data, handle)

    return rho


def calculate_disclination_ldos(energy_axis: np.ndarray, eta: float, nkz: int, nx: int, model_params: dict, core_mu=None, core_hopping=False, fname='ed_disclination_ldos'):
    import cupy as cp
    import cupy.linalg as clg

    norb = 4

    kz_ax = np.linspace(0, 2 * pi, nkz + 1)[:-1]

    dk = kz_ax[1] - kz_ax[0]

    ldos = cp.zeros((number_of_sites(nx), len(energy_axis)))

    for kz in tqdm(kz_ax):
        h = cp.asarray(disclination_hamiltonian(kz, nx, model_params, core_mu=core_mu,
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
        elif bulk_ind(nx, threshold, ii):
            q_bound += q

    return q_bound


def calculate_bound_ldos(nx: int, threshold: int, ldos, exclude_core=False):
    core_ind = disc_core_ind(nx)

    ldos_bound = np.zeros(ldos.shape[-1])

    for ii in range(ldos.shape[0]):
        if ii == core_ind and exclude_core:
            continue
        elif bulk_ind(nx, threshold, ii):
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
        calculate_disclination_rho(nkz, nx, m0, bxy, bz, g1, g2, c4_masses=c4_masses, core_mu=core_mu,
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
        calculate_disclination_rho(nkz, nx, m0, bxy, bz, g1, g2, c4_masses=c4_masses, core_mu=core_mu, use_gpu=use_gpu,
                                   fname=data_folder_name + f'/run_0_{ii}')
        calculate_disclination_rho(nkz, nx, m0, bxy, bz + dbz, g1, g2, c4_masses=c4_masses, core_mu=core_mu,
                                   use_gpu=use_gpu, fname=data_folder_name + f'/run_1_{ii}')

        print(f'\n\nFinished run {ii+1}/{len(bz_ax)}.\n')
        # print(f'{response_coef(m0, bz)=}')
        # print(f'{response_coef(m0, bz+dbz)=}')
