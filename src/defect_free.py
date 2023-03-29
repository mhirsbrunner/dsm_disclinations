import numpy as np
import numpy.linalg as nlg
from numpy import sin, cos, pi
from scipy import linalg as slg

from pathlib import Path
import pickle as pkl

import src.utils as utils

from tqdm import tqdm


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

gamma_0 = np.kron(sigma_z, sigma_x)
gamma_x = np.kron(sigma_x, sigma_0)
gamma_y = np.kron(sigma_y, sigma_0)
gamma_z = np.kron(sigma_z, sigma_z)
gamma_5 = np.kron(sigma_z, sigma_y)


def open_z_hamiltonian(kx: float, ky: float, nz: int, mass: float, phs_mass: float, mirror_sym=False):
    norb = 8

    h_0 = np.zeros((norb, norb), dtype=complex)
    h_0_phs = np.zeros_like(h_0)
    h_z = np.zeros_like(h_0)

    h_0 += (mass + cos(kx) + cos(ky)) * np.kron(gamma_0, sigma_z)
    h_0 += np.kron(sin(kx) * gamma_x + sin(ky) * gamma_y, sigma_0)

    h_0_phs += phs_mass * np.kron(gamma_5, sigma_0)

    h_z += 1 / 2 * np.kron(gamma_0, sigma_z) + 1j / 2 * np.kron(gamma_z, sigma_0)

    h_z += 1j / 2 * sin(kx) * np.kron(gamma_0, sigma_x) + 1j / 2 * sin(ky) * np.kron(gamma_0, sigma_y)

    h = np.kron(np.identity(nz), np.array(h_0))

    if mirror_sym:
        mirror_factor = -1
    else:
        mirror_factor = 1
    h[:norb, :norb] += h_0_phs
    h[-norb:, -norb:] += mirror_factor * h_0_phs
    h += np.kron(np.diag(np.ones(nz - 1), k=1), h_z)
    h += np.kron(np.diag(np.ones(nz - 1), k=-1), h_z.conj().T)

    return h


def calculate_open_z_bands(nz: int, dk: float, mass: float, phs_mass: float, fname='open_z_bands'):
    norb = 8
    ks, k_nodes = utils.high_symmetry_lines(dk)

    evals = np.zeros((len(ks), nz * norb))

    for ii, k in enumerate(tqdm(ks)):
        kx, ky = k
        h = open_z_hamiltonian(kx, ky, nz, mass, phs_mass)
        evals[ii] = np.linalg.eigvalsh(h)

    results = evals
    params = (nz, mass, phs_mass, ks, k_nodes)
    data = (results, params)

    with open(data_dir / (fname + '.pickle'), 'wb') as handle:
        pkl.dump(data, handle)


def calculate_open_z_dos(nz: int, dk: float, mass: float, phs_mass: float, energy_axis, eta=0.05,
                         fname='open_z_dos'):
    ks, k_nodes = utils.high_symmetry_lines(dk)

    dos = np.zeros((len(energy_axis), len(ks)))

    for ii, k in enumerate(tqdm(ks)):
        kx, ky = k
        h = open_z_hamiltonian(kx, ky, nz, mass, phs_mass)
        for jj, energy in enumerate(energy_axis):
            a = utils.spectral_function(ham=h, energy=energy, eta=eta)
            dos[jj, ii] = np.sum(np.diag(a)) / (2 * pi)

    results = dos
    params = (nz, mass, phs_mass, energy_axis, eta, ks, k_nodes, True)
    data = (results, params)

    with open(data_dir / (fname + '.pickle'), 'wb') as handle:
        pkl.dump(data, handle)
