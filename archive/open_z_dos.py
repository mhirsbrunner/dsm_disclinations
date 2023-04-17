import numpy as np
from numpy import sin, cos, pi

import matplotlib.pyplot as plt

from pathlib import Path
import pickle as pkl

from tqdm import tqdm

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

gamma_0 = np.kron(sigma_z, sigma_x)
gamma_x = np.kron(sigma_x, sigma_0)
gamma_y = np.kron(sigma_y, sigma_0)
gamma_z = np.kron(sigma_z, sigma_z)
gamma_5 = np.kron(sigma_z, sigma_y)


# Functions for calculating DOS with 2 periodic and 1 open directions
def h_eq11_open_z(nz: int, kx: float, ky: float, mass: float, phs_mass: float) -> np.ndarray:
    sin_term = sin(kx) * gamma_x + sin(ky) * gamma_y
    mass_term = (mass + cos(kx) + cos(ky)) * gamma_0

    h_onsite = np.kron(sin_term, sigma_0) + np.kron(mass_term, sigma_z)
    h_surf = phs_mass * np.kron(gamma_5, sigma_0)

    h_hopping = -1j / 2 * np.kron(gamma_z, sigma_0) + 1 / 2 * np.kron(gamma_0, sigma_z)

    h = np.kron(np.identity(nz), h_onsite)
    h += np.kron(np.diag(np.concatenate((np.ones(1), np.zeros(nz - 2), np.ones(1)))), h_surf)
    h += np.kron(np.diag(np.ones(nz - 1), k=1), h_hopping)
    h += np.kron(np.diag(np.ones(nz - 1), k=-1), h_hopping.T.conj())

    return h


def calculate_dos_open_z(nz: int, dk: float, mass: float, phs_mass: float, energy_axis, eta=1e-6, fname='dos_open_z'):
    ks, k_nodes = utils.high_symmetry_lines(dk)

    dos = np.zeros((len(energy_axis), len(ks)))

    for ii, k in enumerate(tqdm(ks)):
        kx, ky = k
        h = h_eq11_open_z(nz, kx, ky, mass, phs_mass)
        for jj, energy in enumerate(energy_axis):
            a = utils.spectral_function(ham=h, energy=energy, eta=eta)
            dos[jj, ii] = np.sum(np.diag(a)) / (2 * pi)

    results = dos
    params = (nz, mass, energy_axis, eta, ks, k_nodes)
    data = (results, params)

    with open(data_dir / (fname + '.pickle'), 'wb') as handle:
        pkl.dump(data, handle)


# Functions for calculating the surface DOS of a system with translation invariance in the transverse directions
def plot_dos_open_z(data_fname='dos_open_z', save=True, fig_fname='dos_open_z', vmax=100):
    with open(data_dir / (data_fname + '.pickle'), 'rb') as handle:
        results, params = pkl.load(handle)

    dos = results
    nz, mass, energy_axis, eta, ks, k_nodes = params

    labels = (r'$Y$', r'$\Gamma$', r'$X$', r'$M$')

    plt.style.use(styles_dir / 'bands.mplstyle')

    fig, ax = plt.subplots(figsize=(6, 4))

    im = ax.imshow(dos, origin='lower', cmap="magma", aspect='auto', vmin=0, vmax=vmax)
    # utils.add_colorbar(im, aspect=15, pad_fraction=1.0)

    ax.set_xticks(k_nodes)
    ax.set_xticklabels(labels)

    for k in k_nodes[1:-1]:
        ax.vlines(k, ymin=0, ymax=len(energy_axis) - 1, color='white', linewidth=2, linestyles='--')

    ax.set_ylabel('Energy)')

    ax.set_yticks((-1, len(energy_axis)))
    ax.set_yticklabels((energy_axis[0], energy_axis[-1]))

    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()


def main():
    mass = -2
    phs_mass = 0.0

    nz = 16
    dk = 0.05

    eta = 0.05

    e_min = -5
    e_max = 5
    e_pts = 500

    energy_axis = np.linspace(e_min, e_max, e_pts)

    calculate_dos_open_z(nz, dk, mass, phs_mass, energy_axis, eta)
    plot_dos_open_z(fig_fname='publishing/open_z_dos_phs')


if __name__ == '__main__':
    main()
