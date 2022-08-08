import numpy as np
import numpy.linalg as nlg
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from pathlib import Path
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


def bloch_hamiltonian(k, m0: float, bxy: float, bz: float, g1: float, g2: float, c4_masses=None) -> np.ndarray:
    kx, ky, kz = k

    h = np.zeros((4, 4), dtype=complex)

    h += sin(kx) * gamma_1
    h += sin(ky) * gamma_2
    h += (m0 - bxy * (2 - cos(kx) - cos(ky)) - bz * (1 - cos(kz))) * gamma_3
    h += g1 * (cos(kx) - cos(ky)) * sin(kz) * gamma_4
    h += g2 * sin(kx) * sin(ky) * sin(kz) * gamma_5

    if c4_masses is not None:
        h += sin(kz) * (c4_masses[0] * gamma_4 + c4_masses[1] * gamma_5)

    return h


def high_symmetry_lines(dk: float):
    gamma = (0, 0, 0)
    x = (pi, 0, 0)
    l = (pi, 0, pi)
    z = (0, 0, pi)
    m = (pi, pi, 0)
    r = (pi, pi, pi)

    hsps = (gamma, r, l, z, gamma, x, m, r)

    k_nodes = [0]

    k0 = hsps[0]
    k1 = hsps[1]

    dist = np.sqrt((k1[0] - k0[0]) ** 2 + (k1[1] - k0[1]) ** 2 + (k1[2] - k0[2]) ** 2)
    nk = int(dist // dk)
    kx = np.linspace(k0[0], k1[0], nk)
    ky = np.linspace(k0[1], k1[1], nk)
    kz = np.linspace(k0[2], k1[2], nk)

    k_nodes.append(len(kx) - 1)

    for ii, k in enumerate(hsps[2:]):
        k0 = k1
        k1 = k

        dist = np.sqrt((k1[0] - k0[0]) ** 2 + (k1[1] - k0[1]) ** 2 + (k1[2] - k0[2]) ** 2)
        nk = int(dist // dk)
        kx = np.concatenate((kx, np.linspace(k0[0], k1[0], nk + 1)[1:]))
        ky = np.concatenate((ky, np.linspace(k0[1], k1[1], nk + 1)[1:]))
        kz = np.concatenate((kz, np.linspace(k0[2], k1[2], nk + 1)[1:]))

        k_nodes.append(len(kx) - 1)

    ks = np.stack((kx, ky, kz), axis=1)

    return ks, k_nodes


def plot_band_structure(dk, m0: float, bxy: float, bz: float, g1: float, g2: float, c4_masses=None, save=True,
                        fig_fname='bands'):

    def ham(k_vec) -> np.ndarray:
        return bloch_hamiltonian(k_vec, m0, bxy, bz, g1, g2, c4_masses=c4_masses)

    ks, k_nodes = high_symmetry_lines(dk)

    evals = np.zeros((len(ks), 4))

    for ii, k in enumerate(ks):
        evals[ii] = np.linalg.eigvalsh(ham(k))

    labels = (r'$\Gamma$', r'$R$', r'$L$', r'$Z$', r'$\Gamma$', r'$X$', r'$M$', r'$R$')

    plt.style.use(styles_dir / 'bands.mplstyle')

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(np.zeros(evals.shape[0]), 'k--')
    ax.plot(evals, 'b-')

    ax.set_xticks(k_nodes)
    ax.set_xticklabels(labels)

    for k in k_nodes[1:-1]:
        ax.axvline(k, 0, 1, color='black', linewidth=2, linestyle='-')

    ax.set_ylabel('Energy')

    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()

    return
