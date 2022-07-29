import numpy as np
import numpy.linalg as nlg
from numpy import pi

# import cupy as cp
# import cupy.linalg as clg

import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1

from pathlib import Path
import pickle as pkl

# File structure
project_src = Path(__file__).parent
project_root = project_src.parent
styles_dir = project_root / 'matplotlib_styles'
data_dir = project_root / 'data'
figure_dir = project_root / 'figures'


##############
# Plot Utils #
##############
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


#############################
# High Symmetry Lines Utils #
#############################
def high_symmetry_lines(dk: float):
    gamma = (0, 0)
    x = (pi, 0)
    y = (0, pi)
    m = (pi, pi)

    hsps = (y, gamma, x, m)

    k_nodes = [0]

    k0 = hsps[0]
    k1 = hsps[1]

    dist = np.sqrt((k1[0] - k0[0]) ** 2 + (k1[1] - k0[1]) ** 2)
    nk = int(dist // dk)
    kx = np.linspace(k0[0], k1[0], nk)
    ky = np.linspace(k0[1], k1[1], nk)

    k_nodes.append(len(kx) - 1)

    for ii, k in enumerate(hsps[2:]):
        k0 = k1
        k1 = k

        dist = np.sqrt((k1[0] - k0[0]) ** 2 + (k1[1] - k0[1]) ** 2)
        nk = int(dist // dk)
        kx = np.concatenate((kx, np.linspace(k0[0], k1[0], nk + 1)[1:]))
        ky = np.concatenate((ky, np.linspace(k0[1], k1[1], nk + 1)[1:]))

        k_nodes.append(len(kx) - 1)

    ks = np.stack((kx, ky), axis=1)

    return ks, k_nodes


########################
# Green Function Utils #
########################
def retarded_green_function(hamiltonian: np.ndarray, energy: float, eta=1e-6) -> np.ndarray:
    n = hamiltonian.shape[0]
    return nlg.inv((energy + 1j * eta) * np.identity(n) - hamiltonian)


def surface_green_function(energy, h00, h01, surf_pert=None, return_bulk=False):
    it_max = 20
    tol = 1e-12

    if surf_pert is None:
        surf_pert = np.zeros(h00.shape)

    energy = energy * np.identity(h00.shape[0])

    eps_s = h00

    eps = h00
    alpha = h01.conj().T
    beta = h01

    it = 0
    alpha_norm = 1
    beta_norm = 1

    while alpha_norm > tol or beta_norm > tol:
        g0_alpha = nlg.solve(energy - eps, alpha)
        g0_beta = nlg.solve(energy - eps, beta)

        eps_s = eps_s + alpha @ g0_beta
        eps = eps + alpha @ g0_beta + beta @ g0_alpha

        alpha = alpha @ g0_alpha
        beta = beta @ g0_beta

        alpha_norm = nlg.norm(alpha)
        beta_norm = nlg.norm(beta)

        it += 1

        if it > it_max:
            print(f'Max iterations reached. alpha_norm: {alpha_norm}, beta_norm: {beta_norm}')
            break

    gs = nlg.inv(energy - eps_s - surf_pert)

    if return_bulk:
        gb = nlg.inv(energy - eps)
        return gs, gb
    else:
        return gs


# def cp_surface_green_function(energy, h00, h01, surf_pert=None, return_bulk=False):
#     it_max = 20
#     tol = 1e-12
#
#     if surf_pert is None:
#         cp_surf_pert = cp.zeros(h00.shape)
#     else:
#         cp_surf_pert = cp.asarray(surf_pert)
#
#     energy_mat = energy * cp.identity(h00.shape[0])
#
#     eps_s = cp.asarray(h00)
#
#     eps = cp.copy(eps_s)
#
#     beta = cp.asarray(h01)
#     alpha = cp.conj(cp.transpose(beta))
#
#     it = 0
#     alpha_norm = 1
#     beta_norm = 1
#
#     while alpha_norm > tol or beta_norm > tol:
#         g0_alpha = clg.solve(energy_mat - eps, alpha)
#         g0_beta = clg.solve(energy_mat - eps, beta)
#
#         temp = cp.dot(alpha, g0_beta)
#         eps_s = eps_s + temp
#         eps = eps + temp + cp.dot(beta, g0_alpha)
#
#         alpha = cp.dot(alpha, g0_alpha)
#         beta = cp.dot(beta, g0_beta)
#
#         alpha_norm = clg.norm(alpha)
#         beta_norm = clg.norm(beta)
#
#         it += 1
#
#         if it > it_max:
#             print(f'Max iterations reached. alpha_norm: {alpha_norm}, beta_norm: {beta_norm}')
#             break
#
#     gs = cp.asnumpy(clg.inv(energy_mat - eps_s - cp_surf_pert))
#
#     if return_bulk:
#         gb = cp.asnumpy(clg.inv(energy_mat - eps))
#         return gs, gb
#     else:
#         return gs


def spectral_function(g=None, ham=None, energy=None, eta=None) -> np.ndarray:
    if g is None:
        if ham is not None:
            if energy is None or eta is None:
                raise ValueError('Hamiltonian, energy, and broadening must be passed'
                                 ' if the Green function is no specified.')
            else:
                g = retarded_green_function(ham, energy, eta=eta)
        else:
            raise ValueError('Either Green function or Hamiltonian must be given.')
    elif ham is not None:
        print('Both Green function and Hamiltonian specified, defaulting to using the Green function.')

    return -2 * np.imag(g)


# Data Utils
def load_results(data_fname):
    with open(data_dir / (data_fname + '.pickle'), 'rb') as handle:
        return pkl.load(handle)
