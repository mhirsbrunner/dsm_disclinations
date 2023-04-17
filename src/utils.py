import numpy as np
import numpy.linalg as nlg
from numpy import pi

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
def high_symmetry_bz_path_2d(tot_nk: int):
    """
    Produces a list of crystal momentum points along a path traversing high-symmetry lines of the cubic BZ

    Parameters
    ----------
    tot_nk : int
        The desired number of points on the momentum path

    Returns
    -------
    path: List
        A list of approximately tot_nk crystal momentum points on the high-symmetry path
    nodes: List
        A list of the indices of high-symmetry points on the path
    labels: List
        A list of strings labeling the high-symmetry points enumerated by 'nodes'   

    """
    nk = tot_nk // (2 + np.sqrt(2))

    gamma = (0, 0)
    x = (pi, 0)
    m = (pi, pi)

    point_list = [gamma, x, m, gamma]
    labels = [r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$']
    norm_list  = [1, 1, np.sqrt(2)]
    k_nodes = [int(np.floor(nk * norm)) for norm in norm_list]

    path = []
    for ii, norm in enumerate(norm_list):
        ki = point_list[ii]
        kf = point_list[ii + 1]

        [kx, ky] = [list(np.linspace(ki[jj], kf[jj], k_nodes[ii] + 1))[:-1] for jj in range(2)]

        path = path + list(zip(kx, ky))
    
    path = path + [point_list[-1]]
    
    return path, np.cumsum([0, ] + k_nodes), labels


def high_symmetry_bz_path_3d(tot_nk: int):
    """
    Produces a list of crystal momentum points along a path traversing high-symmetry lines of the cubic BZ

    Parameters
    ----------
    tot_nk : int
        The desired number of points on the momentum path

    Returns
    -------
    path: List
        A list of approximately tot_nk crystal momentum points on the high-symmetry path
    nodes: List
        A list of the indices of high-symmetry points on the path
    labels: List
        A list of strings labeling the high-symmetry points enumerated by 'nodes'   

    """
    nk = tot_nk // (8 + 3 * np.sqrt(2))

    gamma = (0, 0, 0)
    x = (pi, 0, 0)
    z = (0, 0, pi)
    m = (pi, pi, 0)
    r = (pi, 0, pi)
    a = (pi, pi, pi)

    point_list = [gamma, x, m, gamma, z, r, a, z, a, m, x, r]
    labels = [r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$', r'$Z$', r'$R$', r'$A$', r'$Z$', r'$A$', r'$M$', r'$X$', r'$R$']
    norm_list  = [1, 1, np.sqrt(2), 1, 1, 1, np.sqrt(2), np.sqrt(2), 1, 1, 1]
    k_nodes = [int(np.floor(nk * norm)) for norm in norm_list]

    path = []
    for ii, norm in enumerate(norm_list):
        ki = point_list[ii]
        kf = point_list[ii + 1]

        [kx, ky, kz] = [list(np.linspace(ki[jj], kf[jj], k_nodes[ii] + 1))[:-1] for jj in range(3)]

        path = path + list(zip(kx, ky, kz))
    
    return path, np.cumsum([0, ] + k_nodes), labels


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
