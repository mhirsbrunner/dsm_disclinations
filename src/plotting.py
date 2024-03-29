import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator

import networkx as netx
import disclinated_dsm as disc
import src.utils as utils

import numpy as np
import scipy as sp

from pathlib import Path
import pickle as pkl
from os import listdir
from os.path import isfile, join

from collections import Counter

# File structure
project_src = Path(__file__).parent
project_root = project_src.parent
styles_dir = project_root / 'matplotlib_styles'
data_dir = project_root / 'data'
figure_dir = project_root / 'figures'


def disclination_graph(nx: int):
    graph = netx.Graph()

    if nx % 2 == 0:
        site_mod = 0
    else:
        site_mod = 1

    for ii in range(nx):
        for jj in range(nx):
            # x-hoppings
            if jj < nx // 2 + site_mod:
                if ii < nx - 1:
                    graph.add_edge((ii, jj), (ii + 1, jj))
            else:
                if ii < nx // 2 - 1:
                    graph.add_edge((ii, jj), (ii + 1, jj))
            # y-hoppings
            if ii < nx // 2:
                if jj < nx - 1:
                    graph.add_edge((ii, jj), (ii, jj + 1))
            else:
                if jj < nx // 2 - 1 + site_mod:
                    graph.add_edge((ii, jj), (ii, jj + 1))

    for ii in range(nx // 2):
        graph.add_edge((nx // 2 + ii + site_mod, nx // 2 - 1 + site_mod), (nx // 2 - 1, nx // 2 + ii + site_mod))

    pos = netx.kamada_kawai_layout(graph)

    return graph, pos


def plot_disclination_rho(subtract_background=False, data_fname='ed_disclination_rho', save=True,
                          fig_fname='ed_disclination_rho', close_disc=True, threshold=None):
    results, params = utils.load_results(data_fname)
    nkz, nx, model_params = params
    rho = results

    print(f"Parameters: {nkz=}, {nx=}")
    print(f"Model Parameters: {model_params}")

    if threshold is not None:
        print(f'Bound Charge: {disc.bound_charge_density_in_k(nx, threshold, rho)}')
    try:
        print(f'Coefficient: {disc.response_coef(model_params["m0"], model_params["bz"])}')
    except ValueError:
        print('Coefficient: N/A (insulating phase, maybe topological!)')

    # Subtract background charge and calculate the total charge (mod 8)
    if subtract_background:
        data = rho - np.mean(rho)
    else:
        data = rho

    dmax = np.max(np.abs(data))
    normalized_data = data / dmax

    # Generate list of lattice sites and positions
    x = []
    y = []
    graph, pos = disclination_graph(nx)

    # Order the node list by the x index so the plot makes sense
    ordered_nodes = list(graph.nodes)
    ordered_nodes.sort(key=lambda s: s[1])

    for site in ordered_nodes:
        if close_disc:
            coords = pos[site]
        else:
            coords = site
        x.append(coords[0])
        y.append(coords[1])
    # Plot charge density
    fig, ax = plt.subplots(figsize=(6, 4))
    marker_scale = 250
    if subtract_background:
        im = ax.scatter(x, y, s=marker_scale * np.abs(normalized_data), c=data, cmap='bwr', marker='o', vmax=dmax,
                        vmin=-dmax)
    else:
        im = ax.scatter(x, y, s=marker_scale * np.abs(normalized_data), c=data, cmap='Reds', marker='o')

    ax.scatter(x, y, s=2, c='black')
    ax.set_aspect('equal')

    cbar = utils.add_colorbar(im, aspect=15, pad_fraction=1.0)
    cbar.ax.set_title(r'$|\rho|$', size=14)
    cbar.ax.tick_params(labelsize=14)

    ax.margins(x=0.2)

    plt.axis('off')
    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()


def plot_disclination_ldos(energy: float, data_fname='ed_disclination_ldos', save=True,
                          fig_fname='ed_disclination_ldos', close_disc=True, threshold=None):

    results, params = utils.load_results(data_fname)
    energy_axis, eta, nkz, nx, model_params = params
    ldos = results

    if energy < energy_axis[0] or energy > energy_axis[-1]:
        raise ValueError(f'Given energy out of bounds E=({energy_axis[0]:.2f},{energy_axis[-1]:.2f})')
    
    idx = (np.abs(energy_axis - energy)).argmin()

    data = ldos[:, idx]

    print(f"Parameters: energy={energy_axis[idx]:.2f}, {eta=}, {nkz=}, {nx=}")
    print(f"Model Parameters: {model_params}")

    try:
        print(f'Coefficient: {disc.response_coef(model_params["m0"], model_params["bz"])}')
    except ValueError:
        print('Coefficient: N/A (insulating phase, maybe topological!)')

    dmax = np.max(np.abs(data))
    normalized_data = data / dmax

    # Generate list of lattice sites and positions
    x = []
    y = []
    graph, pos = disclination_graph(nx)

    # Order the node list by the x index so the plot makes sense
    ordered_nodes = list(graph.nodes)
    ordered_nodes.sort(key=lambda s: s[1])

    for site in ordered_nodes:
        if close_disc:
            coords = pos[site]
        else:
            coords = site
        x.append(coords[0])
        y.append(coords[1])
    # Plot charge density
    fig, ax = plt.subplots(figsize=(6, 4))
    marker_scale = 250
    
    im = ax.scatter(x, y, s=marker_scale * np.abs(normalized_data), c=data, cmap='Reds', marker='o')

    ax.scatter(x, y, s=2, c='black')
    ax.set_aspect('equal')

    cbar = utils.add_colorbar(im, aspect=15, pad_fraction=1.0)
    cbar.ax.set_title('LDOS', size=14)
    cbar.ax.tick_params(labelsize=14)

    ax.margins(x=0.2)

    plt.axis('off')
    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + f'E_{energy:.2f}' + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + f'E_{energy:.2f}' + '.png'))

    plt.show()


def plot_disclination_rho_vs_r(data_fname='ed_disclination_rho', save=True,
                               fig_fname='ed_disclination_rho_vs_r'):
    results, params = utils.load_results(data_fname)
    nkz, nx, m0, bxy, bz, g1, g2, c4_masses = params
    rho = results

    # Subtract background charge and calculate the total charge (mod 8)
    data = rho - np.mean(rho)

    # Generate list of lattice sites and positions
    x = []
    y = []
    r = []

    core_coords = disc.ind_to_coord(nx, disc.disc_core_ind(nx))
    graph, pos = disclination_graph(nx)

    graph_radii = netx.single_source_shortest_path_length(graph, disc.ind_to_coord(nx, disc.disc_core_ind(nx)))

    # Order the node list by the x index so the plot makes sense
    ordered_nodes = list(graph.nodes)
    ordered_nodes.sort(key=lambda s: s[1])

    for site in ordered_nodes:
        coords = site
        x.append(coords[0])
        y.append(coords[1])

        radius = np.sqrt((core_coords[0] - coords[0]) ** 2 + (core_coords[1] - coords[1]) ** 2)
        r.append(radius)

    temp = zip(r, data)

    rho_dict = Counter()
    for key, value in temp:
        rho_dict[key] += value

    data_list = sorted(rho_dict.items())

    sorted_r = list(x for x, _ in data_list)
    sorted_rho = list(x for _, x in data_list)

    q_vs_r = np.cumsum(sorted_rho)

    charge_density = list(q / (np.pi * r ** 2) for q, r in zip(q_vs_r[1:], sorted_r[1:]))

    # Fitting
    def exp_decay(xx, a, b, c):
        yy = a + b * np.exp(-1 * c * xx)
        return yy

    def powerlaw_decay(xx, a, b, c):
        yy = a + b * xx + c * xx ** 2
        return yy

    params, cov = sp.optimize.curve_fit(exp_decay, sorted_r[1:], charge_density)
    fit_y = exp_decay(np.asarray(sorted_r[1:]), params[0], params[1], params[2])

    # Plot charge density vs r
    plt.style.use(styles_dir / 'line_plot.mplstyle')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(y=0.0, color='k', linestyle='-')
    ax.plot(sorted_r[1:], charge_density, 'ro-')
    ax.plot(sorted_r[1:], fit_y, 'b--')

    ax.set_xlabel(r'$r$ (a.u.)')
    plt.xticks(list(plt.xticks()[0]) + [0, ])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_ylabel(r'$\rho(r)$')

    # ax.margins(x=0.05)
    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()


def plot_q_vs_coef(threshold=1, slope_guess=None, exclude_core=False, data_folder_name=None, save=True,
                   fig_fname="q_vs_coef", ylim=None):
    def read_data(t=1, dfn=None, xc=False):
        if dfn is not None:
            data_location = data_dir / dfn
        else:
            data_location = data_dir

        filenames = [f for f in listdir(data_location) if isfile(join(data_location, f))]

        m0s = []
        bzs = []
        x = []
        y = []

        for fname in filenames:
            with open(data_location / fname, 'rb') as handle:
                rho, params = pkl.load(handle)

            nkz, nx, m0, bxy, bz, g1, g2, c4_mass = params

            m0s.append(m0)
            bzs.append(bz)
            x.append(disc.response_coef(m0, bz))
            y.append(disc.bound_charge_density_in_k(nx, t, rho, exclude_core=xc))

        return x, y

    coefs, charges = read_data(threshold, data_folder_name, xc=exclude_core)
    coefs = np.array(coefs)
    charges = np.array(charges)

    plt.style.use(styles_dir / 'line_plot.mplstyle')
    fig, ax = plt.subplots(figsize=(6, 4))

    if slope_guess is not None:
        ax.plot([0, 1], [0, slope_guess], 'k--')
    ax.plot(coefs, charges, 'ro')

    ax.set_xlabel(r'$v$')
    ax.set_xticks([0, 1])

    ax.set_ylabel(r'$Q_0$')

    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()


def plot_dq_dnu(threshold=1, slope_guess=None, exclude_core=False, data_folder_name=None, save=True,
                fig_fname="dq_dnu_vs_coef", ylim=None):
    def read_data(t=1, dfn=None, xc=False):
        if dfn is not None:
            data_location = data_dir / dfn
        else:
            data_location = data_dir

        filenames = [f for f in listdir(data_location) if isfile(join(data_location, f))]

        m0s = []
        bzs = []
        coefs = []
        rhos = []

        for fname in filenames:
            with open(data_location / fname, 'rb') as handle:
                rho, params = pkl.load(handle)

            nkz, nx, m0, bxy, bz, g1, g2, c4_mass = params

            m0s.append(m0)
            bzs.append(bz)
            coefs.append(disc.response_coef(m0, bz))
            rhos.append(rho)

        temp = sorted(zip(coefs, rhos))
        sorted_coefs = np.asarray(list(x for x, _ in temp))
        sorted_rhos = np.asarray(list(x for _, x in temp))

        d_nu = np.diff(np.reshape(sorted_coefs, (-1, 2)), axis=1).squeeze()
        d_rho = np.diff(np.reshape(sorted_rhos, (sorted_rhos.shape[0] // 2, 2, -1)), axis=1).squeeze()

        x = list(x[-1] for x in np.reshape(sorted_coefs, (-1, 2)))
        y = list(disc.bound_charge_density_in_k(nx, t, d_rho[ii], subtract_avg=True, exclude_core=xc) / d_nu[ii]
                 for ii in range(len(d_rho)))

        return x, y

    nu, dqdnu = read_data(threshold, data_folder_name, xc=exclude_core)

    plt.style.use(styles_dir / 'line_plot.mplstyle')
    fig, ax = plt.subplots(figsize=(6, 4))

    if slope_guess is not None:
        ax.plot([0, 1], [0, slope_guess], 'k--')

    ax.plot(nu, dqdnu, 'ro')

    ax.set_xlabel(r'$v$')
    ax.set_xticks([0, 1])

    ax.set_ylabel(r'$dQ/d\nu$')

    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()
