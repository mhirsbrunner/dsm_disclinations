import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.ticker as plticker

import networkx as netx
import src.disclination as disc
import src.utils as utils

import numpy as np
from numpy import sin, cos, pi

from pathlib import Path
import pickle as pkl
from os import listdir
from os.path import isfile, join

import warnings

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


def plot_disclination_rho(data_fname='ed_disclination_ldos', save=True,
                          fig_fname='ed_disclination_rho', close_disc=True):
    results, params = utils.load_results(data_fname)
    nkz, nx, m0, bxy, bz, g1, g2, c4_mass = params

    print(f"Parameters: {nkz=}, {nx=}, {m0=}, {bxy=}, {bz=}, {g1=}, {g2=}, {c4_mass=}")
    print(f'Coefficient: {disc.response_coef(m0, bz)}')

    rho = results

    # Subtract background charge and calculate the total charge (mod 8)
    data = rho - np.mean(rho)
    dmax = np.max(np.abs(data))
    normalized_data = data / np.max(np.abs(data))
    alpha_data = np.abs(normalized_data)

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

    # Make colormap
    cmap = plt.cm.bwr
    my_cmap = cmap(np.arange(cmap.N // 2, cmap.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N // 2)
    my_cmap = ListedColormap(my_cmap)

    # Plot charge density
    fig, ax = plt.subplots(figsize=(6, 4))
    marker_scale = 250
    # im = ax.scatter(x, y, s=marker_scale * np.abs(normalized_data), c=np.abs(data), cmap=my_cmap, marker='o', vmin=0)
    im = ax.scatter(x, y, s=marker_scale * np.abs(normalized_data), c=data, cmap='bwr', marker='o', vmax=dmax,
                    vmin=-dmax)
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


def read_data(threshold=1, data_folder_name=None):
    if data_folder_name is not None:
        data_location = data_dir / data_folder_name
    else:
        data_location = data_dir

    filenames = [f for f in listdir(data_location) if isfile(join(data_location, f))]

    m0s = []
    bzs = []
    coefs = []
    charges = []

    for fname in filenames:
        with open(data_location / fname, 'rb') as handle:
            rho, params = pkl.load(handle)

        nkz, nx, m0, bxy, bz, g1, g2, c4_mass = params

        m0s.append(m0)
        bzs.append(bz)
        coefs.append(disc.response_coef(m0, bz))
        charges.append(disc.calculate_bound_charge(nx, threshold, rho))

    print(f"Parameters: {nkz=}, {nx=}, {m0=}, {bxy=}, {g1=}, {g2=}, {c4_mass=}")

    return coefs, charges


def plot_q_vs_coef(threshold=1, slope_guess=1, data_folder_name=None, save=True, fig_fname="q_vs_coef", ylim=None):
    coefs, charges = read_data(threshold, data_folder_name)
    coefs = np.array(coefs)
    charges = np.array(charges)

    plt.style.use(styles_dir / 'line_plot.mplstyle')
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot([0, 1], [0, slope_guess], 'k--')
    ax.plot(coefs, np.abs(charges), 'ro')

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
