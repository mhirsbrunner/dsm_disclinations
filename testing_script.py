import src.disclination as disc
import src.plotting as oplot
import src.bloch_hamiltonian as blc
import numpy as np
import matplotlib.pyplot as plt

# %%
nkz = 100
n_x = 19
m0 = 1.0
bxy = 1.0
bz = 0.6
g1 = 0.3
g2 = 0.3
c4_masses = (0.0, 0.0)
core_mu = 1.0
core_hopping = False

print(disc.response_coef(m0, bz))
# %%
blc.plot_band_structure(2 * np.pi / nkz, m0, bxy, bz, g1, g2, c4_masses=c4_masses)

# %%
rho = disc.calculate_disc_rho(nkz, n_x, m0, bxy, bz, g1, g2, c4_masses=c4_masses, core_mu=core_mu,
                                      core_hopping=core_hopping)

# %%
oplot.plot_disclination_rho_vs_r()
oplot.plot_disclination_rho(subtract_background=True, threshold=2)

# %%
coef_min = 0.05
coef_max = 0.95
bz_pts = 5
data_folder_name = 'test_yes_core_mu'
disc.calculate_bound_charge_vs_nu(nkz, n_x, m0, bxy, g1, g2, coef_min, coef_max, bz_pts, c4_masses=c4_masses,
                                  core_mu=core_mu, core_hopping=core_hopping, data_folder_name=data_folder_name)

# %%
oplot.plot_q_vs_coef(threshold=1, data_folder_name=data_folder_name, slope_guess=-1/2)

# %% Plot spectrum vs kz for disclinated lattice
kz_pts = 50
kz_ax = np.linspace(-np.pi, np.pi, kz_pts + 1)[:-1]

u = []
v = []

for k_z in kz_ax:
    print(f'{k_z=:.2g}', end='\r')
    ham = disc.disc_bloch_hamiltonian(k_z, n_x, m0, bxy, bz, g1, g2, c4_masses=c4_masses, core_mu=core_mu,
                                        core_hopping=core_hopping)
    temp_u, temp_v = np.linalg.eigh(ham)

    u.append(temp_u)
    v.append(temp_v)

# %%
fig, ax = plt.subplots()
ax.plot(u, 'b.')
# ax.plot(kz_ax, u, 'b.')
# ax.set_xticks((-np.pi, 0, np.pi))
# ax.set_xticklabels((r'$-\pi$', 0, r'$\pi$'))
ax.set_ylim(-0.5, 0.5)
plt.show()
