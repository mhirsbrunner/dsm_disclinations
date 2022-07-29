import src.plotting as plt

# %% Trivial q vs z
data_fname = 'nx_16_nz_16_mass_-4.0_half_False_other_False_nnn_False'
fig_fname = 'publishing/full_model_q_vs_z_trivial'
plt.plot_charge_per_layer(data_fname=data_fname, save=True, ylim=0.15, fig_fname=fig_fname)

# %% Topological q vs z
data_fname = 'nx_16_nz_16_mass_-2.0_half_False_other_False_nnn_True'
fig_fname = 'publishing/full_model_q_vs_z_topological'
plt.plot_charge_per_layer(data_fname=data_fname, save=True, ylim=0.15, fig_fname=fig_fname)

# %% Topological Charge Density
data_fname = 'nx_16_nz_16_mass_2.0_half_False_other_False_nnn_False'
fig_fname = 'publishing/full_model_charge_density_topological'
plt.plot_disclination_rho('bottom', data_fname=data_fname, save=True, fig_fname=fig_fname)

# %% Half model charge vs mass
data_folder = 'cluster_data'
fig_fname = 'publishing/half_model_charge_vs_mass'
plt.plot_q_vs_coef(data_folder, True, False, False, mod=True, save=True, fig_fname=fig_fname)

# %% Other half model charge vs mass
data_folder = 'cluster_data'
fig_fname = 'publishing/other_half_model_charge_vs_mass'
plt.plot_q_vs_coef(data_folder, True, True, False, mod=True, save=True, fig_fname=fig_fname)

# %% Full model charge vs mass
data_folder = 'cluster_data'
fig_fname = 'publishing/full_model_charge_vs_mass'
plt.plot_q_vs_coef(data_folder, False, False, False, mod=True, save=True, fig_fname=fig_fname)

# %% Full nnn model charge vs mass
data_folder = 'cluster_data'
fig_fname = 'publishing/full_nnn_model_charge_vs_mass_no_mod'
plt.plot_q_vs_coef(data_folder, False, False, True, mod=False, save=True, fig_fname=fig_fname)

# %% Bands
fig_fname = 'publishing/band_structure'
plt.plot_band_structure(0.05, 2.0, 0.0, nnn=True, fig_fname=fig_fname)

# %% PHS DOS
data_fname = 'open_z_dos_phs'
fig_fname = 'publishing/open_z_dos_phs'
plt.plot_open_z_dos(data_fname, fig_fname, save=True, vmax=40)

# %% No PHS DOS
data_fname = 'open_z_dos_no_phs'
fig_fname = 'publishing/open_z_dos_no_phs'
plt.plot_open_z_dos(data_fname, fig_fname, save=True, vmax=40)
