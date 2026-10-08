### Test-set performance (chemical-system-grouped 80/10/10 split)

| target | unit | N_test | GNN MAE | RF MAE | GNN R2 | RF R2 |
|---|---|---|---|---|---|---|
| optb88vdw_bandgap | eV | 8756 | 0.211 | 0.356 | 0.803 | 0.652 |
| mbj_bandgap | eV | 1928 | 0.368 | 0.691 | 0.837 | 0.574 |
| formation_energy_peratom | eV/atom | 8756 | 0.098 | 0.183 | 0.971 | 0.896 |
| avg_elec_mass | log10(m*/m_e) | 317 | 0.321 | 0.377 | 0.439 | 0.342 |
| avg_hole_mass | log10(m*/m_e) | 317 | 0.322 | 0.415 | 0.503 | 0.375 |
| epsilon | log10(eps) | 5661 | 0.216 | 0.250 | 0.690 | 0.660 |

Masses and dielectric constant are evaluated on a log10 scale.

### Polymorph test

Test materials whose formula has >=2 structures with gaps differing by >0.3 eV (n = 838): GNN MAE 0.592 eV vs RF MAE 0.901 eV.