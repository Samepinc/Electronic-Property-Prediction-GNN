### Screening

Criteria: 1.0 <= gap <= 2.0 eV, formation energy <= 0.0 eV/atom (m* cap applied to discovery pool only).

**Validation on held-out test set** (n = 8756): precision 0.43, recall 0.40, base rate 0.06, enrichment over random 6.8x.
Composition-only RF on the same task: precision 0.31, recall 0.47.

**Discovery pool**: 1700 unrelaxed isovalent-substituted structures screened; 70 pass the criteria; top 10 by predicted formation energy:

| formula | parent | substitution | pred_gap_eV | pred_Ef_eV_per_atom | pred_m_e | pred_m_h | pred_dielectric | anion |
|---|---|---|---|---|---|---|---|---|
| Ba2HgF6 | JVASP-8137 | Zn->Hg | 1.201 | -3.212 | 0.255 | 0.519 | 4.028 | F |
| SrPtF4 | JVASP-116976 | Ni->Pt | 1.698 | -2.827 | 0.205 | 0.267 | 3.973 | F |
| Ba2SeS | JVASP-172572 | Sr->Ba | 1.939 | -2.182 | 0.510 | 0.742 | 5.251 | S |
| Ca3PBr3 | JVASP-3783 | As->P | 1.710 | -2.008 | 0.699 | 0.973 | 4.648 | Br |
| K6MnCl8 | JVASP-12725 | Na->K | 1.930 | -1.863 | 0.548 | 0.540 | 2.751 | Cl |
| Sr2AsCl | JVASP-118125 | P->As | 1.024 | -1.687 | 0.513 | 0.976 | 7.065 | Cl |
| SrCaTe2 | JVASP-138046 | Ba->Ca | 1.344 | -1.556 | 0.429 | 0.877 | 5.822 | Te |
| Na2YInBr6 | JVASP-110719 | Rb->Na | 1.362 | -1.384 | 0.451 | 0.747 | 3.585 | Br |
| K2YInI6 | JVASP-110380 | Rb->K | 1.683 | -1.307 | 0.565 | 0.857 | 4.943 | I |
| Rb2ScCuI6 | JVASP-100474 | Y->Sc | 1.753 | -1.236 | 0.692 | 0.939 | 5.403 | I |
| KRbTe | JVASP-175828 | S->Te | 1.039 | -0.934 | 0.252 | 0.925 | 4.016 | Te |
| Zn2TeS | JVASP-178482 | Se->Te | 1.351 | -0.621 | 0.159 | 0.782 | 7.794 | S |

Candidates are model predictions on unrelaxed geometries and must be validated with DFT (relaxation, band structure, phonons/E_hull) before any claim.