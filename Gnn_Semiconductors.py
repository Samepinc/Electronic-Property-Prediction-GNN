#!/usr/bin/env python3
"""GNN-guided discovery of semiconductor materials from crystal structures (NIST-JARVIS dft_3d).

Stages (run in order):
    python gnn_semiconductors.py audit      # count usable records per target + overlaps
    python gnn_semiconductors.py train      # build graphs (cached) + train multi-task GNN
    python gnn_semiconductors.py evaluate   # GNN vs composition-only RF, plots, interpretability
    python gnn_semiconductors.py screen     # validate screening on held-out set + screen new candidates

Outputs go to ./cache (graphs, model) and ./results (tables, figures, candidates.csv).
README.md results block is refreshed automatically after audit / evaluate / screen.
"""
import argparse, glob, json, os, re, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from pymatgen.core import Structure, Lattice, Element
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import CGConv, global_mean_pool

CACHE, RES = Path("cache"), Path("results")
TARGETS = ["optb88vdw_bandgap", "mbj_bandgap", "formation_energy_peratom",
           "avg_elec_mass", "avg_hole_mass", "epsilon"]
UNITS = ["eV", "eV", "eV/atom", "log10(m*/m_e)", "log10(m*/m_e)", "log10(eps)"]
LOG_IDX = [3, 4, 5]                      # targets stored on a log10 scale
R_CUT, MAX_NBR, N_GAUSS = 8.0, 12, 41    # graph construction


# ----------------------------------------------------------------------------- elements
def _elem_table():
    """Per-element features [Z, electronegativity, radius, group, row, mass, Mendeleev no.], z-scored."""
    def f(v):
        try:
            return float(v)
        except Exception:
            return np.nan
    T = np.full((95, 7), np.nan)
    for z in range(1, 95):
        e = Element.from_Z(z)
        T[z] = [z, f(e.X), f(e.atomic_radius), f(e.group), f(e.row), f(e.atomic_mass), f(e.mendeleev_no)]
    mu = np.nanmean(T[1:], 0)
    T[1:] = np.where(np.isnan(T[1:]), mu, T[1:])
    T[1:] = (T[1:] - mu) / (T[1:].std(0) + 1e-8)
    T[0] = 0
    return T.astype(np.float32)


ELEM = _elem_table()


def comp_feats(Z):
    """Composition-only descriptors (fraction-weighted mean/std, min, max, range of element props)."""
    u, c = np.unique(Z, return_counts=True)
    fr, P = c / c.sum(), ELEM[u]
    mean = (fr[:, None] * P).sum(0)
    std = np.sqrt((fr[:, None] * (P - mean) ** 2).sum(0))
    return np.concatenate([mean, std, P.min(0), P.max(0), P.max(0) - P.min(0), [len(u)]]).astype(np.float32)


# ----------------------------------------------------------------------------- data
def fl(d, k):
    try:
        v = float(d.get(k))
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def extract_y(d):
    g, m, ef = fl(d, "optb88vdw_bandgap"), fl(d, "mbj_bandgap"), fl(d, "formation_energy_peratom")
    me, mh = fl(d, "avg_elec_mass"), fl(d, "avg_hole_mass")
    eps = [fl(d, "epsx"), fl(d, "epsy"), fl(d, "epsz")]
    e = float(np.mean(eps)) if not np.isnan(eps).any() else np.nan
    me = me if (me > 0 and me < 20) else np.nan       # drop non-physical / metallic placeholders
    mh = mh if (mh > 0 and mh < 20) else np.nan
    e = e if (e > 0 and e < 1000) else np.nan
    lg = lambda x: float(np.log10(x)) if np.isfinite(x) else np.nan
    return [g, m, ef, lg(me), lg(mh), lg(e)]


def load_raw():
    from jarvis.db.figshare import data as jdata
    return jdata("dft_3d")


def struct_to_arrays(s):
    try:
        Z = np.array([site.specie.Z for site in s], dtype=np.int64)
    except Exception:
        return None                                    # disordered / unsupported
    if Z.max() > 94:
        return None
    src, dst, dist = [], [], []
    for i, nb in enumerate(s.get_all_neighbors(R_CUT)):
        for n in sorted(nb, key=lambda n: n.nn_distance)[:MAX_NBR]:
            src.append(n.index); dst.append(i); dist.append(n.nn_distance)
    if not src:
        return None
    return Z, np.array(src), np.array(dst), np.array(dist, dtype=np.float32), comp_feats(Z)


def arrays_to_data(arr, y, jid, formula):
    Z, src, dst, dist, cf = arr
    return Data(x=torch.tensor(Z), edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
                edge_attr=torch.tensor(dist).view(-1, 1), cf=torch.tensor(cf).view(1, -1),
                y=torch.tensor(np.asarray(y, dtype=np.float32)).view(1, -1), jid=jid, formula=formula)


def _raw_worker(job):
    jid, lat, el, co, cart, y = job
    try:
        s = Structure(Lattice(lat), el, co, coords_are_cartesian=cart)
        arr = struct_to_arrays(s)
        return None if arr is None else (jid, s.composition.reduced_formula, arr, y)
    except Exception:
        return None


def _struct_worker(s):
    try:
        return struct_to_arrays(s)
    except Exception:
        return None


def get_dataset(max_atoms=100):
    path = CACHE / "dataset.pt"
    if path.exists():
        obj = torch.load(path, weights_only=False)
        return obj["data"], obj["split"]
    CACHE.mkdir(exist_ok=True)
    raw = load_raw()
    jobs = []
    for d in raw:
        a = d["atoms"]
        if len(a["elements"]) <= max_atoms:
            jobs.append((d["jid"], a["lattice_mat"], a["elements"], a["coords"],
                         bool(a.get("cartesian", False)), extract_y(d)))
    data = []
    with Pool(os.cpu_count()) as pool:
        for r in tqdm(pool.imap(_raw_worker, jobs, chunksize=64), total=len(jobs), desc="building graphs"):
            if r is not None:
                data.append(arrays_to_data(r[2], r[3], r[0], r[1]))
        # grouped split: all structures sharing a chemical system (same element set) stay in one split
    sysid = [",".join(map(str, sorted(d.x.unique().tolist()))) for d in data]
    uniq = sorted(set(sysid))
    perm = np.random.RandomState(42).permutation(len(uniq))
    rank = {uniq[j]: r for r, j in enumerate(perm)}
    cut1, cut2 = int(.8 * len(uniq)), int(.9 * len(uniq))
    split = {"train": [], "val": [], "test": []}
    for i, g in enumerate(sysid):
        r = rank[g]
        split["train" if r < cut1 else "val" if r < cut2 else "test"].append(i)
    n = len(data)
    torch.save({"data": data, "split": split}, path)
    print(f"built {n} graphs (train/val/test = {len(split['train'])}/{len(split['val'])}/{len(split['test'])})")
    return data, split


def stack_y(dl):
    return np.concatenate([d.y.numpy() for d in dl], 0)


def to_orig(P):
    P = P.copy()
    P[:, LOG_IDX] = 10 ** P[:, LOG_IDX]
    return P


# ----------------------------------------------------------------------------- model
class CGNN(nn.Module):
    """Compact CGCNN-style network: element embedding -> gated message passing -> mean pool -> MLP."""

    def __init__(self, n_tasks, hidden=64, n_layers=3):
        super().__init__()
        self.emb = nn.Embedding(95, hidden)
        self.register_buffer("props", torch.tensor(ELEM))
        self.register_buffer("mu", torch.linspace(0, R_CUT, N_GAUSS))
        self.inp = nn.Linear(hidden + ELEM.shape[1], hidden)
        self.convs = nn.ModuleList([CGConv(hidden, dim=N_GAUSS, batch_norm=True) for _ in range(n_layers)])
        self.h1, self.h2 = nn.Linear(hidden, 128), nn.Linear(128, n_tasks)

    def forward(self, b, return_emb=False):
        x = self.inp(torch.cat([self.emb(b.x), self.props[b.x]], -1))
        e = torch.exp(-((b.edge_attr - self.mu) ** 2) / 0.2 ** 2)
        for c in self.convs:
            x = c(x, b.edge_index, e)
        h = global_mean_pool(x, b.batch)
        out = self.h2(F.softplus(self.h1(F.softplus(h))))
        return (out, h) if return_emb else out


@torch.no_grad()
def predict(model, dl, norm, dev, bs=256, return_emb=False):
    """Returns predictions in the *stored* space (log10 for LOG_IDX targets)."""
    model.eval()
    P, E = [], []
    for b in DataLoader(dl, batch_size=bs):
        o, h = model(b.to(dev), return_emb=True)
        P.append(o.cpu()); E.append(h.cpu())
    P = torch.cat(P).numpy() * norm["std"] + norm["mean"]
    return (P, torch.cat(E).numpy()) if return_emb else P


def load_model(dev):
    ck = torch.load(CACHE / "model.pt", map_location=dev, weights_only=False)
    m = CGNN(len(TARGETS), ck["cfg"]["hidden"], ck["cfg"]["layers"]).to(dev)
    m.load_state_dict(ck["state"])
    return m, ck["norm"]


def masked_mae(p, y):
    m = ~np.isnan(y)
    return float(np.abs(p[m] - y[m]).mean())


# ----------------------------------------------------------------------------- helpers
def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def update_readme():
    p = Path("README.md")
    if not p.exists():
        return
    parts = [Path(f).read_text() for f in ["results/audit.md", "results/metrics.md", "results/screening.md"]
             if Path(f).exists()]
    block = "<!-- RESULTS_START -->\n" + "\n\n".join(parts) + "\n<!-- RESULTS_END -->"
    p.write_text(re.sub(r"<!-- RESULTS_START -->.*?<!-- RESULTS_END -->", lambda m: block, p.read_text(), flags=re.S))


# ----------------------------------------------------------------------------- stage 1: audit
def cmd_audit(args):
    RES.mkdir(exist_ok=True)
    raw = load_raw()
    keys = set().union(*[set(d.keys()) for d in raw[:200]])
    for k in ["optb88vdw_bandgap", "mbj_bandgap", "formation_energy_peratom", "avg_elec_mass",
              "avg_hole_mass", "epsx", "ehull"]:
        if k not in keys:
            print(f"WARNING: key '{k}' not found in dft_3d records. Available keys: {sorted(keys)}")
    Y = np.array([extract_y(d) for d in raw])
    has = ~np.isnan(Y)
    gap = Y[:, 0]
    semi = has[:, 0] & (gap >= 0.1)
    win = has[:, 0] & (gap >= 0.5) & (gap <= 4.0)
    rows = [("All records", len(raw), len(raw))]
    for i, t in enumerate(TARGETS):
        rows.append((f"has {t}", int(has[:, i].sum()), int((has[:, i] & semi).sum())))
    rows += [("gap & formation energy", int((has[:, 0] & has[:, 2]).sum()), int((has[:, 0] & has[:, 2] & semi).sum())),
             ("gap & electron+hole mass", int((has[:, 0] & has[:, 3] & has[:, 4]).sum()),
              int((has[:, 0] & has[:, 3] & has[:, 4] & semi).sum())),
             ("gap & dielectric", int((has[:, 0] & has[:, 5]).sum()), int((has[:, 0] & has[:, 5] & semi).sum())),
             ("gap & Ef & masses & dielectric (all)", int(has[:, [0, 2, 3, 4, 5]].all(1).sum()),
              int((has[:, [0, 2, 3, 4, 5]].all(1) & semi).sum()))]
    df = pd.DataFrame(rows, columns=["subset", "records", "of which gap >= 0.1 eV"])
    txt = (f"### Data audit (JARVIS dft_3d)\n\nSemiconductor-like (OptB88vdW gap >= 0.1 eV): **{int(semi.sum())}**; "
           f"in the 0.5-4.0 eV window: **{int(win.sum())}**.\n\n" + md_table(df))
    (RES / "audit.md").write_text(txt)
    df.to_csv(RES / "audit.csv", index=False)
    print(txt)
    update_readme()


# ----------------------------------------------------------------------------- stage 2: train
def cmd_train(args):
    CACHE.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
    data, split = get_dataset(args.max_atoms)
    tr, va = [data[i] for i in split["train"]], [data[i] for i in split["val"]]
    Ytr = stack_y(tr)
    norm = {"mean": np.nanmean(Ytr, 0), "std": np.nanstd(Ytr, 0) + 1e-8}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CGNN(len(TARGETS), args.hidden, args.layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    mean, std = torch.tensor(norm["mean"], dtype=torch.float, device=dev), torch.tensor(norm["std"], dtype=torch.float, device=dev)
    loader = DataLoader(tr, batch_size=args.bs, shuffle=True)
    Yva, best, log = stack_y(va), 1e9, []
    torch.manual_seed(args.seed)
    for ep in range(1, args.epochs + 1):
        model.train(); tot = n = 0
        for b in loader:
            b = b.to(dev)
            mask = ~torch.isnan(b.y)
            ys = torch.nan_to_num((b.y - mean) / std)
            diff = (model(b) - ys).abs() * mask
            cnt = mask.sum(0)
            loss = (diff.sum(0) / cnt.clamp(min=1))[cnt > 0].mean()   # masked, per-task-averaged L1
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += loss.item() * b.num_graphs; n += b.num_graphs
        sched.step()
        val_mae = masked_mae(predict(model, va, norm, dev)[:, 0], Yva[:, 0])
        log.append({"epoch": ep, "train_loss": tot / n, "val_gap_mae_eV": val_mae})
        flag = ""
        if val_mae < best:
            best = val_mae
            torch.save({"state": model.state_dict(), "norm": norm,
                        "cfg": {"hidden": args.hidden, "layers": args.layers}}, CACHE / "model.pt")
            flag = "  *saved*"
        print(f"epoch {ep:3d} | train loss {tot / n:.4f} | val band-gap MAE {val_mae:.4f} eV{flag}")
    pd.DataFrame(log).to_csv(RES / "train_log.csv", index=False)


# ----------------------------------------------------------------------------- stage 3: evaluate
def cmd_evaluate(args):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.decomposition import PCA
    RES.mkdir(exist_ok=True)
    data, split = get_dataset(args.max_atoms)
    tr, te = [data[i] for i in split["train"]], [data[i] for i in split["test"]]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, norm = load_model(dev)
    Pg, E = predict(model, te, norm, dev, return_emb=True)
    Yte, Ytr = stack_y(te), stack_y(tr)
    Xtr, Xte = np.concatenate([d.cf.numpy() for d in tr]), np.concatenate([d.cf.numpy() for d in te])
    Pr = np.full_like(Pg, np.nan)
    for i, t in enumerate(TARGETS):                   # composition-only baseline, one RF per target
        m = ~np.isnan(Ytr[:, i])
        rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0).fit(Xtr[m], Ytr[m, i])
        Pr[:, i] = rf.predict(Xte)
        print("RF baseline done:", t)

    def r2(p, y):
        m = ~np.isnan(y)
        return float(1 - ((y[m] - p[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum())
    rows = []
    for i, t in enumerate(TARGETS):
        rows.append({"target": t, "unit": UNITS[i], "N_test": int((~np.isnan(Yte[:, i])).sum()),
                     "GNN MAE": masked_mae(Pg[:, i], Yte[:, i]), "RF MAE": masked_mae(Pr[:, i], Yte[:, i]),
                     "GNN R2": r2(Pg[:, i], Yte[:, i]), "RF R2": r2(Pr[:, i], Yte[:, i])})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "metrics.csv", index=False)
    (RES / "metrics.md").write_text("### Test-set performance (chemical-system-grouped 80/10/10 split)\n\n" + md_table(df) +
                                    "\n\nMasses and dielectric constant are evaluated on a log10 scale.")
    print(md_table(df))

    y, pg, pr = Yte[:, 0], Pg[:, 0], Pr[:, 0]
    m = ~np.isnan(y)
    # 1) parity plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for a, p, nm in zip(ax, [pg, pr], ["GNN (crystal structure)", "Random Forest (composition only)"]):
        a.hexbin(y[m], p[m], gridsize=60, bins="log", mincnt=1)
        lim = [0, max(y[m].max(), p[m].max())]
        a.plot(lim, lim, "r--", lw=1)
        a.set_title(f"{nm}\nMAE = {np.abs(y[m] - p[m]).mean():.3f} eV"); a.set_xlabel("DFT band gap (eV)")
    ax[0].set_ylabel("Predicted band gap (eV)")
    fig.tight_layout(); fig.savefig(RES / "parity_bandgap.png", dpi=200); plt.close(fig)
    # 2) error vs band gap
    bins = [0, 0.1, 0.5, 1, 2, 3, 4, 6, 20]
    lab = [f"{a}-{b}" for a, b in zip(bins[:-1], bins[1:])]
    idx = np.digitize(y[m], bins) - 1
    eg = [np.abs(pg[m] - y[m])[idx == k].mean() if (idx == k).any() else 0 for k in range(len(lab))]
    er = [np.abs(pr[m] - y[m])[idx == k].mean() if (idx == k).any() else 0 for k in range(len(lab))]
    fig, a = plt.subplots(figsize=(7, 4)); w = 0.4; xs = np.arange(len(lab))
    a.bar(xs - w / 2, eg, w, label="GNN"); a.bar(xs + w / 2, er, w, label="RF (composition)")
    a.set_xticks(xs); a.set_xticklabels(lab); a.set_xlabel("DFT band-gap bin (eV)"); a.set_ylabel("MAE (eV)"); a.legend()
    fig.tight_layout(); fig.savefig(RES / "error_by_gap.png", dpi=200); plt.close(fig)
    # 3) learned-representation map
    pc = PCA(2).fit_transform(E)
    fig, a = plt.subplots(figsize=(6, 5))
    sc = a.scatter(pc[m, 0], pc[m, 1], c=y[m], s=4, cmap="viridis", vmax=6)
    plt.colorbar(sc, label="DFT band gap (eV)"); a.set_xlabel("PC1 of GNN embedding"); a.set_ylabel("PC2")
    fig.tight_layout(); fig.savefig(RES / "embedding_pca.png", dpi=200); plt.close(fig)
    # 4) per-element error (which chemistries are hard?)
    err = np.abs(pg - y)
    acc = {}
    for d, e in zip(te, err):
        if np.isnan(e):
            continue
        for z in d.x.unique().tolist():
            acc.setdefault(z, []).append(e)
    el = pd.DataFrame([{"element": Element.from_Z(z).symbol, "n": len(v), "mae_eV": float(np.mean(v))}
                       for z, v in acc.items() if len(v) >= 50]).sort_values("mae_eV")
    el.to_csv(RES / "element_error.csv", index=False)
    show = pd.concat([el.head(15), el.tail(15)])
    fig, a = plt.subplots(figsize=(6, 7))
    a.barh(show["element"], show["mae_eV"], color=["tab:green"] * 15 + ["tab:red"] * 15)
    a.set_xlabel("GNN band-gap MAE (eV), test materials containing element"); a.invert_yaxis()
    fig.tight_layout(); fig.savefig(RES / "element_error.png", dpi=200); plt.close(fig)
    # 5) polymorph test: same composition, different structure -> composition-only models cannot tell them apart
    fm = {}
    for k, d in enumerate(te):
        fm.setdefault(d.formula, []).append(k)
    ks = [k for v in fm.values() if len(v) >= 2 and np.nanmax(y[v]) - np.nanmin(y[v]) > 0.3
          for k in v if not np.isnan(y[k])]
    if ks:
        ks = np.array(ks)
        pm = (f"\n\n### Polymorph test\n\nTest materials whose formula has >=2 structures with gaps differing by "
              f">0.3 eV (n = {len(ks)}): GNN MAE {np.abs(pg[ks] - y[ks]).mean():.3f} eV vs "
              f"RF MAE {np.abs(pr[ks] - y[ks]).mean():.3f} eV.")
        (RES / "metrics.md").write_text((RES / "metrics.md").read_text() + pm)
        print(pm)
    update_readme()


# ----------------------------------------------------------------------------- stage 4: screen
def substitution_pool(raw, data, n_parents, pool_size, seed=0):
    """Isovalent element substitution on stable, gapped JARVIS parents -> compositions absent from JARVIS."""
    rng = np.random.RandomState(seed)
    cnt = {}
    for d in data:
        for z in d.x.unique().tolist():
            cnt[z] = cnt.get(z, 0) + 1
    allowed = []
    for z, c in cnt.items():
        e = Element.from_Z(z)
        if c >= 200 and z > 1 and e.group != 18 and not (e.is_lanthanoid or e.is_actinoid) and z <= 83 and z not in (43, 61):
            allowed.append(z)
    group = {z: Element.from_Z(z).group for z in allowed}
    parents = [d for d in raw if len(d["atoms"]["elements"]) <= 20 and fl(d, "ehull") <= 0.05 and fl(d, "optb88vdw_bandgap") >= 0.5]
    rng.shuffle(parents)
    seen = {d.formula for d in data}
    pool = []
    for d in parents[:n_parents]:
        a = d["atoms"]
        s = Structure(Lattice(a["lattice_mat"]), a["elements"], a["coords"], coords_are_cartesian=bool(a.get("cartesian", False)))
        ox = s.composition.oxi_state_guesses(max_sites=-1)
        if not ox:
            continue
        for sym in sorted({e.symbol for e in s.composition.elements}):
            z = Element(sym).Z
            if z not in group or sym not in ox[0]:
                continue
            q = int(round(ox[0][sym]))
            for z2 in allowed:
                if z2 == z or group[z2] != group[z]:
                    continue
                s2 = s.copy()
                new = Element.from_Z(z2).symbol
                s2.replace_species({sym: new})
                try:
                    ratio = float(Element.from_Z(z2).atomic_radius / Element(sym).atomic_radius)
                except TypeError:
                    continue
                if not 0.7 <= ratio <= 1.4:   # skip big size mismatches: unrelaxed cell would be unphysical
                    continue
                s2.scale_lattice(s.volume * (1 + s.composition.get_atomic_fraction(sym) * (ratio ** 3 - 1)))
                f = s2.composition.reduced_formula
                if f not in seen and q in Element.from_Z(z2).common_oxidation_states:
                    seen.add(f); pool.append((s2, d["jid"], f"{sym}->{new}", f))
    rng.shuffle(pool)
    return pool[:pool_size]


def cmd_screen(args):
    RES.mkdir(exist_ok=True)
    data, split = get_dataset(args.max_atoms)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, norm = load_model(dev)
    lo, hi, efm = args.gap_min, args.gap_max, args.ef_max
    # (a) validation on held-out DFT-labelled structures
    te = [data[i] for i in split["test"]]
    Y, P = to_orig(stack_y(te)), to_orig(predict(model, te, norm, dev))
    ok = ~np.isnan(Y[:, 0]) & ~np.isnan(Y[:, 2])
    crit = lambda A: (A[:, 0] >= lo) & (A[:, 0] <= hi) & (A[:, 2] <= efm)
    t, p = crit(Y)[ok], crit(P)[ok]
    tp = int((t & p).sum())
    prec, rec, base = tp / max(p.sum(), 1), tp / max(t.sum(), 1), t.mean()
    from sklearn.ensemble import RandomForestRegressor
    trd = [data[i] for i in split["train"]]
    Xtr, Ytr = np.concatenate([d.cf.numpy() for d in trd]), stack_y(trd)
    Xte = np.concatenate([d.cf.numpy() for d in te])
    Prf = P.copy()
    for i in (0, 2):
        mm = ~np.isnan(Ytr[:, i])
        Prf[:, i] = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0).fit(Xtr[mm], Ytr[mm, i]).predict(Xte)
    pr_ = crit(Prf)[ok]
    rf_prec, rf_rec = int((t & pr_).sum()) / max(pr_.sum(), 1), int((t & pr_).sum()) / max(t.sum(), 1)
    txt = (f"### Screening\n\nCriteria: {lo} <= gap <= {hi} eV, formation energy <= {efm} eV/atom"
           f"{f' (m* cap applied to discovery pool only)' if args.mass_max else ''}.\n\n"
           f"**Validation on held-out test set** (n = {int(ok.sum())}): precision {prec:.2f}, recall {rec:.2f}, "
           f"base rate {base:.2f}, enrichment over random {prec / max(base, 1e-9):.1f}x.\n"
           f"Composition-only RF on the same task: precision {rf_prec:.2f}, recall {rf_rec:.2f}.\n")
    print(txt)
    # (b) discovery pool: substituted structures (+ optional CIF directory)
    pool = []
    if args.pool_size > 0:
        pool += substitution_pool(load_raw(), data, args.n_parents, args.pool_size)
    if args.cif_dir:
        for f in sorted(glob.glob(os.path.join(args.cif_dir, "*.cif"))):
            try:
                s = Structure.from_file(f)
                pool.append((s, "cif:" + os.path.basename(f), "user", s.composition.reduced_formula))
            except Exception:
                pass
    if pool:
        with Pool(os.cpu_count()) as pl:
            arrs = list(tqdm(pl.imap(_struct_worker, [x[0] for x in pool], chunksize=32), total=len(pool), desc="graphs"))
        keep = [(x, a) for x, a in zip(pool, arrs) if a is not None]
        nanY = [np.nan] * len(TARGETS)
        ds = [arrays_to_data(a, nanY, x[1], x[3]) for x, a in keep]
        Pc = to_orig(predict(model, ds, norm, dev))
        df = pd.DataFrame({"formula": [x[3] for x, _ in keep], "parent": [x[1] for x, _ in keep],
                           "substitution": [x[2] for x, _ in keep], "pred_gap_eV": Pc[:, 0],
                           "pred_Ef_eV_per_atom": Pc[:, 2], "pred_m_e": Pc[:, 3], "pred_m_h": Pc[:, 4],
                           "pred_dielectric": Pc[:, 5]})
        sel = (df.pred_gap_eV.between(lo, hi)) & (df.pred_Ef_eV_per_atom <= efm)
        if args.mass_max:
            sel &= (df.pred_m_e.between(0.05, args.mass_max)) & (df.pred_m_h.between(0.05, args.mass_max))
        cand = df[sel].drop_duplicates("formula").copy()
        cand["anion"] = cand.formula.map(lambda f: next((a for a in ["F", "Cl", "Br", "I", "O", "S", "Se", "Te", "N", "P", "As"]
                                                         if a in re.findall(r"[A-Z][a-z]?", f)), "other"))
        cand = (cand.sort_values("pred_Ef_eV_per_atom").groupby("parent").head(3)
                .groupby("anion").head(15).head(args.top))
        cand.to_csv(RES / "candidates.csv", index=False)
        fig, a = plt.subplots(figsize=(6, 4.5))
        a.scatter(df.pred_gap_eV, df.pred_Ef_eV_per_atom, s=3, alpha=.3, c="grey")
        a.scatter(cand.pred_gap_eV, cand.pred_Ef_eV_per_atom, s=8, c="tab:red", label="candidates")
        a.axvspan(lo, hi, color="tab:blue", alpha=.08); a.axhline(efm, color="k", lw=.8, ls="--")
        a.set_xlabel("Predicted band gap (eV)"); a.set_ylabel("Predicted formation energy (eV/atom)"); a.legend()
        fig.tight_layout(); fig.savefig(RES / "candidates_scatter.png", dpi=200); plt.close(fig)
        txt += (f"\n**Discovery pool**: {len(df)} unrelaxed isovalent-substituted structures screened; "
                f"{int(sel.sum())} pass the criteria; top 10 by predicted formation energy:\n\n"
                + md_table(cand.groupby("anion").head(2).head(12)) + "\n\nCandidates are model predictions on unrelaxed geometries and "
                "must be validated with DFT (relaxation, band structure, phonons/E_hull) before any claim.")
    (RES / "screening.md").write_text(txt)
    update_readme()


# ----------------------------------------------------------------------------- CLI
if __name__ == "__main__":
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--max-atoms", type=int, default=100)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit", parents=[common])
    t = sub.add_parser("train", parents=[common])
    t.add_argument("--epochs", type=int, default=60); t.add_argument("--bs", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--layers", type=int, default=3); t.add_argument("--seed", type=int, default=0)
    sub.add_parser("evaluate", parents=[common])
    s = sub.add_parser("screen", parents=[common])
    s.add_argument("--gap-min", type=float, default=1.0); s.add_argument("--gap-max", type=float, default=2.0)
    s.add_argument("--ef-max", type=float, default=0.0, help="max predicted formation energy (eV/atom)")
    s.add_argument("--mass-max", type=float, default=None, help="optional cap on predicted e/h effective mass")
    s.add_argument("--pool-size", type=int, default=20000); s.add_argument("--n-parents", type=int, default=1500)
    s.add_argument("--cif-dir", type=str, default=None); s.add_argument("--top", type=int, default=200)
    a = ap.parse_args()
    {"audit": cmd_audit, "train": cmd_train, "evaluate": cmd_evaluate, "screen": cmd_screen}[a.cmd](a)