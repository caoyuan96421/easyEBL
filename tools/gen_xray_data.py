"""Generate ``easyebl/xray/data/xray_data.npz`` from the xraydb tables (developer tool).

    pip install xraydb
    python tools/gen_xray_data.py

The npz ships with easyEBL, so xraydb is not a runtime dependency.  Source data:
xraydb (M. Newville et al.), which compiles the Elam, Ravel & Sieber (2002)
tables (line energies, branching ratios, edge energies, fluorescence yields and
mass attenuation coefficients) – public-domain reference data.
"""
from __future__ import annotations

import os

import numpy as np
import xraydb

OUT = os.path.join(os.path.dirname(__file__), "..", "easyebl", "xray", "data", "xray_data.npz")

Z_MAX = 94
E_MIN_EV = 100.0  # lines below 0.1 keV are useless for EDS
E_MAX_EV = 30000.0
# electrons in each subshell (relative ionisation weight of the subshells of one family)
OCCUPANCY = {"K": 2, "L1": 2, "L2": 2, "L3": 4, "M1": 2, "M2": 2, "M3": 4, "M4": 4, "M5": 6}
FAMILY = {"K": "K", "L1": "L", "L2": "L", "L3": "L", "M1": "M", "M2": "M", "M3": "M", "M4": "M", "M5": "M"}
SHELLS = list(OCCUPANCY)


def main() -> None:
    symbols, masses = [], []
    edge_e = np.zeros((Z_MAX + 1, len(SHELLS)))  # keV, 0 = none
    edge_w = np.zeros((Z_MAX + 1, len(SHELLS)))  # fluorescence yield
    edge_j = np.zeros((Z_MAX + 1, len(SHELLS)))  # jump ratio
    rows = []
    for z in range(1, Z_MAX + 1):
        sym = xraydb.atomic_symbol(z)
        symbols.append(sym)
        masses.append(xraydb.atomic_mass(z))
        edges = xraydb.xray_edges(sym)
        for k, sh in enumerate(SHELLS):
            if sh in edges:
                e = edges[sh]
                edge_e[z, k] = e.energy / 1e3
                edge_w[z, k] = e.fyield
                edge_j[z, k] = e.jump_ratio
        for sh in SHELLS:
            if sh not in edges:
                continue
            try:
                lines = xraydb.xray_lines(sym, initial_level=sh)
            except Exception:  # noqa: BLE001
                continue
            for name, ln in lines.items():
                if not (E_MIN_EV <= ln.energy <= E_MAX_EV) or ln.intensity <= 1e-4:
                    continue
                rows.append((z, name, FAMILY[sh], sh, ln.energy / 1e3, float(ln.intensity),
                             edges[sh].energy / 1e3, float(edges[sh].fyield), OCCUPANCY[sh]))
    # combine lines of identical name+energy (xraydb lists some twice)
    z = np.array([r[0] for r in rows], np.int16)
    name = np.array([r[1] for r in rows])
    fam = np.array([r[2] for r in rows])
    shell = np.array([r[3] for r in rows])
    energy = np.array([r[4] for r in rows])
    branch = np.array([r[5] for r in rows])
    edge = np.array([r[6] for r in rows])
    fy = np.array([r[7] for r in rows])
    occ = np.array([r[8] for r in rows], float)
    # mass attenuation coefficient of every absorber at every line energy (cm²/g)
    mac = np.zeros((len(rows), Z_MAX + 1), np.float32)
    for a in range(1, Z_MAX + 1):
        mac[:, a] = xraydb.mu_elam(a, energy * 1e3)
    # attenuation grid for detector windows / continuum absorption
    grid = np.geomspace(0.05, 40.0, 700)
    mu_grid = np.zeros((Z_MAX + 1, grid.size), np.float32)
    for a in range(1, Z_MAX + 1):
        mu_grid[a] = xraydb.mu_elam(a, grid * 1e3)
    dens = np.array([0.0] + [xraydb.atomic_density(s) for s in symbols])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(
        OUT,
        symbols=np.array([""] + symbols),
        mass=np.array([0.0] + masses),
        density=dens,
        shells=np.array(SHELLS),
        edge_e=edge_e, edge_w=edge_w, edge_j=edge_j,
        line_z=z, line_name=name, line_family=fam, line_shell=shell, line_e=energy, line_branch=branch,
        line_edge=edge, line_fy=fy, line_occ=occ,
        mac=mac, grid_e=grid, mu_grid=mu_grid,
        source=np.array("xraydb %s (Elam, Ravel & Sieber 2002 tables)" % getattr(xraydb, "__version__", "?")),
    )
    print(f"{len(rows)} lines, {OUT}: {os.path.getsize(OUT) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
