"""Build the Session-1 spatial example object (cellxgene-ready).

Mirrors the spatial analysis notebook
(``03_figure2_plot_panels.ipynb``): loads a single representative macaque MERFISH
section, keeps the spatial coordinates in ``obsm['spatial']`` and the V2 / laminar
annotations in obs, and writes a small static h5ad for cellxgene.

Output: ``/results/SpC_workshop_spatial_example.h5ad``.

Run::  python 02_build_spatial_example.py
"""
import os
import sys

import scanpy as sc

sys.path.append(os.path.dirname(__file__))
from importlib import import_module

cfg = import_module('00_config')

# obs columns that are meaningful for the workshop spatial visualization.
KEEP_OBS = [
    'species', 'Class_V2', 'Subclass_V2', 'Supergroup_V2', 'Group_V2',
    'Neighborhood', 'rexed_lamina', 'spatial_descriptor_v2',
    'spc_segment', 'spc_subregion', '_section',
]


def main():
    cfg.set_all_seeds()

    src = os.path.join(cfg.SPATIAL_EXAMPLE_DIR, cfg.SPATIAL_EXAMPLE_SECTION)
    print('Reading spatial example section:', src)
    adata = sc.read_h5ad(src)

    # ── Standardize spatial coordinates into obsm['spatial'] (cellxgene/scanpy) ──
    # The source stores tissue coordinates as X_spatial or as _plot_x/_plot_y.
    if 'X_spatial' in adata.obsm:
        adata.obsm['spatial'] = adata.obsm['X_spatial'][:, :2].copy()
    elif {'_plot_x', '_plot_y'}.issubset(adata.obs.columns):
        adata.obsm['spatial'] = adata.obs[['_plot_x', '_plot_y']].to_numpy(float)

    # ── Keep only the informative obs columns ──────────────────────────────────
    keep = [c for c in KEEP_OBS if c in adata.obs.columns]
    adata.obs = adata.obs[keep].copy()
    for c in adata.obs.columns:
        if adata.obs[c].dtype == object:
            adata.obs[c] = adata.obs[c].astype('category')

    # ── Provide log-normalized values in X for marker visualization ────────────
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Carry the curated V2 taxonomy palette from the processed atlas object.
    print('Applying V2 taxonomy colors from the atlas object...')
    tax = import_module('_taxonomy_colors')
    tax.apply_v2_colors(adata, cfg.V2_COLOR_SOURCE_H5AD, cfg.V2_ANNOTATION_COLS)

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    adata.write(cfg.SPATIAL_OUT)
    print('Wrote', cfg.SPATIAL_OUT)
    print(adata)


if __name__ == '__main__':
    main()
