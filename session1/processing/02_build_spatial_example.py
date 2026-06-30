"""Build the Session-1 spatial example object (cellxgene-ready).

Mirrors the spatial analysis notebook
(``03_figure2_plot_panels.ipynb``): loads the three representative cross-species
MERFISH/Xenium sections (human, macaque, mouse) that the manuscript Figure 2
plots side-by-side, keeps the transformed tissue coordinates (``_plot_x`` /
``_plot_y``, also mirrored into ``obsm['spatial']``) and the V2 / laminar
annotations in obs, and writes a small static h5ad for cellxgene plus two
companion artifacts that let the student notebook reproduce the manuscript
section panel:

* ``/results/SpC_workshop_spatial_example.h5ad``      – 3-species neuron object
* ``/results/SpC_workshop_spatial_nn_overlay.tsv.gz`` – non-neuron grey overlay
* ``/results/SpC_workshop_spatial_meta.json``         – crop bounds + Group_V2 palette

Run::  python 02_build_spatial_example.py
"""
import json
import os
import sys

import anndata as ad
import pandas as pd
import scanpy as sc

sys.path.append(os.path.dirname(__file__))
from importlib import import_module

cfg = import_module('00_config')

# obs columns that are meaningful for the workshop spatial visualization.
# `_plot_x`/`_plot_y` are the transformed (oriented + cropped-hemisphere) tissue
# coordinates that reproduce the butterfly grey-matter shape; `_section` and
# `_in_example_hemi` drive the per-section faceting and cropping in the notebook.
KEEP_OBS = [
    'species', 'Class_V2', 'Subclass_V2', 'Supergroup_V2', 'Group_V2',
    'Neighborhood', 'rexed_lamina', 'spatial_descriptor_v2',
    'spc_segment', 'spc_subregion', '_section',
    '_plot_x', '_plot_y', '_in_example_hemi',
]


def main():
    cfg.set_all_seeds()

    # ── Load the three representative cross-species sections ───────────────────
    parts = []
    for species, fname in cfg.SPATIAL_EXAMPLE_SECTIONS.items():
        src = os.path.join(cfg.SPATIAL_EXAMPLE_DIR, fname)
        print(f'Reading {species} spatial example section:', src)
        a = sc.read_h5ad(src)
        keep = [c for c in KEEP_OBS if c in a.obs.columns]
        a.obs = a.obs[keep].copy()
        parts.append(a)

    # Same 947-gene panel across all three → straightforward concatenation. The
    # sections live in disjoint coordinate ranges, so they stay visually separate.
    adata = ad.concat(parts, join='inner', index_unique='-')
    print(f'Combined spatial object: {adata.n_obs:,} neurons across '
          f'{adata.obs["_section"].nunique()} sections, {adata.n_vars} genes')

    # ── Standardize spatial coordinates into obsm['spatial'] (cellxgene/scanpy) ─
    if {'_plot_x', '_plot_y'}.issubset(adata.obs.columns):
        adata.obsm['spatial'] = adata.obs[['_plot_x', '_plot_y']].to_numpy(float)

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

    # ── Non-neuron grey overlay, subset to the three representative sections ───
    with open(cfg.SPATIAL_SECTION_META) as f:
        sec_meta = json.load(f)
    rep_sections = sec_meta['representative_sections']
    rep_crop = {k: list(v) for k, v in sec_meta['rep_crop'].items()}

    nn = pd.read_csv(cfg.SPATIAL_NN_OVERLAY, sep='\t', index_col=0,
                     compression='gzip')
    nn = nn[nn['_section'].astype(str).isin(set(rep_sections.values()))].copy()
    nn.to_csv(cfg.SPATIAL_NN_OVERLAY_OUT, sep='\t', compression='gzip')
    print(f'Wrote {cfg.SPATIAL_NN_OVERLAY_OUT} ({len(nn):,} non-neuron cells)')

    # ── Metadata: representative sections, crop bounds, full Group_V2 palette ───
    with open(cfg.SPATIAL_GROUP_COLORS) as f:
        group_color = json.load(f)
    meta = {
        'representative_sections': rep_sections,
        'rep_crop': rep_crop,
        'group_color': group_color,
    }
    with open(cfg.SPATIAL_META_OUT, 'w') as f:
        json.dump(meta, f, indent=2)
    print('Wrote', cfg.SPATIAL_META_OUT)


if __name__ == '__main__':
    main()
