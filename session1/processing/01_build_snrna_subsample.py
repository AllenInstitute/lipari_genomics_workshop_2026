"""Build the Session-1 snRNA workshop object (cellxgene-ready).

Recipe (from workshop_prep.txt):
  * Start from the UNFILTERED integrated spinal-cord snRNA object.
  * Keep up to ``CELLS_PER_GROUP_PER_SPECIES`` cells for every (Group_V2, species)
    combination among the cells that PASSED QC (these carry the final V2 taxonomy).
  * Add QC-failed cells so that they make up ``FILTERED_OUT_FRACTION`` (40%) of the
    final object — these let students rediscover QC filtering.
  * Pull the raw counts for the selected cells out of the unfiltered object and
    apply the V2 obs annotations from the filtered object onto the kept cells.

Output: ``/results/SpC_workshop_snRNA.h5ad`` (raw counts in X, integrated scVI
latent + UMAP carried in obsm for visualization/comparison).

Run::  python 01_build_snrna_subsample.py
"""
import os
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.append(os.path.dirname(__file__))
from importlib import import_module

cfg = import_module('00_config')


def read_obs_column(f, name):
    """Read a single obs column from an open h5py file as a pandas Series.

    Handles both AnnData categorical groups and plain dataset encodings.
    """
    g = f['obs'][name]
    if isinstance(g, h5py.Group):  # categorical encoding
        cats = g['categories'][:]
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in cats])
        codes = g['codes'][:]
        out = np.where(codes >= 0, cats[np.clip(codes, 0, None)], None)
        return pd.Series(out)
    arr = g[:]
    if arr.dtype.kind == 'S':
        arr = np.array([x.decode() for x in arr])
    return pd.Series(arr)


def read_obs_index(f):
    idx = f['obs']['_index'][:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in idx])


def gather_csr_rows(h5_X, row_positions):
    """Gather a subset of rows from an on-disk CSR matrix.

    The selected rows are scattered across the full matrix, so instead of issuing
    one scattered HDF5 read per row (very slow), we read the ``data`` and
    ``indices`` arrays **sequentially** (the on-disk dataset is uncompressed) and
    slice the requested rows in memory. ``row_positions`` must be sorted/unique.
    """
    shape = tuple(h5_X.attrs['shape'])
    indptr = h5_X['indptr'][:]
    print('    reading data/indices sequentially (uncompressed full read)...')
    data_full = h5_X['data'][:]
    indices_full = h5_X['indices'][:]

    new_indptr = np.zeros(len(row_positions) + 1, dtype=np.int64)
    data_parts, indices_parts = [], []
    for i, r in enumerate(row_positions):
        s, e = int(indptr[r]), int(indptr[r + 1])
        if e > s:
            data_parts.append(data_full[s:e])
            indices_parts.append(indices_full[s:e])
        new_indptr[i + 1] = new_indptr[i] + (e - s)

    data = np.concatenate(data_parts) if data_parts else np.array([], dtype=data_full.dtype)
    indices = np.concatenate(indices_parts) if indices_parts else np.array([], dtype=np.int64)
    del data_full, indices_full
    return sp.csr_matrix((data, indices, new_indptr), shape=(len(row_positions), shape[1]))


def main():
    cfg.set_all_seeds()
    rng = np.random.default_rng(cfg.SEED)

    # ── 1. Read obs from the filtered/annotated object (kept cells + V2 taxonomy)
    print('Reading filtered obs (V2 taxonomy)...')
    with h5py.File(cfg.FILTERED_H5AD, 'r') as f:
        fin_index = read_obs_index(f)
        fin_obs = pd.DataFrame({c: read_obs_column(f, c) for c in cfg.V2_ANNOTATION_COLS})
        fin_obs[cfg.SPECIES_KEY] = read_obs_column(f, cfg.SPECIES_KEY)
        fin_obs.index = fin_index

    # ── 2. Subsample up to N cells per (Group_V2, species) among kept cells
    print('Subsampling kept cells (<=%d per %s per %s)...'
          % (cfg.CELLS_PER_GROUP_PER_SPECIES, cfg.GROUP_KEY, cfg.SPECIES_KEY))
    valid = fin_obs[cfg.GROUP_KEY].notna() & fin_obs[cfg.SPECIES_KEY].notna()
    kept_ids = []
    for (_, _), sub in fin_obs[valid].groupby([cfg.SPECIES_KEY, cfg.GROUP_KEY], observed=True):
        n = min(cfg.CELLS_PER_GROUP_PER_SPECIES, len(sub))
        kept_ids.extend(rng.choice(sub.index.values, size=n, replace=False))
    kept_ids = np.array(kept_ids)
    n_kept = len(kept_ids)
    print(f'  kept cells: {n_kept:,}')

    # ── 3. Read obs from the unfiltered object; identify QC-failed pool
    print('Reading unfiltered obs...')
    carry_cols = ['keeper_cells', cfg.SPECIES_KEY] + cfg.QC_CARRY_COLS
    with h5py.File(cfg.UNFILTERED_H5AD, 'r') as f:
        unf_index = read_obs_index(f)
        unf_obs = pd.DataFrame(
            {c: read_obs_column(f, c).values for c in carry_cols if c in f['obs']},
            index=unf_index)
    keeper = unf_obs['keeper_cells'].astype(bool).values
    pos_of = {cid: i for i, cid in enumerate(unf_index)}

    # ── 4. Sample QC-failed cells so they are 40% of the final object
    n_filtered_out = int(round(n_kept * cfg.FILTERED_OUT_FRACTION /
                               (1.0 - cfg.FILTERED_OUT_FRACTION)))
    failed_ids = unf_index[~keeper]
    n_filtered_out = min(n_filtered_out, len(failed_ids))
    filtered_out_ids = rng.choice(failed_ids, size=n_filtered_out, replace=False)
    print(f'  filtered-out cells: {n_filtered_out:,} '
          f'({n_filtered_out / (n_kept + n_filtered_out):.1%} of total)')

    # ── 5. Resolve integer row positions in the unfiltered object (sorted/unique)
    selected_ids = np.concatenate([kept_ids, filtered_out_ids])
    positions = np.array(sorted({pos_of[c] for c in selected_ids}), dtype=np.int64)
    selected_index = unf_index[positions]

    # ── 6. Gather raw-count rows + obsm slices from the unfiltered object
    print(f'Gathering counts for {len(positions):,} cells from unfiltered X...')
    with h5py.File(cfg.UNFILTERED_H5AD, 'r') as f:
        X = gather_csr_rows(f['X'], positions)
        var_index = np.array([x.decode() if isinstance(x, bytes) else x
                              for x in f['var']['_index'][:]])
        obsm = {}
        # Carry the published atlas embeddings under *_atlas names; the workshop's
        # own integrated scVI/UMAP (computed on the full subsample by 01b) take the
        # canonical X_scVI / X_umap_prefilter keys.
        atlas_obsm_map = {'X_scVI': 'X_scVI_atlas',
                          'X_umap_scvi_integrated': 'X_umap_atlas'}
        for src, dst in atlas_obsm_map.items():
            if src in f.get('obsm', {}):
                obsm[dst] = f['obsm'][src][positions, :]

    # ── 7. Assemble obs: carried QC/propagated cols + V2 annotations on kept cells
    obs = unf_obs.reindex(selected_index).copy()
    obs['keeper_cells'] = keeper[positions]
    obs['qc_status'] = np.where(obs['keeper_cells'], 'passed_qc', 'filtered_out')
    for col in cfg.V2_ANNOTATION_COLS:
        obs[col] = fin_obs[col].reindex(selected_index).values
    # Correct known source-atlas transmitter mislabels (e.g. Sp8 CHRNA5 GABA-Gly).
    cfg.apply_v2_group_relabel(obs)
    # species: prefer the filtered-object value, backfill from unfiltered for QC-failed.
    species_fin = fin_obs[cfg.SPECIES_KEY].reindex(selected_index)
    obs['species'] = species_fin.fillna(obs[cfg.SPECIES_KEY])
    cat_cols = (cfg.V2_ANNOTATION_COLS + ['species', 'qc_status', 'leiden',
                'Class_propagated', 'Subclass_propagated', 'Group_propagated',
                'batch', 'study', 'donor_name'])
    for col in cat_cols:
        if col in obs:
            obs[col] = obs[col].astype('category')

    adata = ad.AnnData(X=X, obs=obs,
                       var=pd.DataFrame(index=var_index), obsm=obsm)
    adata.layers['counts'] = adata.X.copy()

    # Carry the curated V2 taxonomy palette from the processed atlas object so the
    # workshop plots use the same colours as the manuscript figures.
    print('Applying V2 taxonomy colors from the atlas object...')
    tax = import_module('_taxonomy_colors')
    tax.apply_v2_colors(adata, cfg.V2_COLOR_SOURCE_H5AD, cfg.V2_ANNOTATION_COLS)

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    adata.write(cfg.SNRNA_OUT)
    print('Wrote', cfg.SNRNA_OUT)
    print(adata)


if __name__ == '__main__':
    main()
