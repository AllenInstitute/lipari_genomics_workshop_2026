"""Build the CLEAN, atlas-filtered Session-2 object (cellxgene-ready).

Session 1 is about QC: students filter the noisy subsample themselves. Session 2
(mapping / annotation) instead needs a *clean* starting point, so this script:

  1. loads the Session-1 workshop subsample (``SpC_workshop_snRNA.h5ad``),
  2. keeps only the nuclei the **published atlas** passed (``keeper_cells`` /
     ``qc_status == 'passed_qc'``) - i.e. we apply the atlas filtering, not a
     student's,
  3. **retrains scVI from scratch** on just those clean nuclei (integrating across
     ``species`` / ``study`` exactly like the atlas) and recomputes a UMAP,
  4. log-normalizes ``X`` for gene visualization (raw counts kept in a layer),
  5. writes a full processed copy **and** a cellxgene-safe copy to ``/results``.

Import order matters: torch + scvi MUST be imported before numpy/anndata/scanpy or
scVI training dies with ``CUBLAS_STATUS_NOT_INITIALIZED`` (BLAS load-order clash).

Outputs:
  * ``cfg.SESSION2_OUT``            - full processed clean object (X_scVI + X_umap)
  * ``cfg.SESSION2_CELLXGENE_OUT``  - cellxgene-safe copy
  * ``cfg.SESSION2_SCVI_MODEL_DIR`` - the retrained scVI model

Run::  python 02b_build_session2_clean.py
"""
import os
import shutil
import subprocess
import sys

import torch
import scvi

import anndata as ad
import numpy as np

sys.path.append(os.path.dirname(__file__))
from importlib import import_module

cfg = import_module('00_config')

# obsm carried over from the subsample that no longer make sense once we refilter
# and re-embed (the prefilter UMAP was computed on QC-passed + QC-failed nuclei).
_DROP_OBSM = ['X_umap_prefilter']


def main():
    import random
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    scvi.settings.seed = cfg.SEED
    torch.set_float32_matmul_precision('high')

    print('Loading', cfg.SNRNA_OUT)
    adata = ad.read_h5ad(cfg.SNRNA_OUT)
    print(f'  {adata.n_obs:,} nuclei x {adata.n_vars:,} genes (pre-filter subsample)')

    # ── 1. Apply the ATLAS filtering: keep only nuclei the atlas passed ─────────
    keep = adata.obs['keeper_cells'].astype(bool).values
    adata = adata[keep].copy()
    print(f'  {adata.n_obs:,} nuclei retained after atlas filtering '
          f'(qc_status == "passed_qc")')

    # Drop now-meaningless embeddings; we recompute a fresh integrated UMAP below.
    for k in _DROP_OBSM:
        if k in adata.obsm:
            del adata.obsm[k]

    # Raw counts (scVI requires integer-like counts, not normalized values).
    if 'counts' not in adata.layers:
        adata.layers['counts'] = adata.X.copy()
    adata.X = adata.layers['counts'].copy()

    # Sequencing-depth covariate used by the atlas integration.
    adata.obs['log_umi_counts'] = np.log10(
        np.asarray(adata.layers['counts'].sum(1)).ravel() + 1.0)

    # scVI needs clean categorical batch / covariate columns (no NaN, no unused cats).
    for col in [cfg.SCVI_BATCH_KEY] + cfg.SCVI_CATEGORICAL_COVS:
        s = adata.obs[col].astype(str).fillna('unknown')
        adata.obs[col] = s.astype('category')

    # ── 2. Retrain scVI from scratch on the clean nuclei ───────────────────────
    print('Setting up scVI (batch_key=%r, categorical=%r, continuous=%r)...'
          % (cfg.SCVI_BATCH_KEY, cfg.SCVI_CATEGORICAL_COVS, cfg.SCVI_CONTINUOUS_COVS))
    scvi.model.SCVI.setup_anndata(
        adata,
        layer='counts',
        batch_key=cfg.SCVI_BATCH_KEY,
        categorical_covariate_keys=cfg.SCVI_CATEGORICAL_COVS,
        continuous_covariate_keys=cfg.SCVI_CONTINUOUS_COVS,
    )
    model = scvi.model.SCVI(
        adata,
        n_latent=cfg.SCVI_N_LATENT,
        n_layers=cfg.SCVI_N_LAYERS,
        n_hidden=cfg.SCVI_N_HIDDEN,
    )
    print(f'Training scVI for up to {cfg.SCVI_MAX_EPOCHS} epochs...')
    model.train(max_epochs=cfg.SCVI_MAX_EPOCHS)
    adata.obsm[cfg.SESSION2_LATENT_KEY] = model.get_latent_representation()
    print('  wrote latent', cfg.SESSION2_LATENT_KEY,
          adata.obsm[cfg.SESSION2_LATENT_KEY].shape)

    model.save(cfg.SESSION2_SCVI_MODEL_DIR, overwrite=True, save_anndata=False)
    print('  saved scVI model to', cfg.SESSION2_SCVI_MODEL_DIR)

    # ── 3. Fresh integrated UMAP from the clean scVI latent ────────────────────
    print('Computing integrated UMAP from the clean scVI latent...')
    import scanpy as sc  # imported here (after training) — see note at top of file
    sc.settings.seed = cfg.SEED
    tmp = ad.AnnData(np.zeros((adata.n_obs, 1), dtype='float32'))
    tmp.obsm['X_scVI'] = adata.obsm[cfg.SESSION2_LATENT_KEY]
    sc.pp.neighbors(tmp, use_rep='X_scVI',
                    n_neighbors=cfg.SCVI_UMAP_N_NEIGHBORS, random_state=cfg.SEED)
    sc.tl.umap(tmp, random_state=cfg.SEED)
    adata.obsm[cfg.SESSION2_UMAP_KEY] = tmp.obsm['X_umap']
    print('  wrote UMAP', cfg.SESSION2_UMAP_KEY,
          adata.obsm[cfg.SESSION2_UMAP_KEY].shape)

    # ── 4. Log-normalize X for gene visualization (raw counts kept in layer) ────
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Drop constant QC bookkeeping columns that are meaningless on a clean object.
    adata.obs.drop(columns=[c for c in ['keeper_cells', 'qc_status']
                            if c in adata.obs.columns], inplace=True)

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    adata.write(cfg.SESSION2_OUT)
    print('Wrote', cfg.SESSION2_OUT)
    print('  obsm now:', list(adata.obsm.keys()))

    # ── 5. cellxgene-safe copy (make_safe prunes uns in place → use a temp copy) ─
    script = os.path.join(os.path.dirname(__file__), 'make_safe_h5ad.py')
    tmp_path = cfg.SESSION2_CELLXGENE_OUT + '.tmp.h5ad'
    shutil.copy(cfg.SESSION2_OUT, tmp_path)
    subprocess.run([sys.executable, script, tmp_path, cfg.SESSION2_CELLXGENE_OUT],
                   check=True)
    os.remove(tmp_path)
    print('Wrote', cfg.SESSION2_CELLXGENE_OUT)


if __name__ == '__main__':
    main()
