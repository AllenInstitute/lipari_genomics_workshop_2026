"""Add a pre-filter scVI latent space + integrated UMAP to the workshop object.

Why this exists
---------------
The student notebook's main lesson is QC: students choose thresholds and *remove*
low-quality nuclei. To make that tangible we want them to **see where their
filtered vs. unfiltered nuclei land on a UMAP** before they cut anything. That
requires an embedding computed on the WHOLE subsample (QC-passed *and* QC-failed
cells), which is exactly what this script builds.

It trains a fresh scVI model on every nucleus in ``SpC_workshop_snRNA.h5ad``,
integrating across ``species`` (and ``study``) the same way the published atlas
did, then computes a UMAP from that latent space. Both are written back into the
same h5ad so the notebook can simply read ``obsm`` - no GPU needed at workshop
time.

Output obsm keys (added in place to ``cfg.SNRNA_OUT``):
  * ``cfg.SCVI_LATENT_KEY``  (``X_scVI``)          - workshop scVI latent (all cells)
  * ``cfg.SCVI_UMAP_KEY``    (``X_umap_prefilter``) - integrated UMAP (all cells)

Run::  python 01b_scvi_umap_prefilter.py
"""
import os
import sys

# ── Import order matters here ──────────────────────────────────────────────────
# torch + scvi MUST be imported before numpy/anndata/scanpy. If a numpy/scipy
# BLAS library is loaded first it shadows the one torch needs and scVI training
# dies with ``CUBLAS_STATUS_NOT_INITIALIZED``. So torch/scvi come first, and
# scanpy is imported lazily inside the (CPU-only) UMAP step after training.
import torch
import scvi

import anndata as ad
import numpy as np

sys.path.append(os.path.dirname(__file__))
from importlib import import_module

cfg = import_module('00_config')


def main():
    # Seed RNGs (numpy / random) and scvi. We avoid cfg.set_all_seeds() here so we
    # do not import scanpy or re-init torch CUDA before the scvi import above.
    import random
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    scvi.settings.seed = cfg.SEED
    torch.set_float32_matmul_precision('high')

    print('Loading', cfg.SNRNA_OUT)
    adata = ad.read_h5ad(cfg.SNRNA_OUT)
    print(f'  {adata.n_obs:,} nuclei x {adata.n_vars:,} genes '
          f'(pre-filter: includes QC-passed AND QC-failed)')

    # Raw counts (scVI requires integer-like counts, not normalized values).
    if 'counts' not in adata.layers:
        adata.layers['counts'] = adata.X.copy()

    # Sequencing-depth covariate used by the atlas integration.
    adata.obs['log_umi_counts'] = np.log10(
        np.asarray(adata.layers['counts'].sum(1)).ravel() + 1.0)

    # ── Train scVI, integrating across species/study ───────────────────────────
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
    adata.obsm[cfg.SCVI_LATENT_KEY] = model.get_latent_representation()
    print('  wrote latent', cfg.SCVI_LATENT_KEY, adata.obsm[cfg.SCVI_LATENT_KEY].shape)

    # Persist the trained model so the embedding is reproducible / reusable.
    model.save(cfg.SCVI_MODEL_DIR, overwrite=True, save_anndata=False)
    print('  saved scVI model to', cfg.SCVI_MODEL_DIR)

    # ── UMAP from the scVI latent (computed on a throwaway copy so we do not ────
    #    leave a neighbors graph in the saved object that the notebook would
    #    otherwise inherit when it builds its own).
    print('Computing integrated UMAP from the scVI latent...')
    import scanpy as sc  # imported here (after training) — see note at top of file
    sc.settings.seed = cfg.SEED
    tmp = ad.AnnData(np.zeros((adata.n_obs, 1), dtype='float32'))
    tmp.obsm['X_scVI'] = adata.obsm[cfg.SCVI_LATENT_KEY]
    sc.pp.neighbors(tmp, use_rep='X_scVI',
                    n_neighbors=cfg.SCVI_UMAP_N_NEIGHBORS, random_state=cfg.SEED)
    sc.tl.umap(tmp, random_state=cfg.SEED)
    adata.obsm[cfg.SCVI_UMAP_KEY] = tmp.obsm['X_umap']
    print('  wrote UMAP', cfg.SCVI_UMAP_KEY, adata.obsm[cfg.SCVI_UMAP_KEY].shape)

    adata.write(cfg.SNRNA_OUT)
    print('Updated', cfg.SNRNA_OUT)
    print('  obsm now:', list(adata.obsm.keys()))


if __name__ == '__main__':
    main()
