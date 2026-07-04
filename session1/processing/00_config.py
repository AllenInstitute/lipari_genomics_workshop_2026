"""Shared configuration for Lipari Genomics Workshop 2026 — Session 1.

Centralizes file paths, the global random SEED, and a ``set_all_seeds`` helper so
that every processing script and the student notebook produce *identical* results
on every machine. Import with::

    import sys; sys.path.append('/code/lipari_genomics_workshop_2026/session1/processing')
    from importlib import import_module
    cfg = import_module('00_config')
    cfg.set_all_seeds()
"""
import os
import random

import numpy as np
import pandas as pd

# ── Global seed ────────────────────────────────────────────────────────────────
# A single seed drives python `random`, numpy, scanpy, and (if present) torch so
# that subsampling, clustering, and UMAP are byte-for-byte reproducible.
SEED = 0

# ── Source objects (read-only, on /scratch and /data) ──────────────────────────
UNFILTERED_H5AD = '/scratch/multispecies_integrated_realigned_qcunfiltered_SpC.h5ad'
FILTERED_H5AD = (
    '/scratch/multispecies_integrated_realigned_qcfiltered_SpC_scvi_'
    'final_cluster_manual_annotations_mnqcfiltered.h5ad'
)
SPATIAL_MANUSCRIPT_DIR = (
    '/data/SpinalCord/review_integrated/nnn_clusters/spatial/manuscript_ready'
)
SPATIAL_EXAMPLE_DIR = os.path.join(SPATIAL_MANUSCRIPT_DIR, 'example_sections')
# The three representative cross-species sections (human / macaque / mouse) that
# the manuscript Figure 2 plots side-by-side. The workshop spatial teaser mirrors
# that panel, so we load all three rather than a single section.
SPATIAL_EXAMPLE_SECTIONS = {
    'human':   'human_example_section.h5ad',
    'macaque': 'macaque_example_section.h5ad',
    'mouse':   'mouse_example_section.h5ad',
}
# Manuscript metadata reused for the spatial teaser (crop bounds, section ids,
# non-neuron overlay, and the curated Group_V2 palette). These mirror the inputs
# used by 03_figure2_plot_panels.ipynb.
SPATIAL_SECTION_META = os.path.join(SPATIAL_MANUSCRIPT_DIR, 'section_metadata.json')
SPATIAL_NN_OVERLAY = os.path.join(SPATIAL_MANUSCRIPT_DIR, 'nn_obs_overlay.tsv.gz')
SPATIAL_GROUP_COLORS = os.path.join(SPATIAL_MANUSCRIPT_DIR, 'group_colors.json')

# Canonical source for the curated V2 taxonomy colours (uns['<col>_colors']).
# This is the injury-excluded object the manuscript figures are coloured from; we
# read palettes from here rather than FILTERED_H5AD so the workshop matches it.
V2_COLOR_SOURCE_H5AD = (
    '/scratch/multispecies_integrated_realigned_qcfiltered_SpC_scvi_'
    'final_cluster_manual_annotations_mnqcfiltered_noinjury.h5ad'
)

# ── Subsample recipe parameters ────────────────────────────────────────────────
CELLS_PER_GROUP_PER_SPECIES = 100   # up to N kept cells per (Group_V2, species)
FILTERED_OUT_FRACTION = 0.40        # 40% of the final object are QC-failed cells
GROUP_KEY = 'Group_V2'
SPECIES_KEY = 'species'

# ── Workshop outputs (cellxgene-ready, must live in /results) ──────────────────
RESULTS_DIR = '/results'
SNRNA_OUT = os.path.join(RESULTS_DIR, 'SpC_workshop_snRNA.h5ad')
SPATIAL_OUT = os.path.join(RESULTS_DIR, 'SpC_workshop_spatial_example.h5ad')
# Companion artifacts for the 3-species spatial teaser (non-neuron grey overlay
# and a small metadata json with crop bounds + the Group_V2 palette), so the
# student notebook can reproduce the manuscript section panel from /results only.
SPATIAL_NN_OVERLAY_OUT = os.path.join(
    RESULTS_DIR, 'SpC_workshop_spatial_nn_overlay.tsv.gz')
SPATIAL_META_OUT = os.path.join(RESULTS_DIR, 'SpC_workshop_spatial_meta.json')

# Annotation columns copied from the filtered object onto kept cells.
V2_ANNOTATION_COLS = ['Class_V2', 'Subclass_V2', 'Supergroup_V2', 'Group_V2']

# ── Canonical taxonomy corrections ─────────────────────────────────────────────
# The source atlas mislabels the ventral inhibitory (Renshaw) interneuron
# `Sp8 CHRNA5 GABA-Gly` as glutamatergic (Class_V2=Glut / Subclass_V2=Glut-V /
# Supergroup_V2='Glut-V spoke'). It is a GABAergic/glycinergic cell, so we force it
# onto the GABA arm; applying this in the builders keeps every workshop object (and
# its cellxgene copy) consistent even if /results is regenerated from the source.
V2_GROUP_RELABEL = {
    'Sp8 CHRNA5 GABA-Gly': {
        'Class_V2': 'GABA',
        'Subclass_V2': 'GABA-V',
        'Supergroup_V2': 'GABA-V spoke',
    },
    # Split group: 4 of 5 clusters (n_202-205, ~90% of cells) are GABA-M, while the
    # minor cluster n_6 was GABA-D. Collapse the whole group onto its dominant
    # GABA-M (GABA-M TFAP2B) subclass so it renders as one coherent block.
    'Sp4M PAX5 GABA-Gly': {
        'Class_V2': 'GABA',
        'Subclass_V2': 'GABA-M',
        'Supergroup_V2': 'GABA-M TFAP2B',
    },
}


def apply_v2_group_relabel(obs, group_key=GROUP_KEY, relabel=None):
    """Force curated Class/Subclass/Supergroup_V2 for specific Group_V2 values.

    Operates in place on a pandas ``obs`` DataFrame (columns may be categorical or
    object) and returns it, correcting known source-atlas transmitter mislabels so
    all downstream workshop objects agree. Relabelled columns are left as plain
    object dtype; callers convert them back to ``category`` as usual.
    """
    relabel = V2_GROUP_RELABEL if relabel is None else relabel
    if group_key not in obs.columns:
        return obs
    groups = obs[group_key].astype(str).to_numpy()
    for group, fixes in relabel.items():
        mask = groups == group
        if not mask.any():
            continue
        for col, value in fixes.items():
            if col not in obs.columns:
                continue
            vals = obs[col].astype(str).to_numpy(dtype=object)
            vals[mask] = value
            obs[col] = vals
            print(f'    relabel: {int(mask.sum())} "{group}" cells -> {col}={value}')
    return obs

# ── Rexed lamina palette / ordering ────────────────────────────────────────────
# Curated dorsal-to-ventral lamina colours + ordering (mirrors the manuscript
# Figure 2 spatial panels). `'example'` is NOT a lamina: it is a placeholder from
# the upstream GeoJSON where a "representative hemisphere" marker polygon leaked
# into the lamina-assignment step, so cells inside that marker (but not inside any
# real lamina polygon) got the literal label 'example'. We drop it back to '' so
# it renders as unassigned grey, exactly like any other unlabelled neuron.
REGION_PALETTE = {
    'L':   '#400000',
    '1':   '#217b9b',
    '2i':  '#c56e76',
    '2o':  '#b12864',
    '3':   '#ed278a',
    '4L':  '#0fce45',
    '4M':  '#458271',
    '5L':  '#bce233',
    '5M':  '#ea49ea',
    '6L':  '#595907',
    '6M':  '#720dcc',
    '7':   '#ffd00b',
    '8':   '#d87b00',
    '9':   '#841921',   # lamina IX – skeletal motor neurons
    '10':  '#37c6f4',
    'IML': '#665cc1',
    'IMM': '#9991af',
    '':    '#b8b8b8',   # unassigned
}

REGION_ORDER = ['L', '1', '2o', '2i', '3', '4L', '4M', '5M', '5L', '6M', '6L',
                '7', '8', '10', 'IMM', 'IML', '9', '']

# Non-lamina placeholder labels to fold back into '' (unassigned).
REXED_LAMINA_DROP = ('example',)


def clean_rexed_lamina(adata, col='rexed_lamina'):
    """Normalize the ``rexed_lamina`` column on an AnnData in place.

    Drops non-lamina placeholder labels (see ``REXED_LAMINA_DROP``) to ''
    (unassigned), sets an ordered categorical following ``REGION_ORDER`` (only
    categories actually present, with any unexpected extras appended), and stores
    a matching ``uns['<col>_colors']`` palette from ``REGION_PALETTE`` so the
    colours travel with the object like the other curated palettes.
    """
    if col not in adata.obs.columns:
        return adata
    vals = adata.obs[col].astype(str)
    vals = vals.replace({lbl: '' for lbl in REXED_LAMINA_DROP})
    present = set(vals.unique())
    cats = [c for c in REGION_ORDER if c in present]
    cats += [c for c in sorted(present) if c not in REGION_ORDER]  # keep any extras
    adata.obs[col] = pd.Categorical(vals, categories=cats, ordered=True)
    adata.uns[f'{col}_colors'] = [REGION_PALETTE.get(c, '#b8b8b8') for c in cats]
    return adata


# Precomputed QC metrics + propagated taxonomy carried from the UNFILTERED object
# for every cell (these drive the sciduck class-specific QC in the notebook).
# `*_propagated` labels exist even for QC-failed cells, enabling per-class filtering.
QC_CARRY_COLS = [
    'doublet_score', 'solo_doublet', 'percent_ribo', 'log.gene.counts.0',
    'total_counts', 'total_genes',
    'Class_propagated', 'Subclass_propagated', 'Group_propagated', 'leiden',
    # batch / study metadata used to integrate the workshop scVI model (01b).
    'batch', 'study', 'donor_name',
]

# ── Pre-filter scVI + UMAP (built by 01b_scvi_umap_prefilter.py) ───────────────
# A fresh scVI model is trained on the WHOLE subsample (QC-passed + QC-failed)
# so the workshop object ships with an integrated UMAP in which students can see
# where their filtered vs. unfiltered nuclei land *before* they do any filtering.
SCVI_BATCH_KEY = 'species'                 # primary batch variable to integrate over
SCVI_CATEGORICAL_COVS = ['study']          # extra categorical covariates
SCVI_CONTINUOUS_COVS = ['log_umi_counts']  # depth covariate (log10 total counts)
SCVI_N_LATENT = 30
SCVI_N_LAYERS = 2
SCVI_N_HIDDEN = 128
SCVI_MAX_EPOCHS = 200
SCVI_UMAP_N_NEIGHBORS = 30
# obsm keys written by 01b (the workshop's own, computed on the full subsample).
SCVI_LATENT_KEY = 'X_scVI'
SCVI_UMAP_KEY = 'X_umap_prefilter'
# Trained scVI model is saved here so the workshop UMAP/latent can be reproduced
# (or new query cells projected) without retraining.
SCVI_MODEL_DIR = os.path.join(RESULTS_DIR, 'SpC_workshop_scvi_model')

# ── Session-2 clean object (built by 02b_build_session2_clean.py) ──────────────
# A CLEAN, atlas-filtered object (keep only nuclei the published atlas passed) on
# which we retrain scVI from scratch and recompute a UMAP. This is the starting
# point for Session 2 (mapping / annotation), so it carries NO QC-failed nuclei
# and a freshly integrated latent space. Both a full processed copy and a
# cellxgene-safe copy are written to /results.
SESSION2_OUT = os.path.join(RESULTS_DIR, 'SpC_workshop_snRNA_session2_clean.h5ad')
SESSION2_CELLXGENE_OUT = os.path.join(
    RESULTS_DIR, 'SpC_workshop_snRNA_session2_clean_cellxgene.h5ad')
SESSION2_SCVI_MODEL_DIR = os.path.join(RESULTS_DIR, 'SpC_workshop_session2_scvi_model')
# Latent / UMAP keys written into the clean object.
SESSION2_LATENT_KEY = 'X_scVI'
SESSION2_UMAP_KEY = 'X_umap'


def set_all_seeds(seed: int = SEED) -> None:
    """Seed every RNG we touch so all downstream results are deterministic.

    We deliberately do *not* call ``torch.use_deterministic_algorithms(True)``:
    it requires ``CUBLAS_WORKSPACE_CONFIG`` to be exported *before* the process
    starts (setting it at runtime does not reach cuBLAS), and without that the
    GPU matmuls scVI relies on raise ``CUBLAS_STATUS_NOT_INITIALIZED``. Seeding
    python/numpy/scanpy/torch already makes our pipeline reproducible in practice.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import scanpy as sc
        sc.settings.seed = seed
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Stable, fast matmul path on Tensor-Core GPUs (e.g. NVIDIA L4).
            torch.set_float32_matmul_precision('high')
    except Exception:
        pass
