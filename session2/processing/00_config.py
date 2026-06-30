"""Shared configuration for Lipari Genomics Workshop 2026 — Session 2.

Session 2 is the **"map to whole brain"** interactive section (workshop_prep.txt
Part 2B). It implements a *reciprocal* mapping between the spinal-cord (SpC)
snRNA taxonomy from Session 1 and the Allen mouse whole-brain (AIT21) taxonomy,
then projects the result onto the ABC mouse-brain MERFISH **spatial** atlas:

    FORWARD   SpC query cell      ->  mouse WB *subclass*      (which WB types did we capture?)
    REVERSE   mouse WB subclass   ->  our consensus SpC *type* (which WB types are spinal-like?)
    SPATIAL   detected subclasses ->  where in the mouse brain those types live.

Centralizes file paths, the global ``SEED``, and ``set_all_seeds`` so every
processing script and the student notebook are byte-for-byte reproducible.
Import with::

    import sys; sys.path.append('/code/lipari_genomics_workshop_2026/session2/processing')
    from importlib import import_module
    cfg = import_module('00_config')
    cfg.set_all_seeds()
"""
import os
import random

import numpy as np

# ── Global seed ────────────────────────────────────────────────────────────────
SEED = 0

# ── Query (from Session 1) ─────────────────────────────────────────────────────
# The multi-species SpC snRNA workshop object built by session1/processing.
# Raw counts in .X, human-ortho gene symbols in var_names, V2 taxonomy in obs.
QUERY_H5AD = '/results/SpC_workshop_snRNA.h5ad'
# obs column carrying our spinal-cord cell-type label (used to summarise the
# forward mapping per spinal type).
QUERY_GROUP_KEY = 'Group_V2'
QUERY_SUBCLASS_KEY = 'Subclass_V2'

# ── FORWARD reference: Allen mouse whole-brain, SUBCLASS level ──────────────────
# Pre-built cell_type_mapper reference shipped with the spatial tutorial bundle
# (precompute_stats.h5 + reference_markers.h5; hierarchy class_label→subclass_label).
TUTORIAL_DIR = '/data/mouse_wb_spatial_tutorial'
WB_SUBCLASS_REF = os.path.join(
    TUTORIAL_DIR, 'reference', 'subclass_mapping_reference')
# Taxonomy levels in the forward result (hann_results.json).
WB_SUBCLASS_LEVEL = 'subclass_label'
WB_CLASS_LEVEL = 'class_label'

# ── REVERSE reference: OUR consensus spinal-cord taxonomy ───────────────────────
# A NEW cell_type_mapper reference built from the AIBS consensus SpC taxonomy
# (hierarchy Class→Subclass→Group→consensus_cluster). 02_build_spc_reference.sh
# (re)builds it from the source h5ad; a pre-built copy already lives on /scratch.
SPC_CONSENSUS_H5AD = (
    '/data/SpinalCord/manuscript/RNA/'
    'AIBS_SpC_consensus_taxonomy_harmonized_AIT-pre-print.h5ad')
SPC_CONSENSUS_HIERARCHY = ['Class', 'Subclass', 'Group', 'consensus_cluster']
SPC_CONSENSUS_REF = '/scratch/SpC_consensus_ref'
# Taxonomy levels in the reverse result we care about (readable node names).
SPC_GROUP_LEVEL = 'Group'
SPC_SUBCLASS_LEVEL = 'Subclass'
SPC_CLASS_LEVEL = 'Class'

# ── Mouse WB subclass mean expression (the REVERSE query) ───────────────────────
# One log-normalized mean profile per AIT21 subclass (338 × 32285 genes). Mapping
# these means onto the SpC reference asks "which spinal type does each mouse WB
# subclass resemble?" — the reverse arm, fast enough to run live. Map with
# normalization='log2CPM' (these are already log-normalized, matching how the
# bundled full 4M-cell AIT21→SpC map under …_ABC_MAPPING was produced).
SUBCLASS_MEANS_CSV = (
    '/data/hmba_xs_v1/mouse_wb/AIT21.all.freeze.230815_subclassmeans.csv')
# Full reference taxonomy h5ad (~224 GB, NOT shipped) — only needed to rebuild the
# whole-cell AIT21→SpC mapping; the bundled means + ABC_MAPPING cover the workshop.
AIT21_H5AD = (
    '/data/hmba_xs_v1/mouse_wb/AIT21.all.freeze.230815_humanorthos.h5ad')

# ── Spatial atlas (ABC mouse-brain MERFISH; from the tutorial bundle) ───────────
# Small clean 16-section object (taxonomy + CCF region annotations, ~1.2 M cells).
SPATIAL_ATLAS = os.path.join(
    TUTORIAL_DIR, 'reference', 'spatial_atlas',
    'C57BL6J-638850-16good-sections-raw-meta.h5ad')
# Swap to the full 59-section atlas for whole-brain coverage:
SPATIAL_ATLAS_FULL = os.path.join(
    TUTORIAL_DIR, 'reference', 'spatial_atlas',
    'C57BL6J-638850-raw-meta.h5ad')
# Column in the atlas .obs carrying the mouse-WB subclass label.
ATLAS_SUBCLASS_KEY = 'subclass'

# ── Outputs (small, notebook-ready; live in /results next to Session 1's) ───────
RESULTS_DIR = '/results'
# FORWARD: hann_results.json (one record per SpC query cell).
FWD_MAPPING_DIR = os.path.join(RESULTS_DIR, 'SpC_workshop_WB_SUBCLASS_MAPPING')
FWD_RESULTS_JSON = os.path.join(FWD_MAPPING_DIR, 'hann_results.json')
# REVERSE query + result.
WB_MEANS_H5AD = os.path.join(RESULTS_DIR, 'wb_subclass_means.h5ad')
REV_MAPPING_DIR = os.path.join(RESULTS_DIR, 'WB_SUBCLASS_to_SpC_MAPPING')
REV_RESULTS_JSON = os.path.join(REV_MAPPING_DIR, 'hann_results.json')
# Distilled, notebook-ready tables (written by 04_build_reciprocal_artifacts.py).
FORWARD_SUMMARY_CSV = os.path.join(RESULTS_DIR, 'reciprocal_forward_summary.csv')
REVERSE_SUMMARY_CSV = os.path.join(RESULTS_DIR, 'reciprocal_reverse_summary.csv')
RECIPROCAL_CSV = os.path.join(RESULTS_DIR, 'reciprocal_best_hits.csv')

# ── SUPERTYPE-level reverse arm (subsampled; built by 05-07) ────────────────────
# Approximate mouse-WB supertype mean profiles (subsampled from AIT21) + the
# supertype->subclass->class lookup, then their reverse mapping onto the SpC ref.
WB_SUPERTYPE_MEANS_H5AD = os.path.join(RESULTS_DIR, 'wb_supertype_means.h5ad')
WB_SUBSAMPLED_ABC_H5AD = os.path.join(RESULTS_DIR, 'wb_subsampled_ABC.h5ad')
WB_SUPERTYPE_TO_SUBCLASS_CSV = os.path.join(
    RESULTS_DIR, 'wb_supertype_to_subclass.csv')
REV_SUPERTYPE_MAPPING_DIR = os.path.join(
    RESULTS_DIR, 'WB_SUPERTYPE_to_SpC_MAPPING')
REV_SUPERTYPE_RESULTS_JSON = os.path.join(
    REV_SUPERTYPE_MAPPING_DIR, 'hann_results.json')
# Overlap-coefficient reciprocity tables (written by 07).
SUBCLASS_OVERLAP_CSV = os.path.join(RESULTS_DIR, 'reciprocal_subclass_overlap.csv')
RECIPROCAL_SUPERTYPE_CSV = os.path.join(
    RESULTS_DIR, 'reciprocal_supertype_hits.csv')
# Overlap-coefficient threshold above which a pair counts as "reciprocally mapped".
OVERLAP_MIN = 0.20
ATLAS_SUPERTYPE_KEY = 'supertype'

# Confidence floor used when summarising mappings (drop very low-prob calls).
MIN_PROB = 0.0


def set_all_seeds(seed: int = SEED) -> None:
    """Seed python/numpy/scanpy (and torch if present) for reproducibility."""
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
    except Exception:
        pass
