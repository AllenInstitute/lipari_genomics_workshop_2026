# Session 1 — Spinal cord snRNA QC, clustering & visualization (Interactive Part 1)

Materials for the first interactive session (50 min): single-nucleus spinal-cord
QC, clustering, and visualization, plus an intro to **cellxgene** and the **ABC
whole-brain atlas**.

## Layout
```
session1/
├── processing/                       # heavy, run-once data prep (not run live)
│   ├── 00_config.py                  # shared paths + global SEED + set_all_seeds()
│   ├── 01_build_snrna_subsample.py   # -> /results/SpC_workshop_snRNA.h5ad
│   ├── 01b_scvi_umap_prefilter.py    # trains scVI on the full (pre-filter) subset;
│   │                                 #   adds X_scVI + X_umap_prefilter to the h5ad
│   │                                 #   and saves /results/SpC_workshop_scvi_model/
│   └── 02_build_spatial_example.py   # -> /results/SpC_workshop_spatial_example.h5ad
│   ├── _taxonomy_colors.py           # carries the curated V2 palette from the atlas
│   └── make_safe_h5ad.py             # sanitize any h5ad for cellxgene (drop NaNs, etc.)
└── notebooks/
    └── session1_qc_clustering_visualization.ipynb   # student-facing notebook
```

## Workshop data
Built by the processing scripts and written to `/results/` (where cellxgene reads):

| File | What it is |
|------|------------|
| `SpC_workshop_snRNA.h5ad` | Multi-species snRNA subsample: ≤100 nuclei per `Group_V2` per species that **passed** QC, **plus** QC-failed nuclei making up **40%** of the object (`obs['qc_status']`). Raw counts in `X`. Two embeddings computed on the **full (pre-filter)** set so students can see where filtered vs. unfiltered nuclei land: `obsm['X_scVI']` / `obsm['X_umap_prefilter']` (trained on this subset by `01b`) and `obsm['X_scVI_atlas']` / `obsm['X_umap_atlas']` (published atlas, for comparison). Carries precomputed QC inputs (`doublet_score`, `solo_doublet`, `percent_ribo`, `log.gene.counts.0`) and propagated taxonomy (`Class_propagated`, `Subclass_propagated`, `Group_propagated`, `leiden`) for **every** nucleus, enabling class-specific QC. |
| `SpC_workshop_scvi_model/` | The trained scVI model behind `X_scVI` / `X_umap_prefilter`, saved so the embedding is reproducible and new cells can be projected without a GPU. |
| `SpC_workshop_spatial_example.h5ad` | One representative macaque MERFISH section with `obsm['spatial']` and V2 / Rexed-lamina annotations. |

### Rebuild
```bash
cd processing
python 01_build_snrna_subsample.py    # build the subsample h5ad
python 01b_scvi_umap_prefilter.py     # GPU: train scVI, add pre-filter embeddings + save model
python 02_build_spatial_example.py    # build the spatial example
```
`01b` needs a GPU. NOTE: it imports `torch`/`scvi` **before** numpy/anndata/scanpy
on purpose — importing a numpy/scipy BLAS first shadows the library torch needs and
scVI training dies with `CUBLAS_STATUS_NOT_INITIALIZED`.

### Make an h5ad cellxgene-safe
`make_safe_h5ad.py` coerces an object into the cellxgene schema — numeric/categorical
`obs`/`var` only (NaNs dropped or stringified), bounded category counts, float32 `X`,
no stray `raw`/`obsm`/`varm` DataFrames, and only `*_colors` palettes aligned to their
categorical column kept in `uns`. The `/uns` prune step opens the input **in place**,
so copy to scratch first:
```bash
cp /results/SpC_workshop_snRNA.h5ad /scratch/in.h5ad
python make_safe_h5ad.py /scratch/in.h5ad /scratch/SpC_workshop_snRNA_cellxgene.h5ad
```

## Reproducibility
Every script and the notebook fix all RNG seeds via `SEED = 0`
(`set_all_seeds()` in `00_config.py`): python `random`, numpy, scanpy, torch.
Students' clustering/UMAP results will match exactly.

## Notebook flow
Load → inspect QC metrics → **class-specific QC with `sciduck`** (students set
per-class gene-count bounds + doublet/ribo limits and see what is removed, compared
against the atlas decision) → normalize/HVG → PCA/neighbors/Leiden/UMAP → compare
clusters to `Subclass_V2` → marker genes → export for cellxgene → ABC Atlas
exercise. Each code cell is preceded by a short markdown cell describing the step
(for attendees new to python).

The QC mirrors the original atlas notebook's `sciduck` strategy: different cell
classes (Non-Neurons vs neurons vs Motor Neurons) get **different** gene-count
thresholds, plus group-level (per-cluster) doublet constraints.

## Source notebooks this is distilled from
- QC: `/code/HMBA_Genomics/SpinalCord/xspecies/integration/SpC_External_QC.ipynb`
- Spatial: `/code/HMBA_Genomics/SpinalCord/Analysis/review_figures/notebooks/03_figure2_plot_panels.ipynb`
