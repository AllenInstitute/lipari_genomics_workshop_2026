"""Generate the Session-1 student notebook with markdown explanations before
every code cell and all RNG seeds fixed. Run once to (re)write the .ipynb."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


# ---- Title --------------------------------------------------------------------
md("""# Session 1 - Spinal cord snRNA-seq: QC, clustering & visualization

**Lipari Genomics Workshop 2026 - Interactive Part 1 (50 min)**

In this notebook you will work with a small, multi-species subsample of the HMBA
spinal-cord single-nucleus RNA-seq atlas and:

1. Inspect the dataset and compute **quality-control (QC)** metrics.
2. **Filter** low-quality nuclei using thresholds *you* choose.
3. **Normalize**, reduce dimensionality, **cluster** (Leiden), and embed (UMAP).
4. Compare your clusters to the reference **Group_V2 / Subclass_V2** cell-type labels.
5. Look at **marker genes** and export an object for **cellxgene**.

> Every random seed is fixed, so your results will match everyone else's exactly.

The data: ~100 nuclei per cell-type Group per species that *passed* QC, plus a
batch of nuclei that were *filtered out* of the published atlas - your job in the
QC section is to find them.""")

# ---- 0. Setup -----------------------------------------------------------------
md("""## 0. Setup

Import libraries and **fix all random seeds** (python, numpy, scanpy) so the
clustering and UMAP below are byte-for-byte reproducible across machines.""")
code("""import os, random
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
sc.settings.seed = SEED
sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=90, frameon=False, figsize=(6, 6))
print('scanpy', sc.__version__)""")

# ---- 1. Load ------------------------------------------------------------------
md("""## 1. Load the workshop dataset

Read the pre-built subsample. `X` holds **raw UMI counts**. Two integrated
embeddings are carried in `obsm`, both computed *before* any QC filtering so they
cover **every** nucleus:
- `X_scVI` / `X_umap_prefilter` - a scVI model trained on **this** workshop subset
  (QC-passed **and** QC-failed nuclei). We use its UMAP below to *see where the
  low-quality nuclei sit* before we remove them.
- `X_scVI_atlas` / `X_umap_atlas` - the published full-atlas embeddings (carried
  for comparison).""")
code("""adata = sc.read_h5ad('/results/SpC_workshop_snRNA.h5ad')
adata""")

md("""Look at the cell metadata (`obs`). Key columns:
- `qc_status` / `keeper_cells` - whether the published atlas **kept** or **filtered
  out** each nucleus (our ground truth for the QC exercise).
- `Class_V2` / `Subclass_V2` - the **reference cell-type taxonomy** (the labels we
  will compare our clusters to). These are only defined for nuclei that **passed**
  QC, so the filtered-out cells are `NaN` here.
- `class_coarse` - a *coarse* cell-class label (Non-Neurons, GABAergic,
  Glutamatergic, Motor Neurons, Cholinergic) assigned to **every** nucleus,
  *including* the ones that failed QC. Because it exists for all cells (unlike
  `Class_V2`), we use it - and only it - to apply **different QC thresholds to
  different cell classes** in the next section.
- precomputed QC metrics: `doublet_score`, `solo_doublet` (doublet probabilities),
  `percent_ribo`, `log.gene.counts.0` (= log10 of genes detected).
- `species` - the donor species.

We rename the carried atlas clustering to `atlas_leiden` (we will compute our own
`leiden` later), rename the all-nuclei propagated class to `class_coarse` (used for
QC only), and keep an `atlas_qc` copy of the ground-truth decision.""")
code("""adata.obs = adata.obs.rename(columns={'leiden': 'atlas_leiden',
                                      'Class_propagated': 'class_coarse'})
adata.obs['atlas_qc'] = adata.obs['qc_status']
print(adata.obs['qc_status'].value_counts())
print()
print(adata.obs['species'].value_counts())
adata.obs.head()""")

# ---- 1b. Get oriented ---------------------------------------------------------
md("""### Get oriented: what is actually in this object?

Before any analysis, it helps to *look* at the data. An `AnnData` object bundles
several aligned pieces:
- **`X`** - the count matrix (nuclei x genes). Each row is a nucleus, each column a
  gene.
- **`obs`** - a table of **per-nucleus** metadata (one row per nucleus).
- **`var`** - a table of **per-gene** metadata (one row per gene).
- **`obsm`** - per-nucleus matrices such as the embeddings (`X_scVI`, `X_umap_*`).

First, print the basic shape and the full list of `obs` / `var` columns so you know
what metadata is available to colour and filter by.""")
code("""print(f'{adata.n_obs:,} nuclei  x  {adata.n_vars:,} genes\\n')
print('obs columns ({}):'.format(adata.obs.shape[1]))
print(list(adata.obs.columns))
print('\\nvar columns ({}):'.format(adata.var.shape[1]))
print(list(adata.var.columns))
print('\\nobsm (embeddings):', list(adata.obsm.keys()))""")

md("""Now peek at the first few rows of each metadata table - this is the quickest
way to see what the values look like.""")
code("""from IPython.display import display
print('obs (per-nucleus metadata):')
display(adata.obs.head())
print('var (per-gene metadata):')
display(adata.var.head())""")

md("""Finally, get a feel for the **count depth** of the data. `X` holds raw UMI
counts, which span several orders of magnitude across nuclei, so we plot the
distribution of **log10(total counts per nucleus)**. A healthy snRNA-seq library
usually shows a single broad peak; a long low-count tail on the left is where
empty/low-quality droplets live - the nuclei QC will target.""")
code("""total_counts = np.asarray(adata.X.sum(1)).ravel()
adata.obs['log10_total_counts'] = np.log10(total_counts + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].hist(adata.obs['log10_total_counts'], bins=80, color='#2c7fb8')
axes[0].set_xlabel('log10(total UMI counts per nucleus)')
axes[0].set_ylabel('number of nuclei')
axes[0].set_title('count depth across all nuclei')

genes_per_cell = np.asarray((adata.X > 0).sum(1)).ravel()
axes[1].hist(np.log10(genes_per_cell + 1), bins=80, color='#41ab5d')
axes[1].set_xlabel('log10(genes detected per nucleus)')
axes[1].set_ylabel('number of nuclei')
axes[1].set_title('genes detected across all nuclei')
plt.tight_layout(); plt.show()""")

md("""**Sparsity.** Single-nucleus count matrices are mostly zeros - any given gene
is detected in only a small fraction of nuclei. This *sparsity* is a defining
feature of the data and shapes every downstream method (it is why we store `X` as a
sparse matrix and why we normalize and reduce dimensions before clustering).""")
code("""import scipy.sparse as sp
n_total = adata.n_obs * adata.n_vars
n_nonzero = adata.X.nnz if sp.issparse(adata.X) else int(np.count_nonzero(adata.X))
pct_zero = 100 * (1 - n_nonzero / n_total)
print(f'{pct_zero:.1f}% of the {n_total:,} entries in X are zero '
      f'({n_nonzero:,} non-zero)')

# distribution of the non-zero count values (log y-axis): most are 1-2 UMIs
nz_vals = adata.X.data if sp.issparse(adata.X) else adata.X[adata.X > 0]
plt.figure(figsize=(7, 4))
plt.hist(nz_vals, bins=np.arange(0.5, 30.5, 1), color='#756bb1')
plt.yscale('log')
plt.xlabel('UMI count (non-zero entries only)')
plt.ylabel('number of entries (log scale)')
plt.title('most detected genes have just 1-2 UMIs per nucleus')
plt.tight_layout(); plt.show()""")

md("""**What dominates the reads?** Because we sequenced *nuclei*, a handful of long,
intron-rich and nuclear-retained transcripts (e.g. `MALAT1`) soak up a large share
of every library. `highest_expr_genes` shows the fraction of total counts taken by
the top genes - a useful sanity check that also explains why nuclear data looks
different from whole-cell data.""")
code("""sc.pl.highest_expr_genes(adata, n_top=20)""")

# ---- 2. QC metrics ------------------------------------------------------------
md("""## 2. Inspect the QC metrics

The object already carries the QC metrics the atlas used (computing some of them,
like the SOLO doublet probabilities, needs a GPU and several minutes, so they were
precomputed upstream):
- **`log.gene.counts.0`** - log10 of the number of genes detected per nucleus.
- **`percent_ribo`** - fraction of counts from ribosomal genes.
- **`doublet_score`** and **`solo_doublet`** - two complementary doublet scores.

We recompute the two cheap ones (`percent_ribo`, `log.gene.counts.0`) from the raw
counts so you can see exactly how they are defined, then summarize all four.""")
code("""ribo_genes = adata.var_names.str.startswith(('RPS', 'RPL'))
adata.obs['percent_ribo'] = (np.asarray(adata[:, ribo_genes].X.sum(1)).ravel() /
                             np.asarray(adata.X.sum(1)).ravel())
adata.obs['log.gene.counts.0'] = np.log10(np.asarray((adata.X > 0).sum(1)).ravel() + 1)
adata.obs[['log.gene.counts.0', 'percent_ribo', 'doublet_score',
           'solo_doublet']].describe().round(3)""")

md("""### Different cell classes have different "good" ranges

The key idea behind the atlas QC: **a healthy non-neuron and a healthy neuron have
very different gene counts**, so a single global threshold is wrong. The violin
below shows `log.gene.counts.0` per `class_coarse` (our all-nuclei coarse class) -
neurons (and especially motor neurons) detect more genes than non-neurons. We will
therefore set **class-specific** gene-count bounds.""")
code("""sc.pl.violin(adata, 'log.gene.counts.0', groupby='class_coarse',
             stripplot=False, rotation=30)""")

md("""The doublet scores and ribosomal fraction, split by the atlas decision
(`atlas_qc`), show where the filtered-out nuclei concentrate.""")
code("""def qc_histograms(df):
    metrics = ['doublet_score', 'solo_doublet', 'percent_ribo']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, m in zip(axes, metrics):
        for status, color in [('passed_qc', '#2c7fb8'), ('filtered_out', '#d95f0e')]:
            ax.hist(df.loc[df['atlas_qc'] == status, m], bins=60, alpha=0.6,
                    label=status, color=color)
        ax.set_xlabel(m); ax.set_ylabel('nuclei'); ax.legend()
    plt.tight_layout(); plt.show()

qc_histograms(adata.obs)""")

md("""### Where do the low-quality nuclei sit on the UMAP?

Before we remove anything, colour the **pre-filter UMAP** (`X_umap_prefilter`,
computed on *all* nuclei) by the atlas QC decision, by coarse cell class
(`class_coarse`, which - unlike `Class_V2` - is defined for the filtered cells too),
and by the per-nucleus **doublet score**. Filtered-out nuclei (orange) tend to form
their own low-quality territory and fringes rather than mixing evenly into the
healthy populations, and high doublet scores often light up those same regions -
that is exactly what QC removes. Keep this picture in mind: after you set your own
thresholds you will be cutting these regions.""")
code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sc.pl.embedding(adata, basis='X_umap_prefilter', color='atlas_qc', ax=axes[0],
                show=False, title='atlas QC decision (pre-filter)', size=8)
sc.pl.embedding(adata, basis='X_umap_prefilter', color='class_coarse',
                ax=axes[1], show=False, title='coarse cell class', size=8)
sc.pl.embedding(adata, basis='X_umap_prefilter', color='doublet_score', ax=axes[2],
                show=False, title='doublet score', size=8, cmap='viridis')
plt.tight_layout(); plt.show()""")

# ---- 3. Class-specific QC with sciduck ----------------------------------------
md("""## 3. Class-specific QC with `sciduck`

We reproduce the atlas QC strategy with **`sciduck`**, which lets us register a set
of **constraints** and then apply them all at once. Constraints can be:
- **range** constraints on a metric (`gt`/`lt`), optionally restricted to a
  **subset** of cells (e.g. only `Non-Neurons`), and
- **group-level** constraints that act on a per-cluster summary (e.g. drop whole
  clusters whose *mean* doublet score is too high).

A nucleus is kept only if it satisfies **all** constraints.""")
code("""!pip install -q sciduck""")
code("""import sciduck as sd""")

md("""### Set your thresholds

**These presets start wide open - they keep essentially every nucleus, including
the bad ones.** Your job is to *tighten* them. `GENE_BOUNDS` gives the allowed
`log.gene.counts.0` **(low, high)** range **per cell class** (non-neurons typically
need fewer genes than neurons, and motor neurons the most). The doublet/ribo
cut-offs are global. Edit the numbers, re-run, and watch the diagnostics below
until your filter matches the atlas decision.""")
code("""# >>> EDIT THESE THRESHOLDS <<<
# These defaults are deliberately permissive - they let basically everything
# through. Narrow them until you are only keeping good-quality nuclei.
# allowed log10(genes detected) range, per cell class
GENE_BOUNDS = {
    'Non-Neurons':   (0.0, 10.0),   # wide open - tighten me!
    'neurons':       (0.0, 10.0),   # GABAergic / Glutamatergic / Cholinergic
    'Motor Neurons': (0.0, 10.0),   # motor neurons
}
MAX_DOUBLET_SCORE = 1.0   # drop nuclei above this doublet score (1.0 = keep all)
MAX_SOLO_DOUBLET  = 1.0   # drop nuclei above this SOLO doublet probability
MAX_RIBO          = 1.0   # drop nuclei above this ribosomal fraction
# group-level (per atlas cluster) doublet limits - also wide open to start
MAX_GROUP_DOUBLET = 1.0   # drop whole clusters whose MEAN doublet score exceeds this
MAX_GROUP_SOLO    = 1.0   # drop whole clusters whose MEAN SOLO doublet exceeds this""")

md("""Preview the per-class gene-count bounds before applying them: each panel shows
one class's `log.gene.counts.0` distribution with your low/high lines and the green
**keep** band. Re-run after editing `GENE_BOUNDS`.""")
code("""def preview_gene_bounds(bounds):
    classes = ['Non-Neurons', 'GABAergic', 'Glutamatergic', 'Cholinergic', 'Motor Neurons']
    classes = [c for c in classes if c in adata.obs['class_coarse'].unique()]
    fig, axes = plt.subplots(1, len(classes), figsize=(3.2 * len(classes), 3.2),
                             sharex=True)
    for ax, cls in zip(np.atleast_1d(axes), classes):
        key = cls if cls in bounds else 'neurons'
        lo, hi = bounds[key]
        vals = adata.obs.loc[adata.obs['class_coarse'] == cls, 'log.gene.counts.0']
        ax.hist(vals, bins=40, color='#7570b3', alpha=0.7)
        ax.axvline(lo, color='k', ls='--'); ax.axvline(hi, color='k', ls='--')
        ax.axvspan(lo, hi, color='green', alpha=0.08)
        ax.set_title(f'{cls}  [{lo}, {hi}]', fontsize=9)
        ax.set_xlabel('log.gene.counts.0')
    plt.tight_layout(); plt.show()

preview_gene_bounds(GENE_BOUNDS)""")

md("""### Register and apply the constraints

We register the global doublet/ribo limits, the class-specific gene-count ranges,
and two **group-level** constraints that can remove whole atlas clusters dominated
by doublets. With the wide-open presets nothing is removed yet - `apply_constraints`
writes a fresh `keeper_cells` column that currently keeps (almost) everything.""")
code("""for k in ['qc_constraints', 'qc_filtered']:
    adata.uns.pop(k, None)

classes = list(adata.obs['class_coarse'].cat.categories)
neuron_classes = [c for c in classes if c not in ('Non-Neurons', 'Motor Neurons')]

# global doublet / ribosomal limits
sd.basic_qc.add_range_constraint(adata, 'percent_ribo', lt=MAX_RIBO)
sd.basic_qc.add_range_constraint(adata, 'doublet_score', lt=MAX_DOUBLET_SCORE)
sd.basic_qc.add_range_constraint(adata, 'solo_doublet', lt=MAX_SOLO_DOUBLET)

# class-specific gene-count ranges
sd.basic_qc.add_range_constraint(adata, 'log.gene.counts.0',
    gt=GENE_BOUNDS['Non-Neurons'][0], lt=GENE_BOUNDS['Non-Neurons'][1],
    subset='class_coarse', subset_values=['Non-Neurons'])
sd.basic_qc.add_range_constraint(adata, 'log.gene.counts.0',
    gt=GENE_BOUNDS['Motor Neurons'][0], lt=GENE_BOUNDS['Motor Neurons'][1],
    subset='class_coarse', subset_values=['Motor Neurons'])
sd.basic_qc.add_range_constraint(adata, 'log.gene.counts.0',
    gt=GENE_BOUNDS['neurons'][0], lt=GENE_BOUNDS['neurons'][1],
    subset='class_coarse', subset_values=neuron_classes)

# group-level: drop atlas clusters whose mean doublet signal is high
sd.basic_qc.add_group_level_constraint(adata, 'doublet_score', groupby='atlas_leiden', lt=MAX_GROUP_DOUBLET)
sd.basic_qc.add_group_level_constraint(adata, 'solo_doublet', groupby='atlas_leiden', lt=MAX_GROUP_SOLO)

sd.basic_qc.apply_constraints(adata)
print(adata.obs['keeper_cells'].value_counts())""")

md("""### How well did your QC recover the atlas decision?

Cross-tabulate your `keeper_cells` against the atlas `atlas_qc`, and report
precision/recall. Each registered constraint also records which nuclei it removed
in `adata.uns['qc_filtered']`.""")
code("""keep = adata.obs['keeper_cells']
atlas_keep = adata.obs['atlas_qc'] == 'passed_qc'
precision = (keep & atlas_keep).sum() / max(keep.sum(), 1)
recall = (keep & atlas_keep).sum() / max(atlas_keep.sum(), 1)
print(pd.crosstab(keep, adata.obs['atlas_qc']))
print(f'precision {precision:.2f}, recall {recall:.2f} vs atlas')
print('nuclei removed by each constraint:')
for metric, d in adata.uns['qc_filtered'].items():
    print(f'  {metric}: {sum(len(ids) for ids in d.values())}')""")

md("""### See exactly what *you* are about to remove

Plot your own decision on the pre-filter UMAP: nuclei you marked to **keep** vs
**remove**. Compare this with the atlas-decision UMAP above - tightening or
loosening the thresholds in the cell above moves the boundary of the orange
region. Re-run the constraints with different numbers and watch this plot change.""")
code("""adata.obs['your_qc'] = np.where(adata.obs['keeper_cells'], 'keep', 'remove')
sc.pl.embedding(adata, basis='X_umap_prefilter', color='your_qc',
                title='your QC decision (pre-filter UMAP)',
                palette={'keep': '#2c7fb8', 'remove': '#d95f0e'}, size=8)""")

md("""### Keep the clean nuclei

Subset to your `keeper_cells` for the rest of the analysis.""")
code("""# remember the pre-filter UMAP + both QC decisions for the final comparison
qc_compare = adata.obs[['your_qc', 'atlas_qc']].copy()
qc_compare[['umap1', 'umap2']] = adata.obsm['X_umap_prefilter']

adata = adata[adata.obs['keeper_cells']].copy()
print(f'{adata.n_obs:,} nuclei retained')""")


# ---- 4. Normalize + HVG -------------------------------------------------------
md("""## 4. Normalize, log-transform, select highly variable genes

Library-size normalize to 10,000 counts/cell, `log1p` transform, and select the
top 2,000 **highly variable genes (HVGs)** that drive the clustering. We keep the
raw counts in a layer so nothing is lost.""")
code("""adata.layers['counts'] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat')
adata.raw = adata
print('HVGs:', int(adata.var['highly_variable'].sum()))""")

# ---- 5. scVI latent / neighbors / clustering / UMAP ---------------------------
md("""## 5. Dimensionality reduction & clustering

We cluster on the **batch-corrected scVI latent space** (`X_scVI`, carried in the
object) rather than on PCA of the highly variable genes. scVI was trained across
species/donors, so this latent space has the **batch effects removed** - clusters
will reflect cell identity instead of which species or donor a nucleus came from.
We build a k-nearest-neighbor graph directly on `X_scVI`, cluster with **Leiden**,
and compute a **UMAP** - all with a fixed `random_state` for reproducibility. The
`resolution` controls granularity: we start at a deliberately **high** value
(`15`), which **over-clusters** the data into many small groups. Try lowering it
and watch the cluster count and the crosstabs below change.""")
code("""sc.pp.neighbors(adata, use_rep='X_scVI', n_neighbors=15, random_state=SEED)
sc.tl.leiden(adata, resolution=15, random_state=SEED,
             flavor='igraph', n_iterations=2, directed=False)
sc.tl.umap(adata, random_state=SEED)
print(adata.obs['leiden'].value_counts())""")

md("""Plot the UMAP that **you** just computed from the scVI latent. First, colour it
by your **Leiden clusters** on their own - with so many clusters at `resolution=15`,
we use the high-contrast `godsnot_102` palette and hide the (very long) legend.""")
code("""sc.pl.umap(adata, color='leiden', palette=sc.pl.palettes.godsnot_102,
           legend_loc=None, title='your Leiden clusters')""")

md("""Now colour the same UMAP by **species** and by the reference `Class_V2` and
`Subclass_V2` labels (these are blank/`NaN` for any nuclei without a reference
annotation).""")
code("""sc.pl.umap(adata, color=['species', 'Class_V2', 'Subclass_V2'],
           wspace=0.4, ncols=2)""")

md("""### Why integrate? scVI vs. a non-corrected PCA embedding

To see what batch correction bought us, build a quick UMAP the *naive* way - PCA on
the scaled highly variable genes, with **no** batch correction - and colour both by
species. The PCA embedding tends to split each cell type **by species**, while the
scVI embedding you clustered on mixes the species together.""")
code("""adata_hvg = adata[:, adata.var['highly_variable']].copy()
sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, n_comps=50, random_state=SEED)
sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=30, random_state=SEED)
sc.tl.umap(adata_hvg, random_state=SEED)
adata.obsm['X_umap_pca'] = adata_hvg.obsm['X_umap']

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sc.pl.embedding(adata, basis='X_umap_pca', color='species', ax=axes[0],
                show=False, title='PCA on HVGs (not integrated)')
sc.pl.umap(adata, color='species', ax=axes[1],
           show=False, title='scVI-integrated (used for clustering)')
plt.tight_layout(); plt.show()""")

# ---- 6. Cluster vs reference --------------------------------------------------
md("""## 6. How well do your clusters match the reference cell types?

Cross-tabulate your Leiden clusters against the reference taxonomy - first the
coarse `Class_V2`, then the finer `Subclass_V2`. A good clustering shows each
cluster dominated by a single reference class/subclass (each row summing to ~1
after normalizing).""")
code("""ct_class = pd.crosstab(adata.obs['leiden'], adata.obs['Class_V2'])
ct_class.div(ct_class.sum(1), axis=0).round(2)""")
code("""ct = pd.crosstab(adata.obs['leiden'], adata.obs['Subclass_V2'])
ct.div(ct.sum(1), axis=0).round(2)""")

# ---- 7. Marker genes ----------------------------------------------------------
md("""## 7. Marker genes

The atlas defines cell types largely by their **neurotransmitter identity**. We use
transmitter-pathway genes from the manuscript marker figures
(`04_marker_figures.ipynb`) - the enzymes and transporters that make a neuron
cholinergic, glutamatergic, or GABAergic/glycinergic, plus the motor-neuron
transcription factor `ISL1`:

| pathway | genes | marks |
|---|---|---|
| Cholinergic / motor neuron | `CHAT`, `ISL1` | acetylcholine synthesis; MN identity |
| Glutamatergic | `SLC17A6` | VGLUT2, excitatory |
| GABAergic / glycinergic | `GAD1`, `SLC6A5` | inhibitory |
""")
code("""# canonical neurotransmitter-identity panel (same genes as 04_marker_figures)
NT_MARKERS = {
    'Cholinergic / MN': ['CHAT', 'ISL1'],
    'Glutamatergic':    ['SLC17A6'],
    'GABA / glycine':   ['GAD1', 'SLC6A5'],
}
NT_MARKERS = {grp: [g for g in gs if g in adata.var_names]
              for grp, gs in NT_MARKERS.items()}""")

md("""Show the panel across the reference `Subclass_V2` taxonomy as a dotplot.
`standard_scale='var'` rescales **each gene to 0-1** (exactly like the per-gene
scaling in the manuscript heatmaps), so every subclass lights up its expected
transmitter genes - cholinergic subclasses on `CHAT`, glutamatergic on `SLC17A6`,
and so on.""")
code("""sc.pl.dotplot(adata, NT_MARKERS, groupby='Subclass_V2',
              standard_scale='var', cmap='viridis')""")

md("""The same panel collapsed to the coarse `Class_V2` level gives a clean,
textbook transmitter read-out.""")
code("""sc.pl.matrixplot(adata, NT_MARKERS, groupby='Class_V2',
                 standard_scale='var', cmap='viridis',
                 colorbar_title='scaled mean expr.')""")

md("""Finally, let the **data** nominate markers with no prior knowledge. Like the
manuscript marker figures (`04_marker_figures.ipynb`), we rank the genes that best
distinguish each reference `Supergroup_V2` cell type (Wilcoxon test) and show the
single top marker per group.""")
code("""adata_lab = adata[adata.obs['Supergroup_V2'].notna()].copy()
adata_lab.obs['Supergroup_V2'] = adata_lab.obs['Supergroup_V2'].cat.remove_unused_categories()
sc.tl.rank_genes_groups(adata_lab, 'Supergroup_V2', method='wilcoxon')
sc.pl.rank_genes_groups_dotplot(adata_lab, n_genes=1, standard_scale='var')""")

# ---- 8. Your filtering vs. the reference -------------------------------------
md("""## 8. Your filtering vs. the reference

Finally, compare **the nuclei you removed in this notebook** with **the nuclei that
were removed in the published, processed object**. Both are drawn on the same
pre-filter UMAP so you can see exactly where the two decisions agree and disagree.
If you tightened or loosened the thresholds in section 3, the left panel will shift
relative to the right one.""")
code("""fig, axes = plt.subplots(1, 2, figsize=(14, 6))
panels = [
    (axes[0], 'your_qc', 'your filtering (this notebook)',
     {'keep': '#2c7fb8', 'remove': '#d95f0e'}),
    (axes[1], 'atlas_qc', 'reference filtering (processed h5ad)',
     {'passed_qc': '#2c7fb8', 'filtered_out': '#d95f0e'}),
]
for ax, col, title, cmap_d in panels:
    for val, color in cmap_d.items():
        m = qc_compare[col] == val
        ax.scatter(qc_compare.loc[m, 'umap1'], qc_compare.loc[m, 'umap2'],
                   s=4, c=color, label=val, linewidths=0)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=4, loc='best', frameon=False)
plt.tight_layout(); plt.show()

# how often do the two decisions agree?
agree = ((qc_compare['your_qc'] == 'keep') ==
         (qc_compare['atlas_qc'] == 'passed_qc'))
print(f'agreement: {agree.mean():.1%} of {len(qc_compare):,} nuclei')
print(pd.crosstab(qc_compare['your_qc'], qc_compare['atlas_qc']))""")

# ---- 9. Export for cellxgene --------------------------------------------------
md("""## 9. Export for cellxgene

Write the processed object to `/results/` so it can be opened in **cellxgene** for
interactive exploration. (Your instructor will provide the cellxgene capsule link.)""")
code("""adata.write('/results/SpC_workshop_snRNA_session1_processed.h5ad')
print('Saved /results/SpC_workshop_snRNA_session1_processed.h5ad')""")

md("""### Make a cellxgene-safe copy

cellxgene is strict about its schema (numeric or categorical `obs`/`var` only, no
NaNs in numeric columns, only `*_colors` palettes in `uns`, float `X`). The
`make_safe_h5ad.py` utility in `../processing/` coerces an object into that shape.
We run it here on a throwaway copy of the object we just wrote (the script prunes
`uns` **in place**, so we never point it at a file we want to keep) and write the
cleaned result to `/results/`.""")
code("""import shutil, subprocess, sys, os

src    = '/results/SpC_workshop_snRNA_session1_processed.h5ad'
safe   = '/results/SpC_workshop_snRNA_session1_cellxgene.h5ad'
script = '../processing/make_safe_h5ad.py'

# work on a throwaway copy on the same volume as the output (the script prunes
# uns in place, and the object is large, so avoid small /tmp partitions)
tmp = safe + '.tmp.h5ad'
shutil.copy(src, tmp)
subprocess.run([sys.executable, script, tmp, safe], check=True)
os.remove(tmp)
print('Saved', safe)""")

# ---- 10. Spatial teaser -------------------------------------------------------
md("""## 10. A glimpse of spatial transcriptomics

snRNA-seq tells us **what** cell types exist, but not **where** they sit. Spatial
methods profile transcripts while preserving tissue coordinates. The workshop ships
a small spatial example (`/results/SpC_workshop_spatial_example.h5ad`) - the same
`Class_V2` taxonomy (and colours) mapped onto a spinal-cord section. Plotting the
cells in their real `(x, y)` positions reproduces the butterfly grey-matter shape,
with motor neurons in the ventral horn and dorsal-horn populations up top. We will
dig into this in Session 2.""")
code("""spatial = sc.read_h5ad('/results/SpC_workshop_spatial_example.h5ad')
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
sc.pl.embedding(spatial, basis='X_spatial', color='Class_V2', ax=axes[0],
                show=False, title='spatial section - Class_V2', size=25)
sc.pl.embedding(spatial, basis='X_spatial', color='Subclass_V2', ax=axes[1],
                show=False, title='spatial section - Subclass_V2', size=25)
for ax in axes:
    ax.set_aspect('equal')
plt.tight_layout(); plt.show()""")

# ---- 11. Interactive atlases --------------------------------------------------
md("""## 11. Explore the data interactively

**cellxgene** - the cellxgene-safe object you just saved
(`SpC_workshop_snRNA_session1_cellxgene.h5ad`) and the spatial example
(`/results/SpC_workshop_spatial_example.h5ad`) can be browsed interactively:
colour by gene, by `Group_V2`, lasso-select populations.

**Allen Brain Cell (ABC) Atlas** - explore the whole-brain reference we will map
onto in Session 2: <https://knowledge.brain-map.org/abcatlas>

> **Exercise:** pick a Leiden cluster, find its top marker in cellxgene, and look
> up that gene's whole-brain expression in the ABC Atlas.""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}
out = ('/code/lipari_genomics_workshop_2026/session1/notebooks/'
       'session1_qc_clustering_visualization.ipynb')
nbf.write(nb, out)
print('Wrote', out, 'with', len(cells), 'cells')
