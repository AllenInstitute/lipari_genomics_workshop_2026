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
spinal-cord single-nucleus RNA-seq atlas.

**Main section** (what we will work through together in class):

1. Inspect the dataset and compute **quality-control (QC)** metrics.
2. See **why we integrate across species** (naive PCA vs. scVI).
3. **Filter** low-quality nuclei using thresholds *you* choose.
4. **Normalize** and see **what proportion of each coarse cell class** QC removed.
5. Recompute the integration and view the **final clustering** (Allen
   `transcriptomic_clustering`) alongside the reference **Group_V2 / Subclass_V2** labels.

**Bonus section** (self-guided, likely *not* covered in class): marker genes,
exporting for **cellxgene**, a spatial transcriptomics teaser, and links to the
interactive atlases.

> Every random seed is fixed, so your results will match everyone else's exactly.

**How this runs:** we will walk through sections 0-2 together, then you get some
**free time in section 3** to tune the QC thresholds yourself and try to match the
atlas (the precision/recall readout is your score to beat). We regroup for the
**reveal** and finish sections 4-5 on *your* filtered data. The Bonus section is
yours to explore afterwards.

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
code("""adata = sc.read_h5ad('/data/lipari_workshop/SpC_workshop_snRNA.h5ad')
adata""")

md("""Look at the cell metadata (`obs`). Key columns:
- `qc_status` - whether the published atlas **kept** or **filtered out** each
  nucleus. We copy this into `atlas_sciduck_qc` shortly (computed in section 3
  setup) as the ground-truth target for the QC exercise; we keep it hidden until
  after you have chosen your own thresholds.
- `Class_V2` / `Subclass_V2` - the **reference cell-type taxonomy** (the labels we
  will compare our clusters to). These are only defined for nuclei that **passed**
  QC, so the filtered-out cells are `NaN` here.
- `class_coarse` - a *coarse* cell-class label (Non-Neurons, GABAergic,
  Glutamatergic, Motor Neurons, Cholinergic) assigned to **every** nucleus,
  *including* the ones that failed QC. Because it exists for all cells (unlike
  `Class_V2`), we use it - and only it - to apply **different QC thresholds to
  different cell classes** in the next section.
- precomputed QC metrics: `solo_doublet` (SOLO doublet probability),
  `percent_ribo`, `log.gene.counts.0` (= log10 of genes detected).
- `species` - the donor species.

We rename the carried atlas clustering to `atlas_leiden` (we compute our own
`leiden` from the scVI latent just below, before QC), rename the all-nuclei
propagated class to `class_coarse` (used for
QC only). The ground-truth QC decision (`atlas_sciduck_qc`) is computed silently
in section 3 setup - we will not peek at it until after you have chosen your own
thresholds.""")
code("""adata.obs = adata.obs.rename(columns={'leiden': 'atlas_leiden',
                                      'Class_propagated': 'class_coarse'})
print(f'{adata.n_obs:,} nuclei x {adata.n_vars:,} genes')
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

# distribution of the count values INCLUDING the zeros (log y-axis): the object is
# overwhelmingly zeros, and among detected genes most are just 1-2 UMIs. We bin the
# non-zero values and add the zero count into the 0 bin (X is too big to densify).
nz_vals = adata.X.data if sp.issparse(adata.X) else adata.X[adata.X > 0]
edges = np.arange(-0.5, 30.5, 1)          # bins centred on 0, 1, 2, ... 29
counts, _ = np.histogram(nz_vals, bins=edges)
counts[0] += n_total - n_nonzero          # fold every zero entry into the 0 bin
centers = np.arange(0, 30)
plt.figure(figsize=(7, 4))
plt.bar(centers, counts, width=0.9, color='#756bb1')
plt.yscale('log')
plt.xlabel('UMI count per entry (0 = gene not detected)')
plt.ylabel('number of entries (log scale)')
plt.title('X is dominated by zeros; detected genes have just 1-2 UMIs')
plt.tight_layout(); plt.show()""")

# ---- 1c. Why integrate across species -----------------------------------------
md("""### Why integrate across species? (naive PCA vs. scVI)

Our subsample pools nuclei from **multiple species**. Before we cluster, it is worth
seeing *why* we cannot just run the textbook pipeline (normalize -> HVGs -> PCA ->
UMAP) directly: without batch correction, the biggest source of variation is often
**which species a nucleus came from**, so the same cell type splits into a separate
island per species.

Below we build that naive embedding on a copy of the (pre-QC) data - normalize,
pick highly variable genes, scale, PCA, then UMAP - and colour it by `species`. We
compare it to the carried **scVI**-integrated embedding (`X_umap_prefilter`), which
was trained across species/donors and therefore **mixes the species together** so
clusters reflect *cell identity* instead of species. This is exactly why the rest of
the notebook clusters on the scVI latent (`X_scVI`) rather than on raw PCA.""")
code("""# Naive, un-integrated embedding on a copy: normalize -> HVG -> scale -> PCA -> UMAP.
demo = adata.copy()
sc.pp.normalize_total(demo, target_sum=1e4)
sc.pp.log1p(demo)
sc.pp.highly_variable_genes(demo, n_top_genes=2000)
demo = demo[:, demo.var['highly_variable']].copy()
sc.pp.scale(demo, max_value=10)
sc.tl.pca(demo, n_comps=50, random_state=SEED)
sc.pp.neighbors(demo, n_neighbors=15, n_pcs=30, random_state=SEED)
sc.tl.umap(demo, random_state=SEED)
adata.obsm['X_umap_pca'] = demo.obsm['X_umap']
del demo

# Left: naive PCA (species split apart). Right: scVI-integrated (species mixed).
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sc.pl.embedding(adata, basis='X_umap_pca', color='species', ax=axes[0],
                show=False, title='naive PCA on HVGs (not integrated)', size=8)
sc.pl.embedding(adata, basis='X_umap_prefilter', color='species', ax=axes[1],
                show=False, title='scVI-integrated (carried)', size=8)
plt.tight_layout(); plt.show()""")

# ---- 1d. Cluster the integrated latent ----------------------------------------
md("""### Cluster the integrated latent now

Because the scVI latent mixes the species sensibly, we can **cluster on it right
away** - on *all* nuclei, *before* any QC filtering. We build a k-nearest-neighbour
graph directly on `X_scVI` and run **Leiden** at a deliberately **high**
`resolution=15`, which **over-clusters** the data into many small groups (try
lowering it later and watch the cluster count change).

Computing the clusters *here* - rather than after filtering - is deliberate and is
part of the point of the QC section: the **group-level constraints** below drop
*whole* Leiden clusters whose average doublet signal is high, so those clusters have
to exist **before** we filter. We then reuse the very same `leiden` labels again in
sections 5-6 to visualize and name the cell types.""")
code("""sc.pp.neighbors(adata, use_rep='X_scVI', n_neighbors=15, random_state=SEED)
sc.tl.leiden(adata, resolution=15, random_state=SEED,
             flavor='igraph', n_iterations=2, directed=False)
print(adata.obs['leiden'].value_counts())

# show the clusters on the carried pre-filter UMAP (all nuclei, high-contrast palette)
with plt.rc_context({'figure.figsize': (8, 8)}):
    sc.pl.embedding(adata, basis='X_umap_prefilter', color='leiden',
                    palette=sc.pl.palettes.godsnot_102, legend_loc=None,
                    title='Leiden clusters on the scVI latent (pre-QC)')""")

# ---- 2. QC metrics ------------------------------------------------------------
md("""## 2. Inspect the QC metrics

The object already carries the QC metrics the atlas used (computing some of them,
like the SOLO doublet probabilities, needs a GPU and several minutes, so they were
precomputed upstream):
- **`log.gene.counts.0`** - log10 of the number of genes detected per nucleus.
- **`percent_ribo`** - fraction of counts from ribosomal genes.
- **`solo_doublet`** - the SOLO doublet probability (a deep-learning doublet score).

We recompute the two cheap ones (`percent_ribo`, `log.gene.counts.0`) from the raw
counts so you can see exactly how they are defined, then summarize all three.""")
code("""ribo_genes = adata.var_names.str.startswith(('RPS', 'RPL'))
adata.obs['percent_ribo'] = (np.asarray(adata[:, ribo_genes].X.sum(1)).ravel() /
                             np.asarray(adata.X.sum(1)).ravel())
adata.obs['log.gene.counts.0'] = np.log10(np.asarray((adata.X > 0).sum(1)).ravel() + 1)
adata.obs[['log.gene.counts.0', 'percent_ribo', 'solo_doublet']].describe().round(3)""")

md("""### Different cell classes have different "good" ranges

The key idea behind the atlas QC: **a healthy non-neuron and a healthy neuron have
very different gene counts**, so a single global threshold is wrong. The violin
below shows `log.gene.counts.0` per `class_coarse` (our all-nuclei coarse class) -
neurons (and especially motor neurons) detect more genes than non-neurons. We will
therefore set **class-specific** gene-count bounds.""")
code("""sc.pl.violin(adata, 'log.gene.counts.0', groupby='class_coarse',
             stripplot=False, rotation=30)""")

md("""Now look at the distributions of the SOLO doublet score and ribosomal fraction.
High doublet scores and very high ribosomal fractions are classic signs of
low-quality nuclei - keep an eye on the right-hand tails. *(We deliberately do not
show the atlas's keep/drop answer yet - you will pick your own thresholds first.)*""")
code("""def qc_histograms(df):
    metrics = ['solo_doublet', 'percent_ribo']
    fig, axes = plt.subplots(1, len(metrics), figsize=(10, 4))
    for ax, m in zip(axes, metrics):
        ax.hist(df[m], bins=60, color='#4c72b0')
        ax.set_xlabel(m); ax.set_ylabel('nuclei')
    plt.tight_layout(); plt.show()

qc_histograms(adata.obs)""")

md("""### Where might the low-quality nuclei sit on the UMAP?

Colour the **pre-filter UMAP** (`X_umap_prefilter`, computed on *all* nuclei) by
coarse cell class (`class_coarse`, defined for every nucleus), the per-nucleus
**SOLO doublet score**, and the **log10 number of UMIs** (`log_umi_counts`, sequencing
depth). Watch for small fringe islands and regions where the doublet score lights
up or the UMI depth drops - those are candidate low-quality territories. After you
set your own thresholds you will be cutting regions like these, and *then* we reveal
the atlas's decision.""")
code("""with plt.rc_context({'figure.figsize': (7, 7)}):
    sc.pl.embedding(adata, basis='X_umap_prefilter',
                    color=['class_coarse', 'solo_doublet', 'log_umi_counts'],
                    title=['coarse cell class', 'SOLO doublet score', 'log10 UMIs'],
                    size=8, cmap='viridis', ncols=3, wspace=0.3)""")

md("""### Normalize & log-transform before viewing gene expression

We are about to **colour cells by individual genes**. Raw UMI counts are *not*
comparable across nuclei - a nucleus with twice the sequencing depth shows ~twice the
counts for every gene - so before visualizing expression we must **library-size
normalize** (put every nucleus on the same total) and **log-transform** (compress the
long tail so a handful of high-count genes don't dominate the colour scale).

We keep the **raw counts in `X`** (the count-based QC metrics above depend on them)
and store the normalized-and-logged values in a separate **`lognorm` layer** that we
use for every gene-expression plot from here on.""")
code("""# raw counts stay in X (needed by the QC metrics); make a log-normalized copy
# purely for *visualizing* gene expression.
lognorm = adata.copy()
sc.pp.normalize_total(lognorm, target_sum=1e4)
sc.pp.log1p(lognorm)
adata.layers['lognorm'] = lognorm.X
del lognorm
print('stored adata.layers[\"lognorm\"] for gene-expression plots')""")

md("""### Spotting doublets: markers that should never co-occur

A **doublet** is two nuclei captured as one, so it can show markers of two distinct
lineages at once. `AQP4` (astrocytes), `MBP` (oligodendrocytes) and `RBFOX3`
(neurons) each mark a *different*, mutually-exclusive cell class - a single nucleus
should light up for **at most one** of them. Nuclei that express **two or more**
are prime doublet suspects, and they tend to carry higher SOLO doublet scores.""")
code("""import scipy.sparse as sp

# How many of these mutually-exclusive lineage markers are DETECTED (count > 0)?
lineage_markers = {'AQP4': 'astrocyte', 'MBP': 'oligodendrocyte', 'RBFOX3': 'neuron','AIF1':'microglia'}
present = [g for g in lineage_markers if g in adata.var_names]

n_lineages = np.zeros(adata.n_obs, dtype=int)
for g in present:
    xg = adata[:, g].X
    xg = xg.toarray().ravel() if sp.issparse(xg) else np.asarray(xg).ravel()
    n_lineages += (xg > 0).astype(int)
adata.obs['n_lineage_markers'] = n_lineages
n_suspect = int((n_lineages >= 2).sum())
print(f'{n_suspect:,} nuclei ({100 * n_suspect / adata.n_obs:.1f}%) co-express '
      f'>=2 mutually-exclusive lineage markers ({", ".join(present)})')

# Each marker should light up a DISTINCT region of the UMAP (log-normalized expression)...
with plt.rc_context({'figure.figsize': (5, 5)}):
    sc.pl.embedding(adata, basis='X_umap_prefilter', color=present, layer='lognorm',
                    size=8, cmap='viridis', ncols=3, wspace=0.3)
    # ...and the multi-lineage (doublet-suspect) nuclei sit where those regions meet.
    sc.pl.embedding(adata, basis='X_umap_prefilter', color='n_lineage_markers',
                    size=8, cmap='Reds', wspace=0.3,
                    title='# mutually-exclusive lineages detected')

# Doublet-suspect nuclei carry higher SOLO doublet scores on average.
_lab = adata.obs['n_lineage_markers'].clip(upper=2).map(
    {0: '0 markers', 1: '1 marker', 2: '>=2 (doublet-suspect)'})
print(adata.obs.groupby(_lab, observed=True)['solo_doublet'].mean())""")

md("""**But this simple co-occurrence test is not enough.** Counting mutually-exclusive
markers only flags the most obvious inter-lineage doublets in very clean sammples, and it fails if data if there is significant ambient RNA (there usually is!), whenever two
*similar* cell types (say two neuronal subtypes) collide, or when dropout hides one
partner's markers. That is exactly why we lean on more sophisticated methods like
**SOLO**: it *simulates* artificial doublets by adding together pairs of real nuclei,
then trains a **deep-learning probabilistic model** to tell those simulated doublets
apart from singlets. The resulting per-nucleus `solo_doublet` probability is the score
we actually filter on below - far more sensitive than the marker-overlap heuristic.""")
md("""## 3. Class-specific QC with `sciduck`

We reproduce the atlas QC strategy with **`sciduck`**, which lets us register a set
of **constraints** and then apply them all at once. A nucleus is kept only if it
satisfies **all** registered constraints (they are combined with logical AND). There
are three kinds of constraint:

- **Range constraint** (`add_range_constraint`) - keep nuclei whose value in a
  metric column sits inside a `[gt, lt]` range (either bound may be left open). This
  is the workhorse for continuous QC metrics like gene counts, doublet score or
  ribosomal fraction.
- **Exclude constraint** (`add_exclude_constraint`) - drop nuclei whose value in a
  column is in a given list (e.g. blacklist specific `batch` or `donor` labels).
  Handy for categorical metadata rather than numeric ranges.
- **Group-level constraint** (`add_group_level_constraint`) - summarize a metric
  **per group** (e.g. per Leiden cluster) with an aggregation (`mean`/`median`/etc.)
  and keep or drop **whole groups** by that summary. This removes entire clusters
  that are, on average, doublet- or debris-dominated even if individual nuclei look
  borderline.

Any of these can be **restricted to a subset** of nuclei via `subset`/`subset_values`
- that is exactly how we apply *different* gene-count ranges to different cell
classes below. `apply_constraints` evaluates them all and writes a boolean
`keeper_cells` column (plus a per-constraint record of what each one removed in
`adata.uns['qc_filtered']`).""")
code("""!pip install -q sciduck""")
code("""import sciduck as sd""")

md("""# 🛠️ Your turn (free time): set the thresholds!

**This is the hands-on part of the session.** Take a few minutes to tune the
thresholds below yourself. **These presets start wide open - they keep essentially
every nucleus, including the bad ones.** Your job is to *tighten* them. `GENE_BOUNDS`
gives the allowed `log.gene.counts.0` **(low, high)** range **per cell class**
(non-neurons typically need fewer genes than neurons, and motor neurons the most).
The doublet/ribo cut-offs are global (the doublet filter uses the SOLO score). Edit
the numbers, re-run this cell and the
diagnostics below - your **score** (precision/recall vs the atlas) will be revealed
at the end. *Don't scroll to the reveal yet* - we will look at the atlas's actual
answer together once everyone has had a go.""")
code("""# >>> EDIT THESE THRESHOLDS <<<
# These defaults are deliberately permissive - they let basically everything
# through. Narrow them until you are only keeping good-quality nuclei.
# allowed log10(genes detected) range, per cell class
GENE_BOUNDS = {
    'Non-Neurons':   (0.0, 5.0),   # wide open - tighten me!
    'neurons':       (0.0, 5.0),   # GABAergic / Glutamatergic / Cholinergic
    'Motor Neurons': (0.0, 5.0),   # motor neurons
}
MAX_SOLO_DOUBLET  = 1.0   # drop nuclei above this SOLO doublet probability
MAX_RIBO          = 1.0   # drop nuclei above this ribosomal fraction
# group-level (per atlas cluster) doublet limit - also wide open to start
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
and two **group-level** constraints that can remove whole **Leiden clusters** (the
ones you computed on the scVI latent near the top) dominated by doublets. With the
wide-open presets nothing is removed yet - `apply_constraints` writes a fresh
`keeper_cells` column that currently keeps (almost) everything.""")
code("""for k in ['qc_constraints', 'qc_filtered']:
    adata.uns.pop(k, None)

classes = list(adata.obs['class_coarse'].cat.categories)
neuron_classes = [c for c in classes if c not in ('Non-Neurons', 'Motor Neurons')]

# global doublet / ribosomal limits
sd.basic_qc.add_range_constraint(adata, 'percent_ribo', lt=MAX_RIBO)
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

# group-level: drop whole Leiden clusters whose mean doublet signal is high
sd.basic_qc.add_group_level_constraint(adata, 'solo_doublet', groupby='leiden', lt=MAX_GROUP_SOLO)

sd.basic_qc.apply_constraints(adata)
print(adata.obs['keeper_cells'].value_counts())""")

md("""### See exactly what *you* are about to remove

Plot your own decision on the pre-filter UMAP: nuclei you marked to **keep** vs
**remove**. Tightening or loosening the thresholds in the cell above moves the
boundary of the orange region. Re-run the constraints with different numbers - then
scroll down to the reveal to see how the atlas actually drew the line.""")
code("""adata.obs['your_qc'] = np.where(adata.obs['keeper_cells'], 'keep', 'remove')
with plt.rc_context({'figure.figsize': (8, 8)}):
    sc.pl.embedding(adata, basis='X_umap_prefilter', color='your_qc',
                    title='your QC decision (pre-filter UMAP)',
                    palette={'keep': '#2c7fb8', 'remove': '#d95f0e'}, size=8)""")

md("""# The reveal: what did the atlas actually filter?

*(Regroup here once everyone has tuned their thresholds.)* We now apply the atlas's
published metric thresholds to produce `atlas_sciduck_qc` - **and reveal your score**.""")
code("""# Compute the atlas metric-threshold answer now (thresholds revealed below)
_saved_keeper = adata.obs['keeper_cells'].copy()
for k in ['qc_constraints', 'qc_filtered']:
    adata.uns.pop(k, None)

_all_cls    = list(adata.obs['class_coarse'].cat.categories)
_non_chol   = [c for c in _all_cls if c != 'Cholinergic']
_neuron_cls = [c for c in _all_cls if c not in ('Non-Neurons', 'Motor Neurons')]
sd.basic_qc.add_range_constraint(adata, 'percent_ribo', lt=0.2)
sd.basic_qc.add_range_constraint(adata, 'solo_doublet', lt=0.55,
    subset='class_coarse', subset_values=_non_chol)
sd.basic_qc.add_range_constraint(adata, 'log.gene.counts.0', gt=2.7, lt=3.7,
    subset='class_coarse', subset_values=['Non-Neurons'])
sd.basic_qc.add_range_constraint(adata, 'log.gene.counts.0', gt=2.8, lt=3.8,
    subset='class_coarse', subset_values=_neuron_cls)
sd.basic_qc.add_range_constraint(adata, 'log.gene.counts.0', gt=2.9, lt=5.0,
    subset='class_coarse', subset_values=['Motor Neurons'])
sd.basic_qc.add_group_level_constraint(adata, 'solo_doublet',  groupby='leiden', lt=0.5)
sd.basic_qc.apply_constraints(adata)
adata.obs['atlas_sciduck_qc'] = adata.obs['keeper_cells'].map({True: 'passed_qc', False: 'filtered_out'})
adata.obs['keeper_cells'] = _saved_keeper  # restore student's choice
for k in ['qc_constraints', 'qc_filtered']:
    adata.uns.pop(k, None)

# Score: how well did the student recover the atlas decision?
keep       = adata.obs['your_qc'] == 'keep'
atlas_keep = adata.obs['atlas_sciduck_qc'] == 'passed_qc'
precision  = (keep & atlas_keep).sum() / max(keep.sum(), 1)
recall     = (keep & atlas_keep).sum() / max(atlas_keep.sum(), 1)
print(f'Your score: precision {precision:.2f}, recall {recall:.2f}')
print(pd.crosstab(keep.map({True: 'keep', False: 'remove'}), adata.obs['atlas_sciduck_qc']))""")

md("""Now we re-draw **all** the QC histograms from above - gene counts, total UMIs,
the SOLO doublet score and the ribosomal fraction - split into the nuclei the atlas
**kept** (blue) vs **filtered out** (orange). Notice how the filtered nuclei pile up
in the **low gene-count / low-UMI** and **high-doublet / high-ribo** tails. On the
pre-filter UMAP the filtered-out nuclei form their own fringe territory rather than
mixing into the healthy populations. Compare with your own `keep`/`remove` map
above: how close did you get?""")
code("""def qc_histograms_reveal(df):
    # the same metrics we plotted above (depth + doublet/ribo), now split by atlas decision
    metrics = ['log.gene.counts.0', 'log10_total_counts',
               'solo_doublet', 'percent_ribo']
    ncols = 3
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, metrics):
        for status, color in [('passed_qc', '#2c7fb8'), ('filtered_out', '#d95f0e')]:
            ax.hist(df.loc[df['atlas_sciduck_qc'] == status, m], bins=60, alpha=0.6,
                    label=status, color=color)
        ax.set_xlabel(m); ax.set_ylabel('nuclei'); ax.legend()
    for ax in axes[len(metrics):]:
        ax.set_visible(False)
    plt.tight_layout(); plt.show()

qc_histograms_reveal(adata.obs)

with plt.rc_context({'figure.figsize': (8, 8)}):
    sc.pl.embedding(adata, basis='X_umap_prefilter', color='atlas_sciduck_qc',
                    title='atlas QC decision (ground truth)',
                    palette={'passed_qc': '#2c7fb8', 'filtered_out': '#d95f0e'},
                    size=8)""")

md("""### What the atlas actually used

The `atlas_sciduck_qc` labels above were computed using the **same three `sciduck`
constraint types** you just worked with - the thresholds from the published
cross-species QC notebook
(`HMBA_Genomics/SpinalCord/xspecies/integration/SpC_External_QC.ipynb`):

**Range constraints (per-cell):**

| metric | range kept | applied to |
|---|---|---|
| `log.gene.counts.0` | `[2.7, 3.7]` | Non-Neurons |
| `log.gene.counts.0` | `[2.8, 3.8]` | neurons (GABAergic / Glutamatergic / Cholinergic) |
| `log.gene.counts.0` | `[2.9, 5.0]` | Motor Neurons |
| `solo_doublet` | `< 0.55` | all classes **except Cholinergic** |
| `percent_ribo` | `< 0.2` | all nuclei |

**Group-level constraints (whole Leiden clusters, by cluster mean):**

| metric | rule | effect |
|---|---|---|
| `solo_doublet` | mean `< 0.5` per cluster | drop clusters that are doublet-dominated on average |

A few design choices worth noting:

- The gene-count ranges are **two-sided**: the *lower* bound removes empty/low-quality
  droplets, while the *upper* bound (e.g. `3.7` for non-neurons) removes suspiciously
  gene-rich nuclei that are likely **doublets** - and motor neurons get a much higher
  ceiling (`5.0`) because they are genuinely large, transcript-rich cells.
- **Cholinergic** nuclei are **exempt from the SOLO doublet filter**: cholinergic /
  motor-neuron cells naturally carry very high counts and co-express broad programs,
  so the doublet detector flags them as false positives.
- The **group-level** rules delete *entire* clusters (not just individual cells) that
  are collectively low-quality, which the per-cell metrics alone would miss.""")

md("""## 4. What proportion of each coarse cell class did QC remove?

Rather than squint at individual nuclei on a UMAP, let's summarize the **outcome** of
your filtering. `class_coarse` is defined for **every** nucleus (QC-passed or not), so
we can ask a simple, practical question: for each coarse cell class, **what fraction
was removed by QC?** Some classes are intrinsically harder to sequence cleanly (fewer
genes, more ambient contamination) and lose a larger share.

We compute this on the **full, pre-filter** set of nuclei - so the denominators are
complete - for **your** thresholds (`your_qc`) and, for reference, the **atlas**
thresholds (`atlas_sciduck_qc`).""")
code("""qc_by_class = pd.DataFrame({
    'your QC':  adata.obs.groupby('class_coarse', observed=True)['your_qc']
                    .apply(lambda s: (s == 'remove').mean()),
    'atlas QC': adata.obs.groupby('class_coarse', observed=True)['atlas_sciduck_qc']
                    .apply(lambda s: (s == 'filtered_out').mean()),
}) * 100
qc_by_class['n_nuclei'] = adata.obs['class_coarse'].value_counts()
qc_by_class = qc_by_class.sort_values('your QC', ascending=False)
print('percent of each coarse class removed by QC:')
qc_by_class.round(1)""")

md("""The same numbers as a bar chart: for each coarse class, the **percent of nuclei
filtered out** by your thresholds vs the atlas's. Classes on the left lose the most -
if your bars sit far from the atlas's, that class is where your thresholds were the
most/least aggressive.""")
code("""ax = qc_by_class[['your QC', 'atlas QC']].plot.bar(
    figsize=(8, 4), color=['#d95f0e', '#2c7fb8'])
ax.set_ylabel('% of class removed by QC')
ax.set_xlabel('coarse cell class')
ax.set_title('Proportion of each coarse cell class filtered in QC')
plt.xticks(rotation=30, ha='right')
plt.tight_layout(); plt.show()""")

md("""### Keep the clean nuclei

Subset to your `keeper_cells`, then library-size normalize to 10,000 counts/cell and
`log1p`-transform the retained nuclei (these log-normalized values feed the marker-gene
plots in the Bonus section). We keep the raw counts in a `counts` layer so nothing is
lost.""")
code("""adata = adata[adata.obs['keeper_cells']].copy()
adata.layers['counts'] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata
print(f'{adata.n_obs:,} nuclei retained and normalized')""")

# ---- 5. Final clustering on the recomputed scVI ------------------------------
md("""## 5. Recompute scVI and get the final clustering

The UMAP you have been looking at so far (`X_umap_prefilter`) was computed on
**all** nuclei - including the low-quality ones you just removed. Now that you have
a clean subset, it is worth rebuilding the integration on just the nuclei you kept
and deriving the **final clustering** from it.

The script `02b_build_session2_clean.py` does exactly that: it takes the
atlas-filtered subsample, **retrains scVI** across species/study, and writes a
fresh `X_scVI` latent + `X_umap` to a new object. Run it from a terminal (GPU
recommended; ~5 min on an L4):

```
python /code/lipari_genomics_workshop_2026/session1/processing/02b_build_session2_clean.py
```

A pre-built copy of this object is provided at
`/data/lipari_workshop/SpC_workshop_snRNA_session2_clean.h5ad`, which we load
below (if you re-run the script yourself it writes a fresh copy to `/results/`).

**How the final clusters are defined.** We do *not* just pick a Leiden resolution by
eye. On the recomputed scVI latent we run the Allen Institute's
[`transcriptomic_clustering`](https://github.com/AllenInstitute/transcriptomic_clustering)
package, which finds the *proper* clustering resolution automatically: it keeps
**splitting clusters only while there are still differentially expressed genes
between the resulting groups**, and stops once neighbouring clusters are no longer
separable by DE genes. The clusters below (`Group_V2`) are the outcome of that
data-driven procedure applied to the integrated latent - the atlas's **final
clustering**, which we then compare to the reference `Subclass_V2` labels.""")
code("""session2 = sc.read_h5ad('/data/lipari_workshop/SpC_workshop_snRNA_session2_clean.h5ad')
print(f'{session2.n_obs:,} nuclei x {session2.n_vars:,} genes')
print(f"{session2.obs['Group_V2'].nunique()} final clusters (Group_V2)")
session2.obs[['species', 'study', 'Group_V2', 'Subclass_V2']].head()""")
code("""with plt.rc_context({'figure.figsize': (7, 7)}):
    # the final clustering (transcriptomic_clustering Group_V2) vs the reference subclass
    sc.pl.umap(session2,
               color=['Group_V2', 'Subclass_V2'],
               palette=sc.pl.palettes.godsnot_102,
               ncols=2, wspace=0.35, legend_loc=None, size=8)
    sc.pl.umap(session2,
               color=['species', 'study'],
               palette=sc.pl.palettes.godsnot_102,
               ncols=2, wspace=0.35,
               legend_fontsize=6, size=8)""")

# ---- BONUS divider ------------------------------------------------------------
md("""---
# Bonus material (self-guided)

**Everything below is bonus** and we will most likely **not** cover it live in class.
It picks up right where the main section left off - you have clusters and reference
labels - and walks through:

- **6. Marker genes** - reading out neurotransmitter identity per cluster.
- **7. Export for cellxgene** - saving a browser-ready object.
- **8. A glimpse of spatial transcriptomics.**
- **9. Explore the data interactively** - links to the online atlases.

Work through it at your own pace after the session.

---""")

# ---- 7b. Bonus: highest-expressed genes ---------------------------------------
md("""### What dominates the reads?

Because we sequenced *nuclei*, a handful of long, intron-rich and nuclear-retained
transcripts (e.g. `MALAT1`) soak up a large share of every library.
`highest_expr_genes` shows the fraction of total counts taken by the top genes - a
useful sanity check that also explains why nuclear data looks different from
whole-cell data. We read from the **raw `counts` layer** (the fraction-of-reads
metric only makes sense on raw UMIs, not the log-normalized `X`).""")
code("""sc.pl.highest_expr_genes(adata, n_top=20, layer='counts')""")

# ---- 6. Marker genes ----------------------------------------------------------
md("""## 6. Marker genes

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

md("""Finally, let the **data** nominate **conserved** markers - genes that mark a
cell type **in every species**, not just on average. For each reference
`Supergroup_V2` cell type we rank genes with **logistic regression** (a multinomial
classifier - each gene's coefficient measures how much its expression predicts that
cell type against all the others) **separately within each species**, then score each
gene by its **minimum** marker score across species: a gene only wins if it is
strongly enriched in **human and macaque and mouse**. A species is **skipped** for a
group when that group has **fewer than 20 cells** there (too few to rank reliably).

This is a **deliberately simple first pass** - keep an eye on the heatmaps below, and
you will see it is far from perfect. We turn improving it into a challenge right
after.""")
code("""# Conserved markers: rank genes per species with logistic regression, then take
# the MIN coefficient across species so a gene only wins if it marks the cell type
# in *every* species.
GROUPBY = 'Supergroup_V2'
MIN_CELLS_PER_SPECIES = 20   # skip a species for a group with fewer cells than this

adata_lab = adata[adata.obs[GROUPBY].notna()].copy()
adata_lab.obs[GROUPBY] = adata_lab.obs[GROUPBY].cat.remove_unused_categories()

# Per-species marker scores (one logistic-regression ranking within each species),
# restricted to the groups that pass the cell-count threshold in that species.
species_scores = {}
for sp in adata_lab.obs['species'].unique():
    a_sp = adata_lab[adata_lab.obs['species'] == sp].copy()
    a_sp.obs[GROUPBY] = a_sp.obs[GROUPBY].cat.remove_unused_categories()
    counts = a_sp.obs[GROUPBY].value_counts()
    valid = counts.index[counts >= MIN_CELLS_PER_SPECIES].tolist()
    if len(valid) < 3:
        continue  # need >=3 groups for a multinomial logistic-regression ranking
    sc.tl.rank_genes_groups(a_sp, GROUPBY, groups=valid, method='logreg',
                            max_iter=1000)
    species_scores[sp] = sc.get.rank_genes_groups_df(a_sp, group=None)
print(f'ranked markers within {len(species_scores)} species: {list(species_scores)}')

# Conserved score for a (group, gene) = min score across the species where the
# group had >=20 cells; the top-scoring gene is that group's conserved marker.
conserved_markers = {}
for g in adata_lab.obs[GROUPBY].cat.categories:
    cols = [df[df['group'] == g].set_index('names')['scores']
            for df in species_scores.values() if (df['group'] == g).any()]
    if not cols:
        continue  # group never reached 20 cells in any species
    conserved_markers[g] = pd.concat(cols, axis=1).min(axis=1).idxmax()

for g, gene in conserved_markers.items():
    print(f'  {g}: {gene}')""")

md("""Now show the conserved-marker panel as **one heatmap per species**. Columns are
the conserved markers (one per group); rows are individual nuclei, grouped by their
reference cell type and **scaled 0-1 per gene within each species**. Because the
same genes light up the same cell types in **human, macaque, and mouse**, the three
panels share the same block-diagonal pattern - that is what "conserved" looks like.""")
code("""# Deduplicate markers (two groups can share the same top conserved gene) while
# preserving their order, so each gene is shown exactly once.
conserved_genes = list(dict.fromkeys(conserved_markers.values()))

# One heatmap per species so students can eyeball that each marker is enriched in
# the same cell type across all three species (a shared block-diagonal pattern).
# We print the species name rather than using suptitle (which overlaps the
# gene-label axis at the top of scanpy heatmaps).
for sp in sorted(adata_lab.obs['species'].unique()):
    print(f'=== {sp} - conserved markers ===')
    a_sp = adata_lab[adata_lab.obs['species'] == sp].copy()
    a_sp.obs[GROUPBY] = a_sp.obs[GROUPBY].cat.remove_unused_categories()
    sc.pl.heatmap(a_sp, conserved_genes, groupby=GROUPBY, standard_scale='var',
                  cmap='viridis', show_gene_labels=True,
                  figsize=(max(8, 0.45 * len(conserved_genes)), 6))""")

md("""### 🧩 Challenge: can you find *better* conserved markers?

Look hard at the heatmaps above - the block-diagonal is **messy**. Several columns
light up in more than one cell type, some cell types share the *same* top gene, and a
marker that looks great in one species can be washed out in another. The recipe we used
is crude on purpose, and you can almost certainly beat it. Some of its weaknesses:

- **Only one gene per cell type.** We keep the single top-scoring gene per group; a
  usable panel usually needs several genes, and the runner-up genes are thrown away.
- **A big coefficient is not a good marker.** The logistic-regression score rewards
  discriminative power, but says nothing about whether the gene is *specific* (off in
  the other cell types) or even *detectable* in most cells of the group.
- **`min` across species is brittle.** One under-powered species can veto an otherwise
  excellent marker, and a single hard bottleneck ignores *how* strong the agreement is.
- **No effect-size or expression-fraction filter,** so lowly-expressed, noisy genes can
  sneak to the top.

**Your task: design and implement a better conserved-marker selector.** A few directions
worth trying:

1. Score each gene per group with **interpretable, comparable** statistics in *every*
   species - e.g. **log fold-change**, and the **fraction of cells expressing it
   in-group vs out-of-group** (a specificity signal) - and require all three species to
   clear a bar on each.
2. Aggregate across species with a **rank product / rank-sum** rather than a raw `min`,
   so no single species dominates but weak agreement is still penalized.
3. Return a **ranked shortlist (top-k)** per cell type, then keep only genes that are
   *specific* - high in the target type and low nearly everywhere else.
4. **Validate by transfer:** hold out one species, pick markers from the other two, and
   check they still separate the cell type in the held-out species. Good conserved
   markers should generalize to a species they were not chosen on.

The starter below re-ranks genes per species with **Wilcoxon** and `pts=True`, which
gives you `logfoldchanges`, `pct_nz_group`, and `pct_nz_reference` alongside the score -
the raw material for a smarter selector. Take it from there.""")
code("""# Starter kit: richer per-species statistics for you to build a better selector on.
# Wilcoxon with pts=True adds logfoldchanges + the fraction of cells expressing each
# gene inside the group (pct_nz_group) vs outside it (pct_nz_reference).
species_stats = {}
for sp in sorted(adata_lab.obs['species'].unique()):
    a_sp = adata_lab[adata_lab.obs['species'] == sp].copy()
    a_sp.obs[GROUPBY] = a_sp.obs[GROUPBY].cat.remove_unused_categories()
    counts = a_sp.obs[GROUPBY].value_counts()
    valid = counts.index[counts >= MIN_CELLS_PER_SPECIES].tolist()
    if len(valid) < 2:
        continue
    sc.tl.rank_genes_groups(a_sp, GROUPBY, groups=valid, method='wilcoxon', pts=True)
    species_stats[sp] = sc.get.rank_genes_groups_df(a_sp, group=None)

# Peek at what you now have to work with (score + effect size + specificity):
example_sp = next(iter(species_stats))
print(f'per-species stats available for: {list(species_stats)}')
print(f'columns: {species_stats[example_sp].columns.tolist()}')
species_stats[example_sp].head()

# TODO (your turn): combine logfoldchanges + (pct_nz_group - pct_nz_reference) across
# ALL species into a single conserved-specificity score, return the top-k genes per
# cell type, and re-draw the heatmaps. Does the block-diagonal get cleaner? Then try
# the hold-one-species-out transfer test.""")

md("""### 🧩 Challenge: which cell types are the most *divergent* across species?

The markers above ask which genes are **conserved**. Flip the question: **which cell
types differ the most between human, macaque and mouse?** Some lineages (e.g. many
non-neurons) are nearly identical across species, while others may have drifted. Design
a way to rank the `Supergroup_V2` (or `Subclass_V2`) cell types from most-conserved to
most-divergent. Two complementary approaches:

- **Cross-species expression correlation.** For each cell type, compute a
  **pseudobulk** profile (mean log-normalized expression across its cells) *separately
  per species*, then correlate the profiles between species pairs (Pearson or Spearman
  over shared genes - restrict to HVGs to cut noise). A **low** average
  between-species correlation means that cell type is divergent. Rank the cell types by
  this score and plot it.
- **Differential expression between species, within a cell type.** Subset to one cell
  type, set `groupby='species'`, and run `sc.tl.rank_genes_groups` to count how many
  genes are significantly different between species (e.g. `pvals_adj < 0.05` and
  `|logfoldchanges| > 1`). More species-DE genes ⇒ more divergent. Which cell types top
  the list, and do they agree with the correlation ranking?

**Watch for confounds:** a cell type with very few cells in one species, or uneven
sequencing depth between species, can masquerade as "divergent." Guard against it by
requiring a minimum cell count per species and by working on the log-normalized (not
raw) values.""")

# ---- 7. Export for cellxgene --------------------------------------------------
md("""## 7. Export for cellxgene

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

# ---- 8. Spatial teaser -------------------------------------------------------
md("""## 8. A glimpse of spatial transcriptomics

snRNA-seq tells us **what** cell types exist, but not **where** they sit. Spatial
methods profile transcripts while preserving tissue coordinates. The workshop ships
a small spatial example (`/data/lipari_workshop/SpC_workshop_spatial_example.h5ad`) holding the
three **representative cross-species sections** (human, macaque, mouse) from the
manuscript, each mapped onto the same `Group_V2` taxonomy and colours.

We plot every neuron at its real, orientation-corrected `(_plot_x, _plot_y)`
position - this reproduces the butterfly grey-matter shape, with motor neurons in
the ventral horn and dorsal-horn populations up top. Non-neurons are drawn as faint
grey background dots (`SpC_workshop_spatial_nn_overlay.tsv.gz`), and per-section
crop bounds + the `Group_V2` palette come from `SpC_workshop_spatial_meta.json`.
This mirrors the Figure 2 section panel; we dig into it in Session 2.""")
code("""import json
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# ── Load the 3-species spatial example + its companion artifacts ──────────────
spatial = sc.read_h5ad('/data/lipari_workshop/SpC_workshop_spatial_example.h5ad')
spatial_obs = spatial.obs

# The plot uses the transformed coordinates in obs['_plot_x'/'_plot_y']. If you
# loaded an older/cellxgene-safe object where those obs columns were dropped,
# fall back to obsm (spatial / X_spatial). If neither is present, the object is
# stale - re-run processing/02_build_spatial_example.py to regenerate it.
if not {'_plot_x', '_plot_y'}.issubset(spatial_obs.columns):
    _coords = spatial.obsm.get('spatial', spatial.obsm.get('X_spatial'))
    if _coords is None:
        raise KeyError(
            "No '_plot_x'/'_plot_y' in obs and no 'spatial' coordinates in obsm - "
            "regenerate /results/SpC_workshop_spatial_example.h5ad by running "
            "processing/02_build_spatial_example.py")
    spatial_obs = spatial_obs.copy()
    spatial_obs['_plot_x'] = np.asarray(_coords)[:, 0].astype(float)
    spatial_obs['_plot_y'] = np.asarray(_coords)[:, 1].astype(float)
if spatial_obs['species'].nunique() < 3:
    print('WARNING: spatial object has <3 species - it looks stale. Re-run '
          'processing/02_build_spatial_example.py for the full cross-species panel.')

nn_obs = pd.read_csv('/data/lipari_workshop/SpC_workshop_spatial_nn_overlay.tsv.gz',
                     sep='\\t', index_col=0, compression='gzip')

with open('/data/lipari_workshop/SpC_workshop_spatial_meta.json') as f:
    _meta = json.load(f)
rep_example_sections = _meta['representative_sections']     # species -> section id
rep_crop    = {k: tuple(v) for k, v in _meta['rep_crop'].items()}
group_color = _meta['group_color']                          # Group_V2 -> hex

# Groups actually present across the three sections, in curated palette order.
_present = set(spatial_obs['Group_V2'].astype(str))
groups_ordered = [g for g in group_color if g in _present]
print(f'{spatial.n_obs:,} neurons across {len(rep_example_sections)} sections, '
      f'{len(groups_ordered)} Group_V2 types present')""")
code("""# ── Plot: all Group_V2 cell types in the representative sections ──────────────
# (Reproduces 03_figure2_plot_panels.ipynb: all groups, representative sections)
_n_sp = len(rep_example_sections)
_sp_wh = []
for _sp in rep_example_sections:
    _c = rep_crop.get(_sp)
    if _c is not None:
        _sp_wh.append((abs(_c[1] - _c[0]), abs(_c[3] - _c[2])))
    else:
        _ss = spatial_obs[spatial_obs['_section'].astype(str) == rep_example_sections[_sp]]
        _sp_wh.append((max(float(_ss['_plot_x'].astype(float).max()
                                 - _ss['_plot_x'].astype(float).min()), 1.0),
                       max(float(_ss['_plot_y'].astype(float).max()
                                 - _ss['_plot_y'].astype(float).min()), 1.0)))
_fig_h_sp = 7.0
_w_ratios_sp = [w / h for w, h in _sp_wh]
_fig_w_sp = _fig_h_sp * (sum(_w_ratios_sp) + 0.02 * max(_n_sp - 1, 0))

fig, axes = plt.subplots(1, _n_sp, figsize=(_fig_w_sp, _fig_h_sp), squeeze=False,
                         gridspec_kw={'width_ratios': _w_ratios_sp})
axes = list(axes[0])
fig.subplots_adjust(wspace=0.02)

for ax, (species, sec) in zip(axes, rep_example_sections.items()):
    sub = spatial_obs[spatial_obs['_section'].astype(str) == sec].copy()
    if len(sub) == 0:
        ax.set_title(f'{species} - no cells found'); ax.axis('off'); continue

    sub = sub[sub['rexed_lamina'].astype(str) != '']
    xl  = sub['_plot_x'].values.astype(float)
    yl  = sub['_plot_y'].values.astype(float)
    grp = sub['Group_V2'].astype(str).values

    # Non-neurons: small grey background dots
    _nn_sec = nn_obs[nn_obs['_section'].astype(str) == sec]
    if len(_nn_sec):
        ax.scatter(_nn_sec['_plot_x'].values.astype(float),
                   _nn_sec['_plot_y'].values.astype(float),
                   s=4., c='#b8b8b8', linewidths=0, alpha=0.25, rasterized=True)

    for gname in groups_ordered:
        m = grp == gname
        if m.any():
            ax.scatter(xl[m], yl[m], s=15., c=[group_color[gname]],
                       linewidths=0, alpha=0.9, rasterized=True)

    ax.set_aspect('equal'); ax.invert_yaxis()
    crop = rep_crop.get(species)
    if crop is not None:
        _x0, _x1, _ytop, _ybot_ext = crop
        ax.set_xlim(_x0, _x1)
        ax.set_ylim(_ybot_ext, _ytop)
    ax.set_title(f'{species.capitalize()}\\n{len(sub):,} cells in labeled laminae',
                 fontsize=9)
    ax.add_artist(AnchoredSizeBar(
        ax.transData, 500, '500 \u00b5m', loc='lower right', pad=0.3,
        color='black', frameon=False, size_vertical=20,
        fontproperties=fm.FontProperties(size=7)))
    ax.set_axis_off()

handles = [mpatches.Patch(color=group_color[g], label=g) for g in groups_ordered]
fig.legend(handles, [h.get_label() for h in handles], loc='lower center', ncol=4,
           fontsize=6, title='Group_V2', title_fontsize=8,
           bbox_to_anchor=(0.5, -0.35), framealpha=0.9)
fig.suptitle('All Cell-Type Groups - Representative Cross-Species Sections',
             fontsize=13)
plt.show()""")

# ---- 9. Interactive atlases --------------------------------------------------
md("""## 9. Explore the data interactively

**cellxgene** - the cellxgene-safe object you just saved
(`SpC_workshop_snRNA_session1_cellxgene.h5ad`) and the spatial example
(`/data/lipari_workshop/SpC_workshop_spatial_example.h5ad`) can be browsed interactively:
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
