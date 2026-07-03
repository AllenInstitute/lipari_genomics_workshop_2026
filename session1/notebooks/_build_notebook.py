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
clustering and UMAP below are byte-for-byte reproducible across machines.

The pre-built workshop artifacts are distributed **read-only** in
`DATA_DIR` (`/data/lipari_workshop`); anything you save goes to the writable
`RESULTS_DIR` (`/results`).""")
code("""import os, random
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Workshop paths ────────────────────────────────────────────────────────────
# Pre-built inputs are read-only under DATA_DIR; write everything to RESULTS_DIR.
DATA_DIR = '/data/lipari_workshop'
RESULTS_DIR = '/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

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
code("""adata = sc.read_h5ad(f'{DATA_DIR}/SpC_workshop_snRNA.h5ad')
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
QC only), and set aside an `atlas_qc` copy of the ground-truth decision **for later**
- we will not peek at it until after you have chosen your own thresholds.""")
code("""adata.obs = adata.obs.rename(columns={'leiden': 'atlas_leiden',
                                      'Class_propagated': 'class_coarse'})
# Ground-truth keep/drop decision, kept aside. No peeking until section 3!
adata.obs['atlas_qc'] = adata.obs['qc_status']
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

md("""Now look at the distributions of the doublet scores and ribosomal fraction.
High doublet scores and very high ribosomal fractions are classic signs of
low-quality nuclei - keep an eye on the right-hand tails. *(We deliberately do not
show the atlas's keep/drop answer yet - you will pick your own thresholds first.)*""")
code("""def qc_histograms(df):
    metrics = ['doublet_score', 'solo_doublet', 'percent_ribo']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, m in zip(axes, metrics):
        ax.hist(df[m], bins=60, color='#4c72b0')
        ax.set_xlabel(m); ax.set_ylabel('nuclei')
    plt.tight_layout(); plt.show()

qc_histograms(adata.obs)""")

md("""### Where might the low-quality nuclei sit on the UMAP?

Colour the **pre-filter UMAP** (`X_umap_prefilter`, computed on *all* nuclei) by
coarse cell class (`class_coarse`, defined for every nucleus) and by the per-nucleus
**doublet score**. Watch for small fringe islands and regions where the doublet
score lights up - those are candidate low-quality territories. After you set your own
thresholds you will be cutting regions like these, and *then* we reveal the atlas's
decision.""")
code("""with plt.rc_context({'figure.figsize': (7, 7)}):
    sc.pl.embedding(adata, basis='X_umap_prefilter',
                    color=['class_coarse', 'doublet_score'],
                    title=['coarse cell class', 'doublet score'],
                    size=8, cmap='viridis', ncols=2, wspace=0.3)""")

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
    'Non-Neurons':   (0.0, 5.0),   # wide open - tighten me!
    'neurons':       (0.0, 5.0),   # GABAergic / Glutamatergic / Cholinergic
    'Motor Neurons': (0.0, 5.0),   # motor neurons
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
**remove**. Tightening or loosening the thresholds in the cell above moves the
boundary of the orange region. Re-run the constraints with different numbers - then
scroll down to the reveal to see how the atlas actually drew the line.""")
code("""adata.obs['your_qc'] = np.where(adata.obs['keeper_cells'], 'keep', 'remove')
with plt.rc_context({'figure.figsize': (8, 8)}):
    sc.pl.embedding(adata, basis='X_umap_prefilter', color='your_qc',
                    title='your QC decision (pre-filter UMAP)',
                    palette={'keep': '#2c7fb8', 'remove': '#d95f0e'}, size=8)""")

md("""### The reveal: what did the atlas actually filter?

Now that *you* have committed to thresholds, we can finally colour by the atlas's
ground-truth decision (`atlas_qc`). The histograms split each QC metric into the
nuclei the atlas **kept** (blue) vs **filtered out** (orange) - notice how the
filtered nuclei pile up in the high-doublet / high-ribo tails. On the pre-filter
UMAP the filtered-out nuclei form their own fringe territory rather than mixing into
the healthy populations. Compare this with your own `keep`/`remove` map above: how
close did you get?""")
code("""def qc_histograms_reveal(df):
    metrics = ['doublet_score', 'solo_doublet', 'percent_ribo']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, m in zip(axes, metrics):
        for status, color in [('passed_qc', '#2c7fb8'), ('filtered_out', '#d95f0e')]:
            ax.hist(df.loc[df['atlas_qc'] == status, m], bins=60, alpha=0.6,
                    label=status, color=color)
        ax.set_xlabel(m); ax.set_ylabel('nuclei'); ax.legend()
    plt.tight_layout(); plt.show()

qc_histograms_reveal(adata.obs)

with plt.rc_context({'figure.figsize': (8, 8)}):
    sc.pl.embedding(adata, basis='X_umap_prefilter', color='atlas_qc',
                    title='atlas QC decision (ground truth)',
                    palette={'passed_qc': '#2c7fb8', 'filtered_out': '#d95f0e'},
                    size=8)""")

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
code("""with plt.rc_context({'figure.figsize': (8, 8)}):
    sc.pl.umap(adata, color='leiden', palette=sc.pl.palettes.godsnot_102,
               legend_loc=None, title='your Leiden clusters')""")

md("""Now colour the same UMAP by **species** and by the reference `Class_V2` and
`Subclass_V2` labels (these are blank/`NaN` for any nuclei without a reference
annotation).""")
code("""with plt.rc_context({'figure.figsize': (6, 6)}):
    sc.pl.umap(adata, color=['species', 'Class_V2', 'Subclass_V2'],
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

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sc.pl.embedding(adata, basis='X_umap_pca', color='species', ax=axes[0],
                show=False, title='PCA on HVGs (not integrated)', size=8)
sc.pl.umap(adata, color='species', ax=axes[1],
           show=False, title='scVI-integrated (used for clustering)', size=8)
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

md("""Finally, let the **data** nominate **conserved** markers - genes that mark a
cell type **in every species**, not just on average. For each reference
`Supergroup_V2` cell type we rank genes (Wilcoxon) **separately within each
species**, then score each gene by its **minimum** marker score across species: a
gene only wins if it is strongly enriched in **human and macaque and mouse**.
A species is **skipped** for a group when that group has **fewer than 20 cells**
there (too few to rank reliably).""")
code("""# Conserved markers: rank genes per species, then take the MIN score across
# species so a gene only wins if it marks the cell type in *every* species.
GROUPBY = 'Supergroup_V2'
MIN_CELLS_PER_SPECIES = 20   # skip a species for a group with fewer cells than this

adata_lab = adata[adata.obs[GROUPBY].notna()].copy()
adata_lab.obs[GROUPBY] = adata_lab.obs[GROUPBY].cat.remove_unused_categories()

# Per-species marker scores (one Wilcoxon ranking within each species), restricted
# to the groups that pass the cell-count threshold in that species.
species_scores = {}
for sp in adata_lab.obs['species'].unique():
    a_sp = adata_lab[adata_lab.obs['species'] == sp].copy()
    a_sp.obs[GROUPBY] = a_sp.obs[GROUPBY].cat.remove_unused_categories()
    counts = a_sp.obs[GROUPBY].value_counts()
    valid = counts.index[counts >= MIN_CELLS_PER_SPECIES].tolist()
    if len(valid) < 2:
        continue  # need at least two groups to rank one against the rest
    sc.tl.rank_genes_groups(a_sp, GROUPBY, groups=valid, method='wilcoxon')
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

# ---- 8. Your filtering vs. the reference -------------------------------------
md("""## 8. Your filtering vs. the reference

Finally, compare **the nuclei you removed in this notebook** with **the nuclei that
were removed in the published, processed object**. Both are drawn on the same
pre-filter UMAP so you can see exactly where the two decisions agree and disagree.
If you tightened or loosened the thresholds in section 3, the left panel will shift
relative to the right one.""")
code("""# Build a tiny AnnData holding just the pre-filter UMAP + both QC labels so we
# can reuse the standard scanpy plotting (the working `adata` is now filtered).
qc_ad = sc.AnnData(obs=qc_compare[['your_qc', 'atlas_qc']].astype('category').copy())
qc_ad.obsm['X_umap'] = qc_compare[['umap1', 'umap2']].to_numpy(float)
qc_ad.obs['your_qc']  = qc_ad.obs['your_qc'].cat.set_categories(['keep', 'remove'])
qc_ad.obs['atlas_qc'] = qc_ad.obs['atlas_qc'].cat.set_categories(['passed_qc', 'filtered_out'])
qc_ad.uns['your_qc_colors']  = ['#2c7fb8', '#d95f0e']
qc_ad.uns['atlas_qc_colors'] = ['#2c7fb8', '#d95f0e']

with plt.rc_context({'figure.figsize': (6, 6)}):
    sc.pl.umap(qc_ad, color=['your_qc', 'atlas_qc'], size=8, wspace=0.3,
               title=['your filtering (this notebook)',
                      'reference filtering (processed h5ad)'])

# how often do the two decisions agree?
agree = ((qc_compare['your_qc'] == 'keep') ==
         (qc_compare['atlas_qc'] == 'passed_qc'))
print(f'agreement: {agree.mean():.1%} of {len(qc_compare):,} nuclei')
print(pd.crosstab(qc_compare['your_qc'], qc_compare['atlas_qc']))""")

# ---- 9. Export for cellxgene --------------------------------------------------
md("""## 9. Export for cellxgene

Write the processed object to `RESULTS_DIR` (`/results/`) so it can be opened in
**cellxgene** for interactive exploration. (Your instructor will provide the
cellxgene capsule link.)""")
code("""out_processed = f'{RESULTS_DIR}/SpC_workshop_snRNA_session1_processed.h5ad'
adata.write(out_processed)
print('Saved', out_processed)""")

md("""### Make a cellxgene-safe copy

cellxgene is strict about its schema (numeric or categorical `obs`/`var` only, no
NaNs in numeric columns, only `*_colors` palettes in `uns`, float `X`). The
`make_safe_h5ad.py` utility in `../processing/` coerces an object into that shape.
We run it here on a throwaway copy of the object we just wrote (the script prunes
`uns` **in place**, so we never point it at a file we want to keep) and write the
cleaned result to `/results/`.""")
code("""import shutil, subprocess, sys, os

src    = f'{RESULTS_DIR}/SpC_workshop_snRNA_session1_processed.h5ad'
safe   = f'{RESULTS_DIR}/SpC_workshop_snRNA_session1_cellxgene.h5ad'
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
a small spatial example (`SpC_workshop_spatial_example.h5ad`) holding the
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
spatial = sc.read_h5ad(f'{DATA_DIR}/SpC_workshop_spatial_example.h5ad')
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
            "regenerate SpC_workshop_spatial_example.h5ad by running "
            "processing/02_build_spatial_example.py")
    spatial_obs = spatial_obs.copy()
    spatial_obs['_plot_x'] = np.asarray(_coords)[:, 0].astype(float)
    spatial_obs['_plot_y'] = np.asarray(_coords)[:, 1].astype(float)
if spatial_obs['species'].nunique() < 3:
    print('WARNING: spatial object has <3 species - it looks stale. Re-run '
          'processing/02_build_spatial_example.py for the full cross-species panel.')

nn_obs = pd.read_csv(f'{DATA_DIR}/SpC_workshop_spatial_nn_overlay.tsv.gz',
                     sep='\\t', index_col=0, compression='gzip')

with open(f'{DATA_DIR}/SpC_workshop_spatial_meta.json') as f:
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

# ---- 11. Interactive atlases --------------------------------------------------
md("""## 11. Explore the data interactively

**cellxgene** - the cellxgene-safe object you just saved
(`SpC_workshop_snRNA_session1_cellxgene.h5ad`) and the spatial example
(`SpC_workshop_spatial_example.h5ad`) can be browsed interactively:
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
