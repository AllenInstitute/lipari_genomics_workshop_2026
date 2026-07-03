"""Generate the Session-2 *literature cell types* student notebook.

This notebook **precedes** ``session2_webportal_mapping_spatial.ipynb``. Before we
map the whole spinal taxonomy onto the mouse brain, we ground ourselves in a few
**named cell types from the classic spinal-cord literature** and hunt for them in
*our* atlas.

To keep it a puzzle, the descriptive ``Group_V2`` names (e.g. ``Sp2i NMU TAC3
Glut``) are **hidden**: every group is shown only as an anonymous **`Group`** ID -
its `Subclass_V2` plus a number (e.g. `Glut-D 7`). Students find each literature
cell type by ranking the anonymous groups on marker combinations and exploring
them on the snRNA UMAP and the example spatial sections; the real name is only
uncovered in the per-target **reveal**.

Run once to (re)write the .ipynb::

    python _build_literature_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


# ---- Title --------------------------------------------------------------------
md("""# Session 2 - Find the textbook cell types in the atlas

**Lipari Genomics Workshop 2026 - Interactive Part 2A (before the whole-brain mapping)**

In Session 1 you built a spinal-cord (SpC) taxonomy of cell types. Each one has a
descriptive `Group_V2` name (like `Sp2i NMU TAC3 Glut`) - but those names basically
*spell out the answer*, so for this exercise **we hide them**. Instead every group
appears as an anonymous **`Group` ID**: its subclass plus a number, e.g. `Glut-D 7`,
`GABA-V 3`. Your job is to work out which anonymous group is which **famous
spinal-cord cell type from the literature** - and only then uncover its real name.

For four classic cell types we give you:

1. a **description** of the cell type and what it *does*,
2. the **marker genes** the field uses to recognise it, and
3. a **toolkit** to test candidates in *our* data.

For each one you will:

- **rank** the anonymous groups by a marker combination (which `Group` scores highest?),
- **look at the markers** on the snRNA **UMAP** and in a **dotplot** across candidates,
- **explore a group** - paint it on the UMAP *and* in space, and repeat for any group,
- then **reveal** its true `Group_V2` name.

The four targets:

| # | Literature cell type | Markers you are given | Where it lives |
|---|---|---|---|
| 1 | **CSF-contacting neurons** | `PKD2L1` (a single marker!) | around the **central canal**, lamina X |
| 2 | **Dorsal-horn itch neurons** | `TAC3`, `NMU` (+ `GRP`/`GRPR`) | superficial dorsal horn, **lamina 2** |
| 3 | **Renshaw cells** | `CHRNA5` (+ `SYNPR`, `CCBE1`) | **ventral horn**, next to motor neurons |
| 4 | **Ascending nociceptive projection neurons** | `PHOX2A`, `RELN`, `LMX1B` | lamina I + deep dorsal horn (anterolateral system) |

> Every random seed is fixed, so your rankings and figures match everyone else's.
> Once you have located these known types, the next notebook
> (`session2_webportal_mapping_spatial.ipynb`) maps the *whole* taxonomy onto the
> mouse brain.""")

# ---- 0. Setup -----------------------------------------------------------------
md("""## 0. Setup

Import libraries, **fix all random seeds**, and load the two datasets:

- the Session-1 **snRNA** object (`SpC_workshop_snRNA.h5ad`) - raw counts,
  which we normalise/log exactly as in Session 1, plus the carried atlas **UMAP**, and
- the **example spatial** object (`SpC_workshop_spatial_example.h5ad`) -
  three representative cross-species sections (human, macaque, mouse), each nucleus
  carrying its cell-type label, its Rexed lamina, and a 947-gene spatial panel.

The pre-built workshop artifacts are distributed **read-only** in `DATA_DIR`
(`/data/lipari_workshop`).""")
code("""import os, json, random, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
sc.settings.seed = SEED
sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=90, frameon=False, figsize=(6, 6))

DATA_DIR     = '/data/lipari_workshop'   # read-only pre-built workshop artifacts
SNRNA_H5AD   = f'{DATA_DIR}/SpC_workshop_snRNA.h5ad'
SPATIAL_H5AD = f'{DATA_DIR}/SpC_workshop_spatial_example.h5ad'
SPATIAL_NN   = f'{DATA_DIR}/SpC_workshop_spatial_nn_overlay.tsv.gz'
SPATIAL_META = f'{DATA_DIR}/SpC_workshop_spatial_meta.json'
UMAP_BASIS   = 'X_umap_atlas'      # carried atlas UMAP embedding
SECRET       = 'Group_V2'          # the descriptive name we are HIDING
GROUPBY      = 'Group'             # the anonymous ID students work with
print('scanpy', sc.__version__)""")

md("""Load and **normalise** the snRNA object (library-size to 10,000 counts + `log1p`,
same as Session 1), keeping only the nuclei that carry a reference cell-type label.
`X` is now log-normalised expression, ready for ranking, dotplots and UMAPs.""")
code("""adata = sc.read_h5ad(SNRNA_H5AD)
print('pre normalization max value:', adata.X.max())  # an integer > 20 means this is raw counts, float <20 is normalized log counts
adata.layers['counts'] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print('post normalization max value:', adata.X.max())  # an integer > 20 means this is raw counts, float <20 is normalized log counts

adata = adata[adata.obs[SECRET].notna()].copy()
adata.obs[SECRET] = adata.obs[SECRET].cat.remove_unused_categories()
adata.obs['Subclass_V2'] = adata.obs['Subclass_V2'].cat.remove_unused_categories()
print(f'{adata.n_obs:,} labelled nuclei x {adata.n_vars:,} genes; '
      f'{adata.obs[SECRET].nunique()} cell-type groups')""")

md("""Load the **example spatial** object and its two companion artifacts (the
non-neuronal grey-background overlay and the metadata JSON with per-section crop
bounds + the canonical cell-type colours). These are the same three representative
sections you met at the end of Session 1.""")
code("""spatial = sc.read_h5ad(SPATIAL_H5AD)
spatial.obs[SECRET] = spatial.obs[SECRET].astype(str)

nn_obs = pd.read_csv(SPATIAL_NN, sep='\\t', index_col=0, compression='gzip')
with open(SPATIAL_META) as f:
    meta = json.load(f)
REP_SECTIONS = meta['representative_sections']            # species -> section id
REP_CROP     = {k: tuple(v) for k, v in meta['rep_crop'].items()}
SECRET_COLOR = meta['group_color']                        # Group_V2 -> hex

print(f'{spatial.n_obs:,} spatial neurons across {len(REP_SECTIONS)} sections '
      f'({", ".join(REP_SECTIONS)})')
print(f'spatial panel: {spatial.n_vars} genes')""")

# ---- 0b. Anonymise ------------------------------------------------------------
md("""### Anonymise the cell types

Here is the trick that turns this into a puzzle. We replace each descriptive
`Group_V2` name with an anonymous **`Group` ID** = its **subclass + a number**,
numbered within each subclass in the taxonomy's own order. So `Sp2i NMU TAC3 Glut`
might become `Glut-D 7` - you can see it is a **dorsal excitatory** neuron (subclass
`Glut-D`), but nothing about *which* one.

We build the hidden `Group_V2 <-> Group` lookup (`SECRET2ANON` / `ANON2SECRET`), add
the anonymous `Group` column to **both** datasets, and carry the colours across. The
lookup itself is kept quiet - the real names are only printed in the reveals.""")
code("""# subclass of each secret group, in the taxonomy's category order
secret_order = list(adata.obs[SECRET].cat.categories)
grp_subclass = (adata.obs.groupby(SECRET, observed=True)['Subclass_V2']
                .agg(lambda s: s.astype(str).mode().iloc[0]))

SECRET2ANON, _counter = {}, {}
for g in secret_order:
    sub = grp_subclass[g]
    _counter[sub] = _counter.get(sub, 0) + 1
    SECRET2ANON[g] = f'{sub} {_counter[sub]}'
ANON2SECRET = {v: k for k, v in SECRET2ANON.items()}
ANON_ORDER  = [SECRET2ANON[g] for g in secret_order]
ANON_COLOR  = {SECRET2ANON[g]: SECRET_COLOR.get(g, '#888888') for g in secret_order}

# add the anonymous ID to both datasets (categorical, in subclass order)
for ad in (adata, spatial):
    ad.obs[GROUPBY] = pd.Categorical(
        ad.obs[SECRET].astype(str).map(SECRET2ANON),
        categories=[a for a in ANON_ORDER if a in set(ad.obs[SECRET].astype(str).map(SECRET2ANON))])

print(f'anonymised {len(SECRET2ANON)} groups into subclass+number IDs '
      '(real names hidden until the reveals)')
print('e.g. the anonymous IDs in subclass Glut-D:',
      [a for a in ANON_ORDER if a.startswith('Glut-D ')][:6], '...')""")

md("""A quick roster so you know what you can pick from. `list_groups_in_subclass()` lists
the anonymous IDs in any subclass; `describe_group_location()` reports how many nuclei a group
has and where (in space) it sits, **without** revealing its name.""")
code("""def list_groups_in_subclass(subclass):
    return [a for a in ANON_ORDER if a.startswith(f'{subclass} ')]


def describe_group_location(group):
    n = int((adata.obs[GROUPBY] == group).sum())
    lam = (spatial.obs.loc[spatial.obs[GROUPBY] == group, 'rexed_lamina']
           .astype(str).replace('', 'unlabelled').value_counts().head(4))
    print(f'{group}: {n} snRNA nuclei; top spatial laminae -> '
          + (', '.join(f'{k}({v})' for k, v in lam.items()) if len(lam) else 'not in the 3 sections'))


print('subclasses:', sorted(set(a.rsplit(' ', 1)[0] for a in ANON_ORDER)))
print('\\nexcitatory dorsal groups (Glut-D):', list_groups_in_subclass('Glut-D'))
describe_group_location(list_groups_in_subclass('Glut-D')[0])""")

# ---- 1. Toolkit ---------------------------------------------------------------
md("""## 1. Your detective toolkit

A handful of reusable helpers do all the work; every cell-type section just calls
them with different markers. They **all speak the anonymous `Group` ID** - the real
names stay hidden.

- **`rank_groups_by_markers(genes)`** - the workhorse. Averages each marker's expression within
  every group, **z-scores each gene across groups** (so a high-baseline gene cannot
  dominate), and returns the groups sorted by their **mean z-score** - i.e. *which
  cell type best matches this marker combination*.
- **`plot_genes_on_umap(genes)`** - paint one or more marker genes on the snRNA **UMAP**.
- **`plot_marker_dotplot(genes, groups)`** - dotplot of the markers across a short list of
  candidate groups (dot **colour** = scaled mean expr., **size** = fraction expressing).
- **`explore_groups_umap_and_space(groups)`** - the explorer: highlight one or more groups **on the
  UMAP and in the spatial sections at once**. Change the group and re-run to browse.
- **`plot_gene_in_space(gene)`** - colour every spatial neuron by a panel gene's expression.""")
code("""def rank_groups_by_markers(genes, top=12, plot=True):
    \"\"\"Rank anonymous groups by a marker combination (mean cross-group z-score).\"\"\"
    genes = [g for g in genes if g in adata.var_names]
    X = adata[:, genes].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    df = pd.DataFrame(X, columns=genes)
    df['g'] = adata.obs[GROUPBY].astype(str).values
    grp_mean = df.groupby('g').mean()
    z = (grp_mean - grp_mean.mean()) / grp_mean.std(ddof=0).replace(0, np.nan)
    score = z.mean(axis=1).sort_values(ascending=False)
    if plot:
        sub = score.head(top)[::-1]
        fig, ax = plt.subplots(figsize=(6, 0.34 * len(sub) + 1))
        ax.barh(range(len(sub)), sub.values,
                color=[ANON_COLOR.get(g, '#888888') for g in sub.index])
        ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub.index, fontsize=9)
        ax.set_xlabel('mean marker z-score across groups')
        ax.set_title('Groups ranked by ' + ' + '.join(genes), fontsize=10)
        plt.tight_layout(); plt.show()
    return score


def plot_genes_on_umap(genes):
    \"\"\"Marker-gene expression on the snRNA UMAP (one panel per gene).\"\"\"
    genes = [g for g in genes if g in adata.var_names]
    with plt.rc_context({'figure.figsize': (4.2, 4.2)}):
        sc.pl.embedding(adata, basis=UMAP_BASIS, color=genes, cmap='viridis',
                        ncols=min(4, len(genes)), size=8, wspace=0.25, frameon=False)


def plot_marker_dotplot(genes, groups, title=None):
    \"\"\"Dotplot of markers across a short list of candidate groups.\"\"\"
    genes = [g for g in genes if g in adata.var_names]
    groups = [groups] if isinstance(groups, str) else list(groups)
    sub = adata[adata.obs[GROUPBY].isin(groups)].copy()
    sub.obs[GROUPBY] = pd.Categorical(sub.obs[GROUPBY].astype(str), categories=groups)
    sc.pl.dotplot(sub, genes, groupby=GROUPBY, standard_scale='var', cmap='viridis',
                  title=title, figsize=(0.55 * len(genes) + 3, 0.4 * len(groups) + 1))""")

md("""The **spatial** helpers reuse the Session-1 montage: three sections side by side,
non-neuronal nuclei as faint grey background, each section cropped to the grey matter
with a 500 um bar. `explore_groups_umap_and_space` combines the UMAP highlight and the spatial montage
so a single call shows *both* views of whatever group you pick.""")
code("""from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def _section_frame(species):
    sec = REP_SECTIONS[species]
    neu = spatial.obs[spatial.obs['_section'].astype(str) == sec].copy()
    nn = nn_obs[nn_obs['_section'].astype(str) == sec]
    return neu, nn, REP_CROP.get(species)


def _finish_axis(ax, crop, title):
    ax.set_aspect('equal'); ax.invert_yaxis()
    if crop is not None:
        x0, x1, ytop, ybot = crop
        ax.set_xlim(x0, x1); ax.set_ylim(ybot, ytop)
    ax.set_title(title, fontsize=9); ax.set_axis_off()
    ax.add_artist(AnchoredSizeBar(ax.transData, 500, '500 um', loc='lower right',
                                  pad=0.3, color='black', frameon=False,
                                  size_vertical=20,
                                  fontproperties=FontProperties(size=7)))


def highlight_groups_on_umap(groups, ax=None, title=None):
    \"\"\"Highlight one or more anonymous groups on the snRNA UMAP; rest grey.\"\"\"
    groups = [groups] if isinstance(groups, str) else list(groups)
    U = np.asarray(adata.obsm[UMAP_BASIS])
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6, 6))
    sel = adata.obs[GROUPBY].isin(groups).values
    ax.scatter(U[~sel, 0], U[~sel, 1], s=3, c='#e0e0e0', linewidths=0, rasterized=True)
    for g in groups:
        m = (adata.obs[GROUPBY] == g).values
        ax.scatter(U[m, 0], U[m, 1], s=12, c=[ANON_COLOR.get(g, '#d62728')],
                   linewidths=0, label=f'{g} (n={int(m.sum())})', rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title or 'snRNA UMAP', fontsize=11)
    ax.legend(fontsize=8, markerscale=2, loc='best', frameon=False)
    if own:
        plt.show()


def highlight_groups_in_space(groups, title=None):
    \"\"\"Highlight one or more anonymous groups across the 3 sections; rest grey.\"\"\"
    groups = [groups] if isinstance(groups, str) else list(groups)
    fig, axes = plt.subplots(1, len(REP_SECTIONS), figsize=(6 * len(REP_SECTIONS), 6))
    for ax, species in zip(np.atleast_1d(axes), REP_SECTIONS):
        neu, nn, crop = _section_frame(species)
        if len(nn):
            ax.scatter(nn['_plot_x'], nn['_plot_y'], s=3, c='#c9c9c9',
                       linewidths=0, alpha=0.25, rasterized=True)
        other = neu[~neu[GROUPBY].isin(groups)]
        ax.scatter(other['_plot_x'], other['_plot_y'], s=5, c='#dcdcdc',
                   linewidths=0, alpha=0.5, rasterized=True)
        n_hit = 0
        for g in groups:
            m = neu[GROUPBY] == g
            n_hit += int(m.sum())
            ax.scatter(neu.loc[m, '_plot_x'], neu.loc[m, '_plot_y'], s=22,
                       c=[ANON_COLOR.get(g, '#d62728')], linewidths=0.3,
                       edgecolors='black', alpha=0.95, rasterized=True)
        _finish_axis(ax, crop, f'{species.capitalize()}  ({n_hit} cells)')
    handles = [mpatches.Patch(color=ANON_COLOR.get(g, '#d62728'), label=g) for g in groups]
    fig.legend(handles=handles, loc='lower center', ncol=min(len(groups), 3),
               fontsize=9, bbox_to_anchor=(0.5, -0.04), frameon=False)
    if title:
        fig.suptitle(title, fontsize=13)
    plt.show()


def explore_groups_umap_and_space(groups, title=None):
    \"\"\"One call, two views: highlight the group(s) on the snRNA UMAP AND in space.\"\"\"
    groups = [groups] if isinstance(groups, str) else list(groups)
    highlight_groups_on_umap(groups, title=(title or 'snRNA UMAP') + ': ' + ', '.join(groups))
    highlight_groups_in_space(groups, title=(title or 'In space') + ': ' + ', '.join(groups))


def plot_gene_in_space(gene):
    \"\"\"Colour every spatial neuron by one gene's expression (panel genes only).\"\"\"
    if gene not in spatial.var_names:
        print(f'\"{gene}\" is not in the {spatial.n_vars}-gene spatial panel - '
              'use explore_groups_umap_and_space() to locate the cell type by identity instead.')
        return
    col = spatial[:, gene].X
    col = col.toarray().ravel() if sp.issparse(col) else np.asarray(col).ravel()
    vmax = np.percentile(col[col > 0], 98) if (col > 0).any() else 1.0
    spatial.obs['_expr'] = col
    fig, axes = plt.subplots(1, len(REP_SECTIONS), figsize=(6 * len(REP_SECTIONS), 6))
    sctr = None
    for ax, species in zip(np.atleast_1d(axes), REP_SECTIONS):
        neu, nn, crop = _section_frame(species)
        if len(nn):
            ax.scatter(nn['_plot_x'], nn['_plot_y'], s=3, c='#ededed',
                       linewidths=0, alpha=0.3, rasterized=True)
        order = neu['_expr'].argsort()
        sctr = ax.scatter(neu['_plot_x'].values[order], neu['_plot_y'].values[order],
                          s=14, c=neu['_expr'].values[order], cmap='viridis',
                          vmin=0, vmax=vmax, linewidths=0, rasterized=True)
        _finish_axis(ax, crop, species.capitalize())
    fig.colorbar(sctr, ax=np.atleast_1d(axes).tolist(), shrink=0.5,
                 label=f'{gene} (log-norm)')
    fig.suptitle(f'{gene} expression in space', fontsize=13)
    plt.show()""")

# ---- 1b. Orientation ----------------------------------------------------------
md("""### Get oriented

Two orientation views before we start hunting.

**(a) The snRNA UMAP**, coloured by **subclass** (the coarse identities that make up
the anonymous IDs) - so you can see the excitatory / inhibitory / motor-neuron
territories the groups live in.

**(b) The Rexed laminae** present in the spatial sections - the classic dorsal-to-
ventral layering (I-X). Superficial laminae 1-2 are dorsal (top); motor-neuron lamina
9 is ventral (bottom). Each target should appear where its anatomy says it should.""")
code("""with plt.rc_context({'figure.figsize': (7, 6)}):
    sc.pl.embedding(adata, basis=UMAP_BASIS, color='Subclass_V2', size=8,
                    title='snRNA UMAP - subclasses', legend_loc='right margin')

lam = spatial.obs['rexed_lamina'].astype(str).replace('', 'unlabelled')
# Preserve the curated dorsal->ventral lamina ordering carried by the categorical.
lam_order = [str(c) or 'unlabelled' for c in spatial.obs['rexed_lamina'].cat.categories] \\
    if hasattr(spatial.obs['rexed_lamina'], 'cat') else sorted(lam.unique())
lam = lam.value_counts().reindex([c for c in lam_order if c in set(lam)]).dropna()
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.bar(lam.index, lam.values, color='#4c72b0')
ax.set_ylabel('# spatial neurons'); ax.set_xlabel('Rexed lamina')
ax.set_title('Rexed laminae represented in the example sections')
plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.show()""")

md("""And here are those **Rexed lamina annotations painted in space** - every neuron in
the three representative sections (human, macaque, mouse) coloured by its lamina, over
the faint non-neuronal grey matter. This is the anatomical ground truth you will match
each target against: the dorsal-to-ventral banding you see here (superficial laminae at
the top, motor-neuron lamina 9 at the bottom) mirrors the **spinal-domain schematic**
below. Every `Group_V2` name starts with one of these domain prefixes (`Sp1`, `Sp2o`,
`Sp2i`, ... `Sp8`, `Sp9`, `Sp10`, plus `LSp`/`5Lx`) - the superficial dorsal horn
(`Sp1`-`Sp3`) up top where the itch neurons live, down to the ventral horn (`Sp8`,
`Sp9`) with the motor neurons and Renshaw cells.

<img src="assets/rexed_laminae.png" width="300" alt="Spinal-cord domains / Rexed laminae schematic">""")
code("""# The lamina palette + dorsal->ventral ordering travel with the object: the
# categorical `rexed_lamina` carries the order, and `uns['rexed_lamina_colors']`
# the matching colours (blank '' laminae are drawn grey as 'unlabelled').
spatial.obs['rexed_lamina'] = spatial.obs['rexed_lamina'].astype('category')
lam_cats = list(spatial.obs['rexed_lamina'].cat.categories)
lam_colors = list(spatial.uns.get('rexed_lamina_colors', []))
if len(lam_colors) != len(lam_cats):
    from scanpy.plotting.palettes import default_20, default_102
    base = default_20 if len(lam_cats) <= 20 else default_102
    lam_colors = [base[i % len(base)] for i in range(len(lam_cats))]

LAM_COLOR = {(c or 'unlabelled'): ('#b8b8b8' if c == '' else col)
             for c, col in zip(lam_cats, lam_colors)}
lam_levels = [c or 'unlabelled' for c in lam_cats]

fig, axes = plt.subplots(1, len(REP_SECTIONS), figsize=(7 * len(REP_SECTIONS), 7))
for ax, species in zip(np.atleast_1d(axes), REP_SECTIONS):
    neu, nn, crop = _section_frame(species)
    if len(nn):
        ax.scatter(nn['_plot_x'], nn['_plot_y'], s=3, c='#ededed',
                   linewidths=0, alpha=0.3, rasterized=True)
    lam = neu['rexed_lamina'].astype(str).replace('', 'unlabelled')
    for c in lam_levels:
        m = (lam == c).values
        if m.any():
            ax.scatter(neu['_plot_x'].values[m], neu['_plot_y'].values[m], s=8,
                       c=[LAM_COLOR[c]], linewidths=0, rasterized=True)
    _finish_axis(ax, crop, species.capitalize())
handles = [mpatches.Patch(color=LAM_COLOR[c], label=c) for c in lam_levels]
fig.legend(handles=handles, loc='center right', fontsize=8, frameon=False,
           title='Rexed lamina')
fig.suptitle('Rexed lamina annotations across the 3 sections', fontsize=13)
plt.show()""")

md("""**(c) The full atlas map.** A **very large** snRNA UMAP with every anonymous
`Group` ID written directly **on the data**, so you can see the whole taxonomy at once
and read off where each group sits. Zoom in to pick out neighbours; you will come back
to this map as you hunt for each literature cell type.""")
code("""with plt.rc_context({'figure.figsize': (24, 24)}):
    sc.pl.embedding(adata, basis=UMAP_BASIS, color=GROUPBY, size=10,
                    legend_loc='on data', legend_fontsize=8,
                    legend_fontoutline=2, palette=ANON_COLOR,
                    title='snRNA UMAP - all groups (anonymous IDs)')""")

# ---- 1c. Free explorer --------------------------------------------------------
md("""### Explore any group you like

This is the tool you will reuse throughout: set `GROUP` to **any** anonymous ID and
run the cell to see that group on the **UMAP** and in **space** side by side. Change
the string and re-run to browse the whole taxonomy - e.g. try a few `Glut-D ...`
(dorsal excitatory) IDs, then a `Chol ...` motor-neuron ID, and watch the spatial dot
jump from the dorsal horn to the ventral horn.""")
code("""# >>> Change this to any anonymous Group ID and re-run to explore. <<<
GROUP = 'sMN 2'          # a common, spatially-present example (a lamina-9 motor-neuron pool)
describe_group_location(GROUP)
explore_groups_umap_and_space(GROUP)""")

# ============================================================================
# 2. CSF-CONTACTING NEURONS
# ============================================================================
md("""## 2. Target 1 - **cerebrospinal-fluid-contacting neurons** (`PKD2L1`)

**The cell type.** Ringing the **central canal** (Rexed **lamina X**) sits one of the
most unusual neurons in the cord: the **cerebrospinal-fluid-contacting neuron**
(**CSF-cN**). Each one extends an apical **microvillar/ciliary bud** through the
ependyma and *directly into the cerebrospinal fluid*, where it works as a **polymodal
intraspinal sensor** - detecting **CSF pH / CO2** and **mechanical** deformation as the
spine bends - via the `PKD2L1` channel. They are **GABAergic**, evolutionarily
ancient, and feed back onto locomotor and postural circuits (in fish and mouse they
help tune swimming/spinal curvature; the pathway is even implicated in idiopathic
scoliosis). A neuron that tastes the spinal fluid and senses your posture from the
inside - a great one to start with, because it is defined by essentially a **single
gene**.

**How to recognise it.** The transient-receptor-potential channel **`PKD2L1`** is
almost perfectly specific to these cells - one of the cleanest single-marker
identities in the whole cord. They are inhibitory (`GAD1`/`GAD2`+, `SLC32A1`+, and
`SLC17A6`-negative) and carry ciliary / microvillar genes (`MYO3B`, `PACRG`, `ESPN`).
In space they hug the **central canal (lamina X)** and nowhere else.

**Step 1 - rank the groups.** Here you barely need a combination: one marker will do.""")
code("""# A single marker is enough - PKD2L1 is almost unique to these cells.
CSF_MARKERS = ['PKD2L1']
csf_rank = rank_groups_by_markers(CSF_MARKERS, top=12)
print('Top candidates:'); print(csf_rank.head(4).round(2))""")

md("""**Step 2 - see the marker.** One group towers over every other on `PKD2L1`. The
supporting panel confirms the identity: ciliary/microvillar genes (`MYO3B`, `PACRG`,
`ESPN`) and the inhibitory marker `GAD1` are on, while the excitatory marker `SLC17A6`
is **off** - these are not glutamatergic neurons.""")
code("""CSF_SUPPORT = ['PKD2L1', 'MYO3B', 'PACRG', 'ESPN', 'GAD1', 'SLC17A6']
plot_genes_on_umap(CSF_SUPPORT)
csf_candidates = csf_rank.head(6).index.tolist()
plot_marker_dotplot(CSF_SUPPORT, csf_candidates, title='CSF-cN marker dotplot')""")

md("""**Step 3 - your call.** Set `MY_CSF_GROUP` to the top hit and explore it. On the
UMAP it is a small, isolated island; in space it should sit right on the **central
canal (lamina X)** - dead centre of the cord, unlike every other target today.""")
code("""# >>> EDIT THIS to your best guess (copy the top ID from the ranking) <<<
MY_CSF_GROUP = csf_rank.index[0]
describe_group_location(MY_CSF_GROUP)
explore_groups_umap_and_space(MY_CSF_GROUP, title='CSF-cN candidate')""")

md("""**Step 4 - reveal.** The CSF-contacting neurons are **`CSF-cN PKD2L1 GABA-Gly`** -
they even get their **own subclass** (`CSF-cN`), because nothing else in the cord looks
like them. `PKD2L1` is not in the spatial panel, but highlighting the group in space
shows every cell pinned to the **central canal (lamina X)**, confirming the anatomy.""")
code("""CSF_ANSWER = 'CSF-cN PKD2L1 GABA-Gly'
csf_anon = SECRET2ANON[CSF_ANSWER]
print(f'  {csf_anon:>12}  =  {CSF_ANSWER}')
highlight_groups_in_space(csf_anon,
        title='CSF-contacting neurons (reveal): central canal / lamina X')""")

# ============================================================================
# 3. ITCH NEURONS
# ============================================================================
md("""## 3. Target 2 - dorsal-horn **itch** neurons (`TAC3` / `NMU`)

**The cell type.** The superficial dorsal horn (Rexed **lamina 1-2**) contains a
dedicated **pruritoceptive (itch)** microcircuit. A well-studied excitatory
population relays itch centrally through the **gastrin-releasing peptide -> GRPR**
pathway (Sun & Chen, *Nature* 2007), and a parallel arm using **neuromedin U (NMU)**
drives a distinct component of itch/mechanical-itch signalling. These are small,
**glutamatergic** interneurons packed into the outer/inner lamina 2 (2o/2i) - exactly
where itch-selective primary afferents terminate.

**How to recognise it.** Look for a group that co-expresses the itch-relay peptides
`TAC3` (tachykinin-3, = neurokinin B) and `NMU`, ideally alongside the GRP-GRPR relay
genes `GRP` / `GRPR` and the neuromedin-U receptor `NMUR2`. It should be a **dorsal
excitatory** (`Glut-D`) type and, in space, sit in the **superficial dorsal horn**.

**Step 1 - rank the groups.** Which anonymous `Group` best matches the itch marker set?""")
code("""ITCH_MARKERS = ['TAC3', 'NMU', 'GRP', 'GRPR', 'NMUR2']
itch_rank = rank_groups_by_markers(ITCH_MARKERS, top=12)
print('Top candidates:'); print(itch_rank.head(4).round(2))""")

md("""**Step 2 - see the markers.** Paint the itch markers on the snRNA UMAP (they
should light up the *same* small territory) and compare the top-ranked candidates in
a dotplot - the winner shows large, bright dots for `TAC3` **and** `NMU`.""")
code("""plot_genes_on_umap(ITCH_MARKERS)
itch_candidates = itch_rank.head(6).index.tolist()
plot_marker_dotplot(ITCH_MARKERS, itch_candidates, title='Itch-marker dotplot (top candidates)')""")

md("""**Step 3 - your call.** Set `MY_ITCH_GROUP` to the anonymous ID you think is the
itch population (copy it from the ranking), then explore it. On the UMAP it is a tight
island; in space it should hug the **superficial dorsal horn** (top of the grey
matter). Change the ID and re-run to compare candidates.""")
code("""# >>> EDIT THIS to your best guess (copy an ID from the ranking) <<<
MY_ITCH_GROUP = itch_rank.index[0]
describe_group_location(MY_ITCH_GROUP)
explore_groups_umap_and_space(MY_ITCH_GROUP, title='Itch candidate')""")

md("""**Step 4 - reveal.** Uncover the real names. The itch neurons are the paired
lamina-2 groups **`Sp2i NMU TAC3 Glut`** and **`Sp2-3 TAC3 NMU Glut`** - the two that
topped your ranking. Painted together they tile the superficial dorsal horn across all
three species, and `TAC3` (in the spatial panel) concentrates in the same band.""")
code("""ITCH_ANSWER = ['Sp2i NMU TAC3 Glut', 'Sp2-3 TAC3 NMU Glut']
for g in ITCH_ANSWER:
    print(f'  {SECRET2ANON[g]:>12}  =  {g}')
itch_anon = [SECRET2ANON[g] for g in ITCH_ANSWER]
highlight_groups_in_space(itch_anon, title='Itch neurons (reveal): the lamina-2 TAC3/NMU groups')
plot_gene_in_space('TAC3')

_itch = spatial.obs[spatial.obs[GROUPBY].isin(itch_anon)]
print('Rexed laminae of the itch groups:')
print(_itch['rexed_lamina'].astype(str).replace('', 'unlabelled').value_counts())""")

# ============================================================================
# 4. RENSHAW CELLS
# ============================================================================
md("""## 4. Target 3 - **Renshaw cells** (`CHRNA5` / `SYNPR` / `CCBE1`)

**The cell type.** Down in the **ventral horn** live the **Renshaw cells** - the
inhibitory interneurons Renshaw described in 1941 as the substrate of **recurrent
inhibition** of motor neurons. A motor neuron's axon sends a collateral onto Renshaw
cells, which fire back inhibition onto that same motor pool - a negative-feedback
brake on motor output. Because their excitatory drive is the motor neuron's own
**cholinergic** collateral, Renshaw cells are studded with **nicotinic acetylcholine
receptors** (notably `CHRNA5`), and they sit in **ventral lamina VII/VIII**, right
next to the lamina-IX motor neurons.

**How to recognise it.** Look for the nicotinic-receptor subunit `CHRNA5` together
with `SYNPR` (synaptoporin), `CCBE1`, and the classic Renshaw markers calbindin
`CALB1` / `CHRNA2`. In space it should sit **ventrally**, among/below the motor
neurons - the opposite end of the cord from the itch neurons. As a **ventral
inhibitory** interneuron it lives in the `GABA-V` subclass - like the CSF-cN it is
inhibitory, but a completely different flavour, and unlike the excitatory `Glut` itch
and projection neurons.

**Step 1 - rank the groups.**""")
code("""RENSHAW_MARKERS = ['CHRNA5', 'SYNPR', 'CCBE1', 'CALB1', 'CHRNA2']
renshaw_rank = rank_groups_by_markers(RENSHAW_MARKERS, top=12)
print('Top candidates:'); print(renshaw_rank.head(4).round(2))""")

md("""**Step 2 - see the markers.** One group should dominate `CHRNA5` on the UMAP and
in the dotplot, standing far above the rest.""")
code("""plot_genes_on_umap(RENSHAW_MARKERS)
renshaw_candidates = renshaw_rank.head(6).index.tolist()
plot_marker_dotplot(RENSHAW_MARKERS, renshaw_candidates, title='Renshaw-marker dotplot')""")

md("""**Step 3 - your call.** Set `MY_RENSHAW_GROUP` and explore it. On the UMAP it is
a small, distinct cluster; in space it should appear **ventrally**, near the motor
neurons - not in the dorsal horn.""")
code("""# >>> EDIT THIS to your best guess (copy an ID from the ranking) <<<
MY_RENSHAW_GROUP = renshaw_rank.index[0]
describe_group_location(MY_RENSHAW_GROUP)
explore_groups_umap_and_space(MY_RENSHAW_GROUP, title='Renshaw candidate')""")

md("""**Step 4 - reveal.** The Renshaw cells are **`Sp8 CHRNA5 GABA-Gly`** - the lone
group defined by the nicotinic subunit `CHRNA5`, sitting in ventral lamina 8 beside
the motor neurons. Both `CHRNA5` and `SYNPR` are in the spatial panel, so we can
confirm the ventral position directly. For contrast we also paint an **alpha
motor-neuron pool** (`aMN ... Chol`) - the cells Renshaw neurons inhibit.""")
code("""RENSHAW_ANSWER = 'Sp8 CHRNA5 GABA-Gly'
renshaw_anon = SECRET2ANON[RENSHAW_ANSWER]
print(f'  {renshaw_anon:>12}  =  {RENSHAW_ANSWER}')
highlight_groups_in_space(renshaw_anon, title='Renshaw cells (reveal)')
plot_gene_in_space('CHRNA5')

# Renshaw cells feed back onto ALPHA motor neurons. Pick the alpha-MN (`aMN`) pool
# that is best represented in the example sections, so it shows up clearly in every
# panel (the first pool in taxonomy order is sparse in the human section).
motor_anon = [SECRET2ANON[g] for g in SECRET2ANON
              if g.startswith('aMN') and SECRET2ANON[g] in set(spatial.obs[GROUPBY])]
_mcount = spatial.obs[GROUPBY].value_counts()
top_motor = max(motor_anon, key=lambda a: _mcount.get(a, 0))
highlight_groups_in_space([renshaw_anon, top_motor],
                title='Renshaw cell next to an alpha motor-neuron pool')""")

# ============================================================================
# 5. ASCENDING PROJECTION NEURONS
# ============================================================================
md("""## 5. Target 4 - ascending **nociceptive projection** neurons (`PHOX2A` / `RELN` / `LMX1B`)

**The cell type.** Pain does not stay in the cord: **projection neurons** of the
**anterolateral system** (ALS) - the spinothalamic and spinoparabrachial tracts -
carry nociceptive signals to the brain. Their cell bodies sit mostly in **lamina I**
(with a deeper lamina V contingent), they are **glutamatergic**, and many express the
substance-P receptor NK1R (`TACR1`). A landmark finding is that the transcription
factor **`PHOX2A`** marks the developmental origin of the anterolateral system in
mouse *and* human (Roome et al., *Nat. Commun.* 2020); the excitatory dorsal-horn
selector **`LMX1B`** and the reelin gene **`RELN`** further label these populations.

**How to recognise it.** A group co-expressing `PHOX2A`, `RELN`, and `LMX1B` (the
projection markers `TACR1` and `GDA` help too). In space it should hug **lamina I /
the deep dorsal horn**, and it is **rarer** than the local interneurons around it.

**The reference study.** Our guide for this target is **Bell et al., *PNAS* 2024**
("Deep sequencing of Phox2a nuclei reveals five classes of anterolateral system
neurons"). They FACS-purified `Phox2a`-lineage nuclei from mouse dorsal horn
(`Phox2a::Cre; Sun1-GFP`) and deep-sequenced them with **Smart-seq2**, resolving the
anterolateral system into **five molecular classes, `ALS1`-`ALS5`**. In this
exercise we *hypothetically could not obtain their raw Smart-seq2 counts*, so rather
than transferring their labels onto our cells we will **use their published marker
genes** to (1) find this projection-neuron family in our own atlas and (2) - for
extra credit - line each `ALS1`-`ALS5` class up with one of our groups.

**Step 1 - rank the groups.**""")
code("""ALS_MARKERS = ['PHOX2A', 'RELN', 'LMX1B', 'TACR1', 'GDA']
als_rank = rank_groups_by_markers(ALS_MARKERS, top=12)
print('Top candidates:'); print(als_rank.head(5).round(2))""")

md("""**Step 2 - see the markers.** On the UMAP notice that several **sibling groups**
share the `PHOX2A` signal - a *family* of related projection-neuron types - rather
than a single island. The dotplot across the top candidates makes the shared
`PHOX2A` / `RELN` / `LMX1B` signature obvious.""")
code("""plot_genes_on_umap(ALS_MARKERS)
als_candidates = als_rank.head(7).index.tolist()
plot_marker_dotplot(ALS_MARKERS, als_candidates, title='Projection-neuron marker dotplot')""")

md("""**Step 3 - your call.** This target is a **family**, so `MY_ALS_GROUPS` is a
list - put the top few `PHOX2A`-high IDs in it. Explore them together: expect a
**sparse** population in the **superficial + deep dorsal horn**, consistent with
lamina I/V projection cells.""")
code("""# >>> EDIT THIS list to your best guesses (copy IDs from the ranking) <<<
MY_ALS_GROUPS = als_rank.head(3).index.tolist()
for g in MY_ALS_GROUPS:
    describe_group_location(g)
explore_groups_umap_and_space(MY_ALS_GROUPS, title='Projection-neuron candidates')""")

md("""**Step 4 - reveal.** The ascending projection neurons are the **`PHOX2A` family**:
`Sp1-5Lx PHOX2A BCL11A/POU6F2/CALCR/CREB5 Glut` (the `Lx` literally denotes the
**lamina I -> deep** span of the anterolateral system). `PHOX2A` is not in the spatial
panel, but the shared selector **`LMX1B`** is - and it marks the dorsal-horn excitatory
territory these cells belong to.""")
code("""ALS_ANSWER = ['Sp1-5Lx PHOX2A BCL11A Glut', 'Sp1-5Lx PHOX2A POU6F2 Glut',
              'Sp1-5Lx PHOX2A CALCR Glut', 'Sp1-5Lx PHOX2A CREB5 Glut']
ALS_ANSWER = [g for g in ALS_ANSWER if g in SECRET2ANON]
for g in ALS_ANSWER:
    print(f'  {SECRET2ANON[g]:>12}  =  {g}')
als_anon = [SECRET2ANON[g] for g in ALS_ANSWER if SECRET2ANON[g] in set(spatial.obs[GROUPBY])]
highlight_groups_in_space(als_anon, title='Ascending projection neurons (reveal): the PHOX2A family')
plot_gene_in_space('LMX1B')""")

# ---- 3b. Extra credit: match ALS1-5 -------------------------------------------
md("""### Extra credit - line up `ALS1`-`ALS5` with our groups

We just uncovered **four** projection-neuron groups (`BCL11A`, `POU6F2`, `CALCR`,
`CREB5`); Bell et al. describe **five** classes. They separate their classes by
**marker genes** (Fig. 1D heatmap) and by **laminar position** (Fig. 2A), and
validate the top markers by RNAscope (Fig. 2B-F). Your challenge: match each
`ALS1`-`ALS5` to one of our four groups.

<img src="assets/pnas.2314213121fig01.jpg" width="640" alt="Bell et al. 2024 Fig. 1 - ALS1-5 UMAP and per-class marker heatmap">

<img src="assets/pnas.2314213121fig02.jpg" width="640" alt="Bell et al. 2024 Fig. 2 - ALS1-5 laminar location and RNAscope validation">

Reading their figures, the discriminating signatures are:

| class | signature markers (Fig. 1D / RNAscope Fig. 2B-F) | laminae (Fig. 2A) |
|---|---|---|
| `ALS1` | `Nmu`+, `Tacr3`, `Npy1r`; **`Cdh12`-** | lamina I (broad, I-IV) |
| `ALS2` | `Cdh12`+, **`Baiap3`+**, `Grik1` | lamina I (superficial) |
| `ALS3` | `Cdh12`+, `Calb1`, `Necab2`; `Baiap3` weak | lamina I (superficial) |
| `ALS4` | `Gpr88`+, **`Bcl11a`**, `Sst`, `Cck`; `Erbb4`- | deep (lamina V / lateral) |
| `ALS5` | `Gpr88`+, **`Erbb4`+**, `Penk`, `Ret`, `Onecut2` | deep / ventral (V-VII, medial) |

**Your job.** Eyeball the dotplot below (their markers across our four groups), then
fill in `MY_ALS_MATCH` with your best guess before running the reveal.""")
code("""# The RNAscope-validated discriminators Bell et al. use to tell the classes apart.
ALS_DISCRIMINATORS = ['NMU', 'CDH12', 'BAIAP3', 'CALB1', 'NECAB2',
                      'BCL11A', 'SST', 'GPR88', 'ERBB4', 'PENK', 'RET']
plot_marker_dotplot(ALS_DISCRIMINATORS, als_anon,
                    title='ALS1-5 discriminators across our 4 projection groups')
for g in ALS_ANSWER:            # laminar position to compare against Fig. 2A
    describe_group_location(SECRET2ANON[g])

# >>> EDIT with your best guess (class -> one of BCL11A/POU6F2/CALCR/CREB5) <<<
MY_ALS_MATCH = {'ALS1': 'CALCR', 'ALS2': 'POU6F2', 'ALS3': 'POU6F2',
                'ALS4': 'BCL11A', 'ALS5': 'CREB5'}""")

md("""**Extra-credit reveal.** Rather than hand-wave, we score each `ALS` signature
against our four groups: average the (per-gene, cross-group z-scored) expression of
its **discriminating** markers - `Cdh12` counts *negatively* for `ALS1`, since that
class is the Nmu+/**Cdh12-** type - and take the best-matching group. This is exactly
`rank_groups_by_markers`, restricted to the four projection groups.""")
code("""# signed discriminators: +1 = marker ON, -1 = marker OFF for that class
ALS_SIGNED = {
    'ALS1': {'NMU': +1, 'CDH12': -1},
    'ALS2': {'CDH12': +1, 'BAIAP3': +1},
    'ALS3': {'CDH12': +1, 'CALB1': +1, 'NECAB2': +1},
    'ALS4': {'BCL11A': +1, 'SST': +1},
    'ALS5': {'PENK': +1, 'RET': +1},
}
_genes = sorted({g for w in ALS_SIGNED.values() for g in w if g in adata.var_names})
_sub = adata[adata.obs[GROUPBY].isin(als_anon)]
_X = _sub[:, _genes].X
_X = _X.toarray() if sp.issparse(_X) else np.asarray(_X)
_gm = (pd.DataFrame(_X, columns=_genes)
       .assign(g=_sub.obs[GROUPBY].astype(str).values)
       .groupby('g').mean().reindex(als_anon))
_z = (_gm - _gm.mean()) / _gm.std(ddof=0).replace(0, np.nan)

short = {SECRET2ANON[g]: g.split('PHOX2A ')[1].replace(' Glut', '') for g in ALS_ANSWER}
print('ALS class  ->  best-matching group   (score)')
for als, w in ALS_SIGNED.items():
    score = sum(sign * _z[g] for g, sign in w.items() if g in _z.columns) / len(w)
    best = score.idxmax()
    mark = 'OK' if MY_ALS_MATCH.get(als) == short[best] else 'reconsider'
    print(f'  {als}   ->   {short[best]:<8}  ({score[best]:+.2f})   your guess: '
          f'{MY_ALS_MATCH.get(als, \"-\"):<8} [{mark}]')""")

md("""**The answer.** Four of our groups carry the five classes, with the two
superficial `Cdh12`+ classes (`ALS2`/`ALS3`) collapsing onto our single `POU6F2`
group:

| Bell et al. class | our `Group_V2` | deciding evidence |
|---|---|---|
| `ALS1` | `Sp1-5Lx PHOX2A CALCR Glut` | highest `Nmu`, lowest `Cdh12` - the Nmu+/Cdh12- lamina-I type |
| `ALS2` + `ALS3` | `Sp1-5Lx PHOX2A POU6F2 Glut` | `Cdh12`+ with `Grik1`/`Calb1`/`Necab2` - the superficial Cdh12+ cells (one group spans their two classes) |
| `ALS4` | `Sp1-5Lx PHOX2A BCL11A Glut` | the only `Bcl11a`-high group; matches ALS4's deep (lamina V) `Bcl11a` marker |
| `ALS5` | `Sp1-5Lx PHOX2A CREB5 Glut` | highest `Penk`/`Ret` - the deep/ventral class |

**Wait - isn't `Nmu` the itch marker from Target 2?** Exactly, and that is the trap.
Our lamina-II **itch** interneurons (`Sp2i NMU TAC3 Glut`, `Sp2-3 TAC3 NMU Glut`)
express *more* `Nmu` than our `CALCR` group - so ranking on `Nmu` **across the whole
atlas** would wrongly land on the Target-2 itch cells, not on `ALS1`. What separates
them is **projection identity**: `ALS1` (like all of Bell et al.'s classes) is
`Phox2a`-lineage and `Tacr1`(NK1R)+/`Tac1`+, whereas the `Nmu` itch interneurons are
`Tacr1`-/`Tac1`- and are **not** `Phox2a`-lineage (they were never in Bell et al.'s
sorted dataset). That is why the scoring is deliberately **restricted to the four
`Phox2a` projection groups** - within that family, `CALCR` is the `Nmu`+/`Cdh12`-
member, so it is our `ALS1`.

**Caveats - why this is only a best guess.** We never mapped the raw data: this is a
*marker-only*, *cross-species* correspondence (Bell et al. is mouse-only), and **5
mouse classes collapse onto 4 multi-species groups**. Two of their markers also
behave differently in *our* data - `Gpr88` is near-zero across all our neurons and
`Erbb4` is broadly expressed - so the call leans on `Nmu`/`Cdh12`/`Bcl11a`/`Penk`/
`Ret`, **not** `Gpr88`/`Erbb4`. A defensible assignment would need their counts and a
proper label-transfer (the whole-brain mapping in the next notebook does exactly
that).""")
code("""# Verify the trap: Nmu is HIGHER in the Target-2 itch interneurons than in our ALS1
# (CALCR), but only the projection neurons are Tacr1(NK1R)+/Tac1+.
_contrast = ['NMU', 'TACR1', 'TAC1', 'TAC3', 'PHOX2A', 'ZFHX3']
_contrast = [g for g in _contrast if g in adata.var_names]
_grp = {'ALS1 = CALCR (projection)': 'Sp1-5Lx PHOX2A CALCR Glut'}
_grp.update({f'{g} (itch interneuron)': g for g in ITCH_ANSWER})
_rows = {}
for label, secret in _grp.items():
    m = (adata.obs[SECRET] == secret).values
    Xc = adata[m, _contrast].X
    Xc = Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)
    _rows[label] = Xc.mean(0)
print(pd.DataFrame(_rows, index=_contrast).round(2))
print('\\n-> Nmu alone points at the itch cells; Tacr1/Tac1 (projection identity) '
      'single out ALS1 = CALCR.')""")

# ---- 6. Recap -----------------------------------------------------------------
md("""## 6. Recap & bridge to the whole-brain mapping

You located four classic spinal-cord cell types in *our* atlas - starting from
**anonymous IDs** and using only marker gene expression (ranking, UMAP, dotplots) and
the example spatial sections:

| Literature cell type | `Group_V2` group(s) | Subclass | Position |
|---|---|---|---|
| CSF-contacting neurons | `CSF-cN PKD2L1 GABA-Gly` | CSF-cN (inhibitory) | central canal (lamina X) |
| Dorsal-horn itch neurons | `Sp2i NMU TAC3 Glut`, `Sp2-3 TAC3 NMU Glut` | Glut-D | superficial dorsal horn (L1-2) |
| Renshaw cells | `Sp8 CHRNA5 GABA-Gly` | GABA-V (inhibitory) | ventral horn (L7-8) |
| Ascending nociceptive projection neurons | `Sp1-5Lx PHOX2A BCL11A/POU6F2/CALCR/CREB5 Glut` | Glut-M | lamina I + deep dorsal horn |

**The workflow you just used** - rank the anonymous groups by a marker combination,
confirm on the UMAP + dotplot, then explore the group in space - is exactly how you
would nominate *any* literature cell type. Try it now with a type of your own: pick
markers, rank, and explore the winner.""")
code("""# >>> Your turn: pick markers for a cell type you know and hunt for its group. <<<
# e.g. preganglionic autonomic (NOS1), proprioceptive relay (choose your own markers)...
MY_MARKERS = ['NOS1']                         # <- edit
my_rank = rank_groups_by_markers([g for g in MY_MARKERS if g in adata.var_names], top=10)
print(my_rank.head(5).round(2))
plot_genes_on_umap(MY_MARKERS)
explore_groups_umap_and_space(my_rank.index[0], title='My candidate')""")

md("""**What's next.** You now know *what* these cell types are and *where* they sit in
the cord. The next notebook,
[`session2_webportal_mapping_spatial.ipynb`](./session2_webportal_mapping_spatial.ipynb),
takes the **whole** taxonomy - including the types you just identified - and maps it
onto the **mouse whole brain** with MapMyCells, then asks *where in the brain* those
spinal cell types find their closest relatives.

---
*snRNA: `SpC_workshop_snRNA.h5ad` (Session 1). Spatial: `SpC_workshop_spatial_example.h5ad`
(3 representative cross-species sections). Selected references: Sun & Chen, Nature 2007
(GRP-GRPR itch relay); Roome et al., Nat. Commun. 2020 (Phox2a and the anterolateral
system); Bell et al., PNAS 2024 (five ALS classes / Phox2a Smart-seq2); Renshaw,
J. Neurophysiol. 1941 (recurrent inhibition).*""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}

OUT = ('/code/lipari_genomics_workshop_2026/session2/notebooks/'
       'session2_literature_cell_types.ipynb')
with open(OUT, 'w') as f:
    nbf.write(nb, f)
print(f'wrote {OUT} with {len(cells)} cells')
