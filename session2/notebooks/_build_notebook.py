"""Generate the Session-2 student notebook (reciprocal mapping + spatial).

Mirrors session1/notebooks/_build_notebook.py: a markdown explanation precedes
every code cell and all RNG seeds are fixed. Run once to (re)write the .ipynb::

    python _build_notebook.py
"""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


# ---- Title --------------------------------------------------------------------
md("""# Session 2 — Reciprocal mapping: spinal cord ↔ mouse whole brain

**Lipari Genomics Workshop 2026 — Interactive Part 2B (Map to whole brain)**

In Session 1 you built a spinal-cord (SpC) cell-type taxonomy. Here you will
relate it to the **Allen mouse whole-brain (AIT21)** atlas with a *reciprocal*
mapping, then ask **where in the mouse brain** the spinal-like cell types live by
projecting onto the ABC mouse-brain **MERFISH spatial atlas**.

1. **Forward** — map our SpC cells onto mouse-WB **subclasses**: *which
   whole-brain cell types did our spinal cord capture?*
2. **Reverse** — map mouse-WB subclasses onto **our consensus spinal cord**:
   *which whole-brain types resemble a spinal type?*
3. **Reciprocal** — keep the pairs that agree **both** ways.
4. **Spatial** — paint the detected / reciprocal subclasses onto the mouse brain.

> All seeds are fixed, and the heavy mapping was run once by the `processing/`
> scripts — here we just **load and explore** the results.""")

# ---- 0. Setup -----------------------------------------------------------------
md("""## 0. Setup

Import libraries, **fix all random seeds**, and point at the pre-computed
artifacts in `/results/` (built by `processing/01`–`04`) and the bundled MERFISH
spatial atlas. `mapping_io.py` carries the reciprocal-mapping helpers.""")
code("""import os, sys, json, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
sc.settings.seed = SEED
sc.settings.verbosity = 1

# Reciprocal-mapping helpers live with the Session-2 processing scripts.
PROC_DIR = '/code/lipari_genomics_workshop_2026/session2/processing'
sys.path.insert(0, PROC_DIR)
import mapping_io as mio

# --- Inputs (built by processing/01-04; do not regenerate live) ----------------
QUERY_H5AD      = '/results/SpC_workshop_snRNA.h5ad'                 # Session-1 query
FORWARD_JSON    = '/results/SpC_workshop_WB_SUBCLASS_MAPPING/hann_results.json'
REVERSE_JSON    = '/results/WB_SUBCLASS_to_SpC_MAPPING/hann_results.json'
RECIPROCAL_CSV  = '/results/reciprocal_best_hits.csv'
SUBCLASS_OVERLAP_CSV   = '/results/reciprocal_subclass_overlap.csv'   # OC reciprocity (subclass)
SUPERTYPE_RECIP_CSV    = '/results/reciprocal_supertype_hits.csv'     # OC reciprocity (supertype)
SUBSAMPLED_ABC_H5AD    = '/results/wb_subsampled_ABC.h5ad'            # student-sized mouse-WB snRNA
SPATIAL_ATLAS   = ('/data/mouse_wb_spatial_tutorial/reference/spatial_atlas/'
                   'C57BL6J-638850-16good-sections-raw-meta.h5ad')

WB_SUBCLASS_LEVEL = 'subclass_label'   # forward result level (mouse WB)
ATLAS_SUBCLASS    = 'subclass'         # atlas .obs column
ATLAS_SUPERTYPE   = 'supertype'        # atlas .obs column (finer level)
QUERY_SUBCLASS    = 'Subclass_V2'      # our spinal subclass label
QUERY_GROUP       = 'Group_V2'         # our spinal group label
OVERLAP_MIN       = 0.20               # overlap-coefficient threshold for "reciprocal"

for p in [QUERY_H5AD, FORWARD_JSON, REVERSE_JSON, RECIPROCAL_CSV,
          SUBCLASS_OVERLAP_CSV, SUPERTYPE_RECIP_CSV, SPATIAL_ATLAS]:
    print(f"{'OK ' if os.path.exists(p) else 'MISSING'} {p}")""")

# ---- 1. Forward ---------------------------------------------------------------
md("""## 1. Forward — which mouse whole-brain types did we capture?

`cell_type_mapper` (the MapMyCells engine) assigned every spinal-cord nucleus to
its best-matching mouse-WB **subclass**. We load that result and attach the
spinal cells' own `Subclass_V2` / `Group_V2` labels so we can compare.""")
code("""mapping_df = mio.load_mapping_results(FORWARD_JSON, levels=[WB_SUBCLASS_LEVEL])
print(mapping_df.shape[0], 'spinal cells mapped to a mouse-WB subclass')

# Attach our spinal labels (read obs only; backed='r' keeps it light).
adata_q = sc.read_h5ad(QUERY_H5AD, backed='r')
qobs = adata_q.obs[[QUERY_SUBCLASS, QUERY_GROUP]].copy()
mapping_df = mapping_df.join(qobs, how='left')
mapping_df.head()""")

md("""### Top detected mouse-WB subclasses

The bar chart shows the mouse whole-brain subclasses our spinal cord maps to most
often. Expect **hindbrain / spinal-trigeminal** types (e.g. `SPVI-SPVC …`) — the
brainstem continuation of the dorsal horn — plus the ubiquitous non-neuronal
classes (Oligo, Microglia).""")
code("""counts = mapping_df[WB_SUBCLASS_LEVEL].value_counts()
fig, ax = plt.subplots(figsize=(8, 9))
counts.head(30)[::-1].plot.barh(ax=ax, color='steelblue')
ax.set_xlabel('# spinal cells'); ax.set_title('Top 30 detected mouse-WB subclasses')
plt.tight_layout(); plt.show()
print(f"{(counts > 0).sum()} distinct mouse-WB subclasses detected.")""")

md("""### Mapping confidence

The per-cell **bootstrapping probability** (0–1) reports how stable each
assignment is across resampled marker sets — a cross-species confidence score.""")
code("""prob_col = f'{WB_SUBCLASS_LEVEL}_prob'
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(mapping_df[prob_col], bins=40, ax=ax)
ax.set_xlabel('subclass bootstrapping probability'); ax.set_title('Forward mapping confidence')
plt.show()
print('median confidence:', round(float(mapping_df[prob_col].median()), 3))""")

md("""### Forward summary by spinal subclass

For each of **our** spinal subclasses, which mouse-WB subclass do its cells most
often map to? This is the per-row "forward best hit".""")
code("""fwd_by_sub = mio.summarize_forward_by_query_level(
    mapping_df, qobs, QUERY_SUBCLASS, WB_SUBCLASS_LEVEL)
fwd_by_sub""")

# ---- 2. Reverse ---------------------------------------------------------------
md("""## 2. Reverse — which mouse-WB subclasses resemble a spinal type?

Now the other direction. We built a **new** `cell_type_mapper` reference from our
**consensus spinal-cord** taxonomy (`Class → Subclass → Group → consensus_cluster`)
and mapped each mouse-WB subclass's mean expression onto it. The result tells us,
for every mouse subclass, the spinal **Subclass / Group** it most resembles.""")
code("""reverse_df = mio.load_mapping_results(
    REVERSE_JSON, levels=['Class', 'Subclass', 'Group'])
reverse_df.index.name = 'mouse_wb_subclass'
print(reverse_df.shape[0], 'mouse-WB subclasses mapped onto the spinal cord')
reverse_df[['Class', 'Subclass', 'Group']].head(10)""")

md("""Which spinal **Subclasses** do the mouse-WB subclasses pile onto? Broad
mouse types collapse onto a few spinal "hub" groups, but the **Subclass** level
(Glut-D/M/V, GABA-D/M/V …) is shared between the two taxonomies and is where the
reciprocal signal is cleanest.""")
code("""fig, ax = plt.subplots(figsize=(7, 5))
reverse_df['Subclass'].value_counts()[::-1].plot.barh(ax=ax, color='indianred')
ax.set_xlabel('# mouse-WB subclasses'); ax.set_title('Reverse: spinal Subclass each mouse type maps to')
plt.tight_layout(); plt.show()""")

# ---- 3. Reciprocal ------------------------------------------------------------
md("""## 3. Reciprocal best hits

A mouse-WB subclass is **reciprocal** when the two directions agree on the spinal
**Subclass**: the spinal cells that map to it (forward) are the same subclass that
it maps back to (reverse). `processing/04` computed this; we load and inspect it.

Watch the dorsal-horn story emerge: brainstem **`SPVI-SPVC`** (spinal-trigeminal)
excitatory `Tlx3` types are reciprocal with our dorsal **`Glut-D`** subclass —
the whole-brain analogue of the spinal dorsal horn.""")
code("""recip = pd.read_csv(RECIPROCAL_CSV, index_col=0)
n_rec = int(recip['reciprocal'].sum())
print(f"{n_rec} of {len(recip)} detected mouse-WB subclasses are reciprocal.")
recip_true = recip[recip['reciprocal']].sort_values('n_spc_cells', ascending=False)
recip_true.head(20)[['n_spc_cells', 'fwd_spc_subclass', 'rev_spc_subclass', 'rev_spc_group']]""")

md("""### Reciprocal hits for the dorsal horn (`Glut-D`)

Filter to the mouse-WB subclasses that are reciprocal with our **dorsal
excitatory** spinal subclass. These are the brain regions whose excitatory
neurons most resemble spinal dorsal-horn neurons.""")
code("""dorsal = recip_true[recip_true['fwd_spc_subclass'].str.contains('Glut-D')]
print('Mouse-WB subclasses reciprocal with spinal Glut-D (dorsal horn):')
dorsal[['n_spc_cells', 'fwd_spc_subclass', 'rev_spc_subclass', 'rev_spc_group']]""")

# ---- 3b. Overlap-coefficient reciprocity --------------------------------------
md("""## 3b. *How* reciprocal? Grading pairs by the overlap coefficient

The reciprocal flag above is all-or-nothing. To grade *how strongly* a mouse-WB
subclass and a spinal subclass correspond, we use the **overlap coefficient**
(Szymkiewicz–Simpson) on the forward assignments. For a spinal subclass `S`
(its cells = set `A`) and a mouse-WB subclass `M` (cells assigned to `M` = set
`B`):

$$\\mathrm{OC}(S, M) = \\frac{|A \\cap B|}{\\min(|A|, |B|)}$$

`OC = 1` means one set sits entirely inside the other — a clean, exclusive
correspondence; `OC ≈ 0` means they barely co-occur. Unlike a raw fraction it
isn't diluted when the two types differ wildly in size. We call a pair
**reciprocal** when *both* directions name the same spinal subclass **and**
`OC ≥ OVERLAP_MIN` ( = 0.20). `processing/07` precomputed this.""")
code("""sub_ov = pd.read_csv(SUBCLASS_OVERLAP_CSV, index_col=0)
n_rec = int(sub_ov['reciprocal'].sum())
print(f"{n_rec} of {len(sub_ov)} mouse-WB subclasses are reciprocal "
      f"(agree both ways AND overlap >= {OVERLAP_MIN}).")
sub_ov.sort_values('overlap', ascending=False).head(15)[
    ['n_spc_cells', 'fwd_spc_subclass', 'overlap', 'rev_spc_subclass', 'reciprocal']]""")

md("""### Reciprocal mapping heatmap (subclass level)

Recompute the full overlap-coefficient matrix (spinal subclasses × detected
mouse-WB subclasses) and show it for the **reciprocal** mouse-WB subclasses,
grouped by the spinal subclass they reciprocate with. The bright block down each
spinal column is its set of whole-brain counterparts — read off, for example, the
`Glut-D` column to see every brainstem dorsal-horn-like subclass.""")
code("""oc = mio.overlap_coefficient_matrix(mapping_df, qobs, QUERY_SUBCLASS,
                                    WB_SUBCLASS_LEVEL)            # spinal × WB
rec_wb = sub_ov.index[sub_ov['reciprocal']]
order = (sub_ov.loc[rec_wb]
         .sort_values(['fwd_spc_subclass', 'overlap'], ascending=[True, False])
         .index)
# Columns: only spinal subclasses that actually have a reciprocal partner.
spc_cols = [c for c in oc.index if c in set(sub_ov.loc[rec_wb, 'fwd_spc_subclass'])]
H = oc.loc[spc_cols, order].T                                    # WB(rows) × spinal(cols)

fig, ax = plt.subplots(figsize=(1.0 + 0.55 * len(spc_cols), max(6, 0.22 * len(H))))
sns.heatmap(H, cmap='magma', vmin=0, vmax=1, ax=ax,
            cbar_kws={'label': 'overlap coefficient', 'shrink': 0.4})
ax.set_xlabel('our spinal Subclass'); ax.set_ylabel('reciprocal mouse-WB subclass')
ax.set_title(f'Reciprocal mapping (overlap coefficient, {len(H)} reciprocal subclasses)')
plt.tight_layout(); plt.show()""")

# ---- 3c. Supertype-level reciprocity ------------------------------------------
md("""## 3c. Finer grain: reciprocal **supertypes**

Subclass is fairly coarse. Using the subsampled mouse-WB snRNA dataset
(`wb_subsampled_ABC.h5ad`, kept small enough to ship to you — the full atlas is
224 GB) we built **supertype** mean profiles and ran the reverse arm again at
supertype resolution. A supertype is **reciprocal** when the spinal subclass it
maps back to matches its parent subclass's forward partner, with the same
`OVERLAP_MIN` gate. This splits each reciprocal subclass into the supertypes that
genuinely reciprocate and those that don't.""")
code("""st_recip = pd.read_csv(SUPERTYPE_RECIP_CSV, index_col=0)
n_st = int(st_recip['reciprocal'].sum())
print(f"{n_st} of {len(st_recip)} mouse-WB supertypes are reciprocal.")
# How many reciprocal supertypes fall under each spinal subclass?
by_spc = (st_recip[st_recip['reciprocal']]
          .groupby('fwd_spc_subclass').size().sort_values(ascending=False))
print('\\nReciprocal supertypes per spinal subclass:'); print(by_spc)
st_recip[st_recip['reciprocal']].sort_values('overlap', ascending=False).head(15)[
    ['parent_wb_subclass', 'fwd_spc_subclass', 'rev_spc_subclass', 'overlap']]""")

# ---- 4. Spatial: build reference ----------------------------------------------
md("""## 4. Spatial projection — load the MERFISH atlas

To see *where in the brain* these types live we use the ABC mouse-brain MERFISH
spatial atlas (16 clean sections, ~1.2 M cells; each cell carries its mouse-WB
subclass and a CCF region). `build_spatial_reference` also returns each
subclass's **spatial specificity** (1 − fraction of sections it appears in) so we
can down-weight ubiquitous types later.""")
code("""adata_atlas, specificity = mio.build_spatial_reference(
    SPATIAL_ATLAS, subclass_key=ATLAS_SUBCLASS)
print(adata_atlas.shape)
print('\\nMost spatially specific subclasses:')
print(specificity.sort_values(ascending=False).head(8))""")

md("""### Lay the sections out on a grid

The 16 MERFISH sections overlap in raw coordinates; tile them into a montage so
we can view the whole brain at once. (Helper adapted from the spatial tutorial.)""")
code("""def tile_sections_on_grid(adata, section_key='brain_section_label',
                          spatial_key='spatial_grid', pad=0.0, new_key='spatial_tiled'):
    coords = np.asarray(adata.obsm[spatial_key])
    if isinstance(adata.obs[section_key].dtype, pd.CategoricalDtype):
        sections = list(adata.obs[section_key].cat.categories)
    else:
        sections = list(pd.unique(adata.obs[section_key]))
    centers, widths, heights = {}, [], []
    for s in sections:
        m = (adata.obs[section_key].values == s); sc_ = coords[m]
        if sc_.size == 0:
            centers[s] = np.array([0., 0.]); widths.append(0.); heights.append(0.); continue
        cmin, cmax = sc_.min(0), sc_.max(0)
        widths.append(cmax[0]-cmin[0]); heights.append(cmax[1]-cmin[1]); centers[s] = sc_.mean(0)
    tile_w = (max(widths) if widths else 0.) + pad
    tile_h = (max(heights) if heights else 0.) + pad
    n_cols = int(np.ceil(np.sqrt(len(sections))))
    new = coords.copy()
    for i, s in enumerate(sections):
        r, c = i // n_cols, i % n_cols
        m = (adata.obs[section_key].values == s)
        new[m] = coords[m] + (np.array([c*tile_w, r*tile_h], float) - centers[s])
    adata.obsm[new_key] = new


def plot_section(xx, yy, cc=None, val=None, cmap=None, s=0.5, alpha=1.0,
                 fig_width=12, fig_height=12, title=None):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if cmap is not None:
        sct = ax.scatter(xx, yy, s=s, c=val, marker='.', cmap=cmap, alpha=alpha)
    else:
        sct = ax.scatter(xx, yy, s=s, color=cc, marker='.', alpha=alpha)
    ax.axis('equal'); ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title)
    return fig, ax, sct

tile_sections_on_grid(adata_atlas, pad=0.0)
adata_atlas.obsm['spatial_tiled'][:, 1] *= -1            # flip y so the brain is upright
adata_atlas.obs['x'] = adata_atlas.obsm['spatial_tiled'][:, 0]
adata_atlas.obs['y'] = adata_atlas.obsm['spatial_tiled'][:, 1]
print('tiled', adata_atlas.n_obs, 'cells across',
      adata_atlas.obs['brain_section_label'].nunique(), 'sections')""")

# ---- 4b. Reference identities in space ----------------------------------------
md("""## 4b. Reference cell-type identities in space (neurons)

Before overlaying our mapping, look at the atlas's **own** taxonomy in space,
**neurons only** (drop the ubiquitous `… NN` non-neuronal subclasses so neuronal
structure is visible). Each cell is drawn in the atlas's *own* colour for that
level — first **subclass**, then the finer **supertype**.""")
code("""def plot_identities(obs, color_col, label_col, title, n_legend=18, s=0.25):
    \"\"\"Montage coloured by an atlas annotation's own per-cell *_color column,
    with a legend of the most common identities.\"\"\"
    import matplotlib.patches as mpatches
    fig, ax, _ = plot_section(obs['x'], obs['y'], cc=obs[color_col].values, s=s,
                              title=title)
    top = obs[label_col].astype(str).value_counts().head(n_legend).index
    cmap = dict(zip(obs[label_col].astype(str), obs[color_col].astype(str)))
    ax.legend(handles=[mpatches.Patch(color=cmap[v], label=v) for v in top],
              bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, frameon=False)
    plt.show()

neuron_obs = adata_atlas.obs[
    ~adata_atlas.obs[ATLAS_SUBCLASS].str.contains(' NN$', regex=True)].copy()
print(neuron_obs.shape[0], 'neuronal cells;',
      neuron_obs['subclass'].nunique(), 'subclasses,',
      neuron_obs['supertype'].nunique(), 'supertypes')""")

md("""### Neuron **subclasses** (atlas colours)""")
code("""plot_identities(neuron_obs, 'subclass_color', 'subclass',
                'Atlas neurons coloured by subclass identity')""")

md("""### Neuron **supertypes** (atlas colours)

Supertype is the next finer level below subclass — many more categories, so the
legend shows only the most abundant ones, but every cell is coloured.""")
code("""plot_identities(neuron_obs, 'supertype_color', 'supertype',
                'Atlas neurons coloured by supertype identity')""")

# ---- 5. Spatial: forward coverage ---------------------------------------------
md("""## 5. Where do the detected types live? (forward coverage)

For every mouse-WB subclass we score how strongly the spinal query "detected" it
(`representation_score` = min(n_spinal_cells / FULL_N, 1)), then paint that score
onto each atlas cell. **Green = detected** by the spinal cord, **red = not** —
i.e. brain regions whose cell types resemble spinal cells light up green.""")
code("""FULL_N = 500   # spinal cells at which a subclass counts as fully represented
rep_df = mio.representation_table(
    detected_counts=mapping_df[WB_SUBCLASS_LEVEL].value_counts(),
    reference_subclasses=adata_atlas.obs[ATLAS_SUBCLASS].cat.categories,
    full_n=FULL_N, specificity=specificity)
mio.attach_representation_to_spatial(adata_atlas, rep_df, subclass_key=ATLAS_SUBCLASS)

# Drop ubiquitous non-neuronal ("… NN") types so neuronal structure is visible.
neurons = adata_atlas[~adata_atlas.obs[ATLAS_SUBCLASS].str.contains(' NN$', regex=True)]
o = neurons.obs
fig, ax, sct = plot_section(o['x'], o['y'], val=o['representation_score'].values,
                            cmap='RdYlGn', s=0.3,
                            title='Forward coverage: spinal-detected (green) vs not (red) subclasses')
plt.colorbar(sct, ax=ax, shrink=0.5, label='representation score'); plt.show()""")

# ---- 6. Spatial: reciprocal highlight -----------------------------------------
md("""## 6. Where do the **reciprocal** types live?

Now the payoff: highlight only the mouse-WB subclasses that are **reciprocal**
with a spinal type (agreeing both ways with overlap coefficient ≥ `OVERLAP_MIN`).
These are the brain locations that are genuinely spinal-cord-like. Expect the
**hindbrain / medulla** and the **spinal-trigeminal** territory to light up.""")
code("""recip_set = set(sub_ov.index[sub_ov['reciprocal']])
adata_atlas.obs['is_reciprocal'] = (
    adata_atlas.obs[ATLAS_SUBCLASS].astype(str).isin(recip_set))
o = adata_atlas.obs
fig, ax, _ = plot_section(o['x'], o['y'],
                          cc=np.where(o['is_reciprocal'], 'crimson', 'lightgrey'),
                          s=0.3, title='Reciprocal spinal-like subclasses (crimson) across the brain')
plt.show()
print('reciprocal cells:', int(o['is_reciprocal'].sum()),
      f"({o['is_reciprocal'].mean():.1%} of atlas)")""")

md("""### Colour the reciprocal cells by the spinal subclass they match

Paint each reciprocal atlas cell by the spinal **Subclass** it reciprocates with
(`Glut-D` dorsal, `Glut-V` ventral, `GABA-*` inhibitory, …). Non-reciprocal cells
are greyed out. This is the spatial answer to *"where do our spinal types map to
in the brain?"*""")
code("""sub_map = sub_ov.loc[sub_ov['reciprocal'], 'fwd_spc_subclass'].to_dict()
o['recip_spc_subclass'] = o[ATLAS_SUBCLASS].astype(str).map(sub_map)
cats = (o['recip_spc_subclass'].dropna().value_counts().head(10).index.tolist())
palette = dict(zip(cats, sns.color_palette('tab10', len(cats))))
colors = o['recip_spc_subclass'].map(lambda v: palette.get(v, (0.85, 0.85, 0.85)))
colors = colors.where(o['recip_spc_subclass'].isin(cats), other=None)
colors = [c if isinstance(c, tuple) else (0.85, 0.85, 0.85) for c in colors]

fig, ax, _ = plot_section(o['x'], o['y'], cc=colors, s=0.3,
                          title='Brain cells coloured by the spinal Subclass they reciprocate with')
import matplotlib.patches as mpatches
ax.legend(handles=[mpatches.Patch(color=palette[c], label=c) for c in cats],
          bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, frameon=False)
plt.show()""")

# ---- 6b. Reciprocal supertypes in space ---------------------------------------
md("""## 6b. Reciprocal **supertypes** in space

The same map at the finer **supertype** grain. Because reciprocity was resolved
per supertype (§3c), only the supertypes that genuinely reciprocate light up —
a subset of the reciprocal-subclass territory above. We colour each reciprocal
supertype by the spinal subclass it matches, so you can see the dorsal/ventral
and excitatory/inhibitory split persist at higher resolution.""")
code("""recip_super = dict(zip(st_recip.index[st_recip['reciprocal']],
                       st_recip.loc[st_recip['reciprocal'], 'fwd_spc_subclass']))
o['is_recip_supertype'] = o[ATLAS_SUPERTYPE].astype(str).isin(recip_super)
o['recip_super_spc'] = o[ATLAS_SUPERTYPE].astype(str).map(recip_super)
print('reciprocal-supertype cells:', int(o['is_recip_supertype'].sum()),
      f"({o['is_recip_supertype'].mean():.1%} of atlas);",
      o['recip_super_spc'].notna().sum(), 'painted')

cats2 = o['recip_super_spc'].dropna().value_counts().head(10).index.tolist()
pal2 = dict(zip(cats2, sns.color_palette('tab10', len(cats2))))
colors2 = [pal2.get(v, (0.88, 0.88, 0.88)) if isinstance(v, str) else (0.88, 0.88, 0.88)
           for v in o['recip_super_spc']]
fig, ax, _ = plot_section(o['x'], o['y'], cc=colors2, s=0.3,
                          title='Reciprocal supertypes coloured by the spinal Subclass they match')
ax.legend(handles=[mpatches.Patch(color=pal2[c], label=c) for c in cats2],
          bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, frameon=False)
plt.show()""")

# ---- 7. CCF regions -----------------------------------------------------------
md("""## 7. Which brain regions are the reciprocal types in?

Cross-tabulate the reciprocal cells against their **CCF region** (anatomical
parcellation) to name the structures most spinal-cord-like in transcriptome.""")
code("""recip_cells = adata_atlas.obs[adata_atlas.obs['is_reciprocal']]
region = (recip_cells['parcellation_division'].astype(str)
          .value_counts().head(15))
fig, ax = plt.subplots(figsize=(7, 5))
region[::-1].plot.barh(ax=ax, color='teal')
ax.set_xlabel('# reciprocal cells'); ax.set_title('CCF divisions of the reciprocal spinal-like cells')
plt.tight_layout(); plt.show()
region""")

# ---- 8. Exercise --------------------------------------------------------------
md("""## 8. Exercise & discussion — functional interpretation

You now have a reciprocal map between the spinal cord and the whole brain, and a
picture of where the spinal-like types live. Discuss / explore:

1. **Dorsal horn analogue.** Section 3 showed `SPVI-SPVC Tlx3 … Glut` is
   reciprocal with spinal `Glut-D`. `Tlx3` is the canonical **dorsal-horn
   excitatory** selector gene. Which CCF division did these cells fall in (§7)?
   Why does the *spinal-trigeminal nucleus* resemble the dorsal horn?
2. **Ventral vs dorsal.** Compare where `Glut-D` vs `Glut-V` reciprocal cells sit
   in §6. Does the dorsal/ventral spinal axis have a whole-brain counterpart?
3. **Coverage gaps.** In §5, which neuronal regions stayed **red** (no spinal
   match)? Those are mouse-WB types with *no* spinal counterpart — sanity-check a
   few (forebrain? cerebellum?).
4. **Confidence.** Re-make the §1 confidence histogram split by `Class_V2`
   (`adata_q.obs`). Do non-neuronal cells map more confidently than neurons?

> **Scaling up.** Here the reverse arm used 338 subclass *means*. The full
> 4,042,976-cell `AIT21 → spinal cord` mapping (≈3.6 h) is bundled at
> `/data/hmba_xs_v1/mouse_wb/AIT21.all.freeze.230815_humanorthos_ABC_MAPPING/` —
> per-cell reciprocity at the same resolution.""")

md("""---
*Reciprocal mapping built with [cell_type_mapper](https://github.com/AllenInstitute/cell_type_mapper)
(MapMyCells). Forward reference: AIT21 mouse whole-brain subclasses. Reverse
reference: AIBS consensus spinal-cord taxonomy. Spatial atlas: ABC MERFISH
`C57BL6J-638850`.*""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'session2_reciprocal_mapping_spatial.ipynb')
with open(OUT, 'w') as f:
    nbf.write(nb, f)
print('wrote', OUT, 'with', len(cells), 'cells')
