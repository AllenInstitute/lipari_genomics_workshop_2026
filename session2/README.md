# Session 2 — Spinal-cord annotation & whole-brain mapping (Interactive Part 2)

Materials for the second interactive session of the **Lipari Genomics Workshop
2026** ([repo overview](../README.md)). Students first **annotate** classic
spinal-cord (SpC) cell types from Session 1, then relate that taxonomy to the
Allen **mouse whole-brain (AIT21)** atlas with **MapMyCells**, and project the
result onto the ABC mouse-brain **MERFISH spatial atlas** to see *where in the
brain* the spinal-like cell types live. Run it in the pre-built **Code Ocean**
capsule (<https://codeocean.allenneuraldynamics.org/>).

> **Student notebooks (run in order):**
> 1. `notebooks/session2_literature_cell_types.ipynb` — Part 2A: name the cell types
> 2. `notebooks/session2_webportal_mapping_spatial.ipynb` — Part 2B: map to the whole brain (MapMyCells)

## The reciprocal idea
[`cell_type_mapper`](https://github.com/AllenInstitute/cell_type_mapper) (the
engine behind **MapMyCells**) is run in both directions:

| direction | query → reference | question |
|---|---|---|
| **Forward** | SpC snRNA → mouse‑WB **subclass** | which whole‑brain types did our cord capture? |
| **Reverse** | mouse‑WB subclass → **our consensus SpC** | which whole‑brain types resemble a spinal type? |
| **Reciprocal** | keep pairs that agree both ways | robust SpC ↔ WB correspondence |
| **Spatial** | reciprocal subclasses → MERFISH atlas | where in the brain do they live? |

The forward arm uses the **pre‑built mouse‑WB subclass reference** that ships
with the spatial tutorial bundle. The reverse arm uses a **new** reference built
from the **AIBS consensus spinal‑cord taxonomy** (`Class → Subclass → Group →
consensus_cluster`) — that is the "building a new mapping reference for our
consensus spinal cord" step.

Reciprocity is anchored on the **Subclass** level (`Glut-D/M/V`, `GABA-D/M/V`,
Astro/Oligo/OPC…), the vocabulary the two spinal taxonomies share cleanly (their
*Group* and *Class* names differ). The headline result: brainstem
**`SPVI-SPVC Tlx3 … Glut`** (spinal‑trigeminal) is reciprocal with our dorsal
excitatory **`Glut-D`** — the whole‑brain analogue of the spinal **dorsal horn**.

### Grading reciprocity (overlap coefficient) and the supertype level
How *strongly* two types correspond is graded by the **overlap coefficient**
(Szymkiewicz–Simpson) `OC(S,M) = |A∩B| / min(|A|,|B|)` on the forward
assignments; a pair counts as reciprocal only when both arms agree **and**
`OC ≥ OVERLAP_MIN` (0.20, in `00_config.py`). The reverse arm is also run one
level finer, at mouse‑WB **supertype** resolution, so each reciprocal subclass is
split into the supertypes that genuinely reciprocate and those that don't. To get
supertype profiles without streaming the 224 GB atlas live, `05` subsamples
AIT21 (≤150 seeded cells/supertype) into a small, **student‑shippable**
`wb_subsampled_ABC.h5ad` and averages it.

## Setup
On Code Ocean the environment and data are already provided; just open the two
student notebooks and run the cells. To run elsewhere:
```bash
pip install -r requirements.txt          # Python 3.12
jupyter lab notebooks/session2_literature_cell_types.ipynb        # Part 2A (first)
jupyter lab notebooks/session2_webportal_mapping_spatial.ipynb    # Part 2B
```
The student notebooks only **load** the pre‑computed results in
**`/data/lipari_workshop/`** (below) plus the bundled MERFISH atlas — they do not
run any mapping live, so `cell_type_mapper` is needed only to rebuild the
artifacts.

## Layout
```
session2/
├── processing/                          # heavy, run-once data prep (not run live)
│   ├── 00_config.py                     # shared paths + global SEED + set_all_seeds()
│   ├── 01_map_spc_to_wb_subclass.sh     # FORWARD: SpC snRNA  -> mouse-WB subclass
│   ├── 02_build_spc_reference.sh        # build NEW consensus-SpC mapping reference
│   ├── 03_map_wb_to_spc.py              # REVERSE: mouse-WB subclass means -> SpC ref
│   ├── 04_build_reciprocal_artifacts.py # distill forward+reverse -> /results CSVs
│   ├── 05_build_supertype_means.py      # subsample AIT21 -> student snRNA + supertype means
│   ├── 06_map_wb_supertype_to_spc.py    # REVERSE at supertype resolution
│   ├── 07_build_supertype_reciprocal_artifacts.py  # overlap-coefficient reciprocity tables
│   └── mapping_io.py                    # reciprocal-mapping helpers (from the tutorial, extended)
└── notebooks/
    ├── _build_literature_notebook.py    # regenerates the literature cell-type .ipynb
    ├── _build_webportal_notebook.py     # regenerates the MapMyCells web-portal .ipynb
    ├── session2_literature_cell_types.ipynb        # Part 2A: find textbook cell types (run first)
    ├── session2_webportal_mapping_spatial.ipynb    # Part 2B: map to mouse brain (MapMyCells portal)
    └── old/                             # advanced, fully-local reciprocal-mapping variant
        ├── _build_notebook.py                          #   (no web portal; for the Python-comfortable)
        └── session2_reciprocal_mapping_spatial.ipynb
```
The two notebooks at the top are the live student notebooks. The **web-portal**
notebook is the primary Part 2B path (map on MapMyCells, visualize back here); the
fully-local **reciprocal** variant under `old/` does the same mapping without the
web portal and is optional for anyone comfortable in Python.

### Notebook order
`session2_literature_cell_types.ipynb` comes **first**: the descriptive `Group_V2`
names are hidden behind anonymous **`Group` IDs** (`Subclass_V2` + a number, e.g.
`Glut-D 6`), and students work out which anonymous group is which classic spinal-cord
cell type (dorsal-horn `TAC3`/`NMU` itch neurons, `PHOX2A`/`RELN`/`LMX1B` ascending
nociceptive projection neurons, and `CHRNA5` Renshaw cells). They rank the anonymous
groups on marker combinations, view the markers on the snRNA UMAP and in dotplots, and
`explore_group()` any ID on the UMAP **and** in the example spatial sections at once;
the real name is uncovered only in each target's reveal. Only then do they map the
*whole* taxonomy onto the mouse whole brain with the web-portal (or reciprocal) notebook.

## Workshop data (provided in `/data/lipari_workshop/`, built by the processing scripts)

| File | What it is |
|------|------------|
| `SpC_workshop_WB_SUBCLASS_MAPPING/hann_results.json` | **Forward** — every SpC nucleus → mouse‑WB subclass (+ confidence). |
| `WB_SUBCLASS_to_SpC_MAPPING/hann_results.json` | **Reverse** — every mouse‑WB subclass → SpC Class/Subclass/Group. |
| `wb_subclass_means.h5ad` | The 338 mouse‑WB subclass mean profiles used as the reverse query. |
| `reciprocal_forward_summary.csv` | Per SpC `Group_V2`: the mouse‑WB subclass it most maps to. |
| `reciprocal_reverse_summary.csv` | Per mouse‑WB subclass: the SpC type it maps back to. |
| `reciprocal_best_hits.csv` | The join: which mouse‑WB subclasses are **reciprocal** (pivoted on WB subclass, Subclass‑anchored). |
| `reciprocal_subclass_overlap.csv` | Per mouse‑WB subclass: best spinal partner by **overlap coefficient**, reverse hit, reciprocal flag. |
| `reciprocal_supertype_hits.csv` | Per mouse‑WB **supertype**: parent subclass, both directions, inherited overlap, reciprocal flag. |
| `wb_supertype_means.h5ad` | 1201 mouse‑WB supertype mean profiles (subsampled) — the supertype reverse query. |
| `wb_subsampled_ABC.h5ad` | Student‑sized mouse‑WB snRNA (≤150 cells/supertype, RAW counts) — the shippable stand‑in for the 224 GB atlas. |

The MERFISH spatial atlas and the two mapping references are read in place from
`/data/mouse_wb_spatial_tutorial/` and `/scratch/SpC_consensus_ref/` (see
`00_config.py`); the web-portal notebook also reads its MapMyCells results `.zip`
(`PORTAL_OUTPUT`) from `/data/lipari_workshop/`. To (re)download the ABC MERFISH
atlas from scratch, run `00_download_spatial_atlas.py` (its module docstring lists
the ABC Atlas cache commands, sizes, and links).

### Rebuild
```bash
cd processing
python 00_download_spatial_atlas.py   # (re)download the ABC MERFISH atlas (only if not already bundled)
bash   01_map_spc_to_wb_subclass.sh   # forward mapping  (~1 min, CPU)
bash   02_build_spc_reference.sh      # SpC consensus reference (reuses the prebuilt copy if present)
python 03_map_wb_to_spc.py            # reverse mapping  (~1 min, CPU)
python 04_build_reciprocal_artifacts.py
python 05_build_supertype_means.py    # subsample AIT21 -> student snRNA + supertype means (~25-30 min, I/O)
python 06_map_wb_supertype_to_spc.py  # reverse mapping at supertype resolution (~1 min, CPU)
python 07_build_supertype_reciprocal_artifacts.py
```
`01`–`03`/`06` install/`use` `cell_type_mapper`. The reverse arm maps the 338
**subclass means** (fast); `05` makes a single sequential, parallel pass over the
224 GB `AIT21` matrix (network‑I/O bound, ~25–30 min) to subsample it. The full
per‑cell `AIT21 → spinal cord` mapping of all 4,042,976 cells (≈3.6 h) is already
bundled at
`/data/hmba_xs_v1/mouse_wb/AIT21.all.freeze.230815_humanorthos_ABC_MAPPING/`.

## Reproducibility
Every script and the notebook fix all RNG seeds via `SEED = 0`
(`set_all_seeds()` in `00_config.py`) and pass `rng_seed` to `cell_type_mapper`,
so the mapping, reciprocal table, and plots match across machines.

## Notebook flow
Forward mapping (top detected subclasses + confidence) → reverse mapping → the
reciprocal best‑hit table (dorsal horn ↔ spinal‑trigeminal) → **overlap‑coefficient
grading + reciprocal mapping heatmap** → **supertype‑level reciprocity** → load the
MERFISH atlas → reference subclass/supertype identities in space (neurons) →
forward coverage map (green = detected) → **reciprocal** subclass *and* supertype
highlights across the brain, coloured by the spinal subclass each region matches →
CCF‑region breakdown → functional‑interpretation exercise. Each code cell is
preceded by a short markdown cell describing the step.

## Source material this is distilled from
- Spatial / reciprocal tutorial bundle: `/data/mouse_wb_spatial_tutorial/`
  (`notebooks/2_explore_and_plot.ipynb`, `python/mapping_io.py`,
  `scripts/1_run_mapping.sh`).
- Provenance of the tutorial: the cetacean preprocessing & SpatialQC pipeline.
- Reverse reference source taxonomy:
  `/data/SpinalCord/manuscript/RNA/AIBS_SpC_consensus_taxonomy_harmonized_AIT-pre-print.h5ad`.
