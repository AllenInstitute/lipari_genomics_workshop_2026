# Lipari Genomics Workshop 2026 — Brain atlases

Companion code for the two **interactive sessions** of the *Brain atlases* lectures
at the 2026 Lipari International Summer School. The sessions are presented by
researchers from the **Allen Institute**, using Allen Institute atlases and tools.
If an instructor
just pointed you here, you're in the right place — everything you need for the
hands-on notebooks is in this repo.

Theme: **how modern brain-cell atlases are built and used.** You take a
multi-species **spinal-cord** snRNA-seq dataset from raw nuclei all the way to
annotated cell types, then map those types onto the Allen **whole-mouse-brain**
atlas to see where their closest relatives live.

## Where to run it
These notebooks are meant to run in the pre-configured **Code Ocean** capsule
(<https://codeocean.allenneuraldynamics.org/>) — the Python environment and all
data are already set up there, so you can just open a notebook and run the cells.
The workshop data lives read-only under **`/data/lipari_workshop/`**; Session 1
writes its outputs to the writable **`/results/`** volume.

To run elsewhere, install the per-session environment (`pip install -r
session1/requirements.txt`, Python 3.12) and supply the data objects listed in
each session README.

## The two sessions

### 🧬 Session 1 — Single-cell analysis & visualization *(Monday)*
> [`session1/notebooks/session1_qc_clustering_visualization.ipynb`](session1/notebooks/session1_qc_clustering_visualization.ipynb)

"You just got your data back." Walk a spinal-cord snRNA-seq subsample through a
real QC → clustering → visualization pipeline in **scanpy**: explore quality
thresholds and marker genes, cluster and embed the nuclei, compare your calls to
the published taxonomy, then export the object for interactive browsing in
**cellxgene** and explore the **ABC whole-brain atlas** online.
See [`session1/README.md`](session1/README.md).

### 🗺️ Session 2 — Spinal-cord annotation & whole-brain mapping *(Tuesday)*
Run the two student notebooks in order:
1. **Part 2A — find the textbook cell types**
   [`session2/notebooks/session2_literature_cell_types.ipynb`](session2/notebooks/session2_literature_cell_types.ipynb)
   The descriptive cell-type names are hidden; use marker genes, the snRNA UMAP,
   and example spatial sections to identify classic spinal-cord neurons (itch,
   nociceptive-projection, Renshaw) behind anonymous IDs.
2. **Part 2B — map to the mouse whole brain**
   [`session2/notebooks/session2_webportal_mapping_spatial.ipynb`](session2/notebooks/session2_webportal_mapping_spatial.ipynb)
   Map your Session-1 taxonomy to the Allen whole-mouse-brain atlas with the
   **MapMyCells** web portal, then visualize the results on the mouse-brain
   MERFISH spatial atlas.

See [`session2/README.md`](session2/README.md).

## Tools you'll touch
| Tool | What it's for | Link |
|------|---------------|------|
| **scanpy** | single-cell QC, clustering, embedding | <https://scanpy.readthedocs.io/> |
| **cellxgene** | interactive h5ad browser (genes, DE, annotation) | <https://cellxgene.cziscience.com/> |
| **ABC Atlas** | explore the Allen whole-brain reference online | <https://knowledge.brain-map.org/abcatlas> |
| **MapMyCells** | map your cells to Allen reference taxonomies | <https://knowledge.brain-map.org/mapmycells/process/> |
| **cell_type_mapper** | the Python engine behind MapMyCells | <https://github.com/AllenInstitute/cell_type_mapper> |

## Repo layout
```
lipari_genomics_workshop_2026/
├── session1/   # QC · clustering · visualization  (+ cellxgene, ABC Atlas)
│   ├── notebooks/     # the student notebook
│   ├── processing/    # run-once data prep (not run live)
│   └── requirements.txt
└── session2/   # spinal-cord annotation · whole-brain mapping (MapMyCells)
    ├── notebooks/     # the two student notebooks (+ old/ advanced variant)
    ├── processing/    # run-once mapping pipeline (not run live)
    └── requirements.txt
```
The `processing/` scripts build the small, notebook-ready data objects and are
**not** part of the live sessions — each session README documents them for anyone
who wants to rebuild the data from scratch.
