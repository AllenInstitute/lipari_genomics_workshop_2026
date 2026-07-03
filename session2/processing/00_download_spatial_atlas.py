#!/usr/bin/env python
"""Download the ABC Atlas mouse whole-brain MERFISH spatial atlas.

Downloads the Allen Brain Cell (ABC) Atlas mouse whole-brain MERFISH dataset
(``MERFISH-C57BL6J-638850``) straight from the ABC Atlas public S3 bucket using
the official ``abc_atlas_access`` cache, and assembles the annotated object the
Session-2 spatial visualisation expects:

    reference/spatial_atlas/C57BL6J-638850-raw-meta.h5ad

``raw-meta`` = raw MERFISH counts (``.X``) + full per-cell annotations on ``.obs``:
  * TAXONOMY : class / subclass / supertype / cluster (+ ``*_color``), neurotransmitter
  * REGION   : CCF parcellation organ/category/division/structure/substructure
               (+ ``*_color``), parcellation_index
and three coordinate systems in ``.obsm``:
  * ``spatial_grid``          (raw MERFISH section x, y)   <- used by the notebook
  * ``spatial_reconstructed`` (registered/reconstructed x, y, z)
  * ``spatial_ccf``           (CCF x, y, z)

Both taxonomy and region come from a single ABC file,
``cell_metadata_with_parcellation_annotation`` (in the ``-CCF`` directory), so no
manual taxonomy join is needed.

This script is adapted from the spatial-tutorial bundle
(``/data/mouse_wb_spatial_tutorial/scripts/0_download_format_spatial_atlas.py``).
The workshop already ships the assembled atlas under ``cfg.TUTORIAL_DIR``; you
only need this to (re)build it on a fresh machine or refresh it from a newer ABC
release. To then keep just the 16 clean sections used by the notebook, subset the
output to the section labels in ``cfg.SPATIAL_ATLAS`` (the ``-16good-sections``
file) afterwards.

Docs / data browser: https://alleninstitute.github.io/abc_atlas_access/intro.html

REQUIREMENTS
    pip install "abc_atlas_access[notebooks] @ git+https://github.com/alleninstitute/abc_atlas_access.git"
    pip install anndata pandas numpy

DISK / NETWORK (first run only; cached afterwards)
    Expression matrix (MERFISH-C57BL6J-638850/raw) ~14.2 GB
    Per-cell metadata + gene metadata             ~1.7 GB
    Output -raw-meta.h5ad                          ~7.4 GB
    Budget ~25-30 GB of free space.

USAGE
    # runnable with no args (defaults from 00_config.py):
    python 00_download_spatial_atlas.py
    # or specify locations explicitly:
    python 00_download_spatial_atlas.py \
        --download-base /scratch/abc_atlas_cache \
        --out /data/mouse_wb_spatial_tutorial/reference/spatial_atlas/C57BL6J-638850-raw-meta.h5ad
"""
import argparse
import os
import sys
from datetime import date
from pathlib import Path

import anndata
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from importlib import import_module

cfg = import_module('00_config')

MERFISH_DIR = "MERFISH-C57BL6J-638850"
CCF_DIR = "MERFISH-C57BL6J-638850-CCF"
RAW_MATRIX = "C57BL6J-638850/raw"          # raw counts h5ad (use ".../log2" for log2)
CELL_PARC = "cell_metadata_with_parcellation_annotation"  # taxonomy + region + coords
GENE_META = "gene"                         # MERFISH gene panel metadata

# Label/color columns to store as categoricals.
CAT_COLS = ["brain_section_label", "neurotransmitter",
            "class", "subclass", "supertype", "cluster",
            "neurotransmitter_color", "class_color", "subclass_color",
            "supertype_color", "cluster_color",
            "parcellation_organ", "parcellation_category", "parcellation_division",
            "parcellation_structure", "parcellation_substructure",
            "parcellation_organ_color", "parcellation_category_color",
            "parcellation_division_color", "parcellation_structure_color",
            "parcellation_substructure_color"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download-base", type=Path,
                   default=Path("/scratch/abc_atlas_cache"),
                   help="Directory for the ABC Atlas download cache (created if missing).")
    p.add_argument("--out", type=Path, default=Path(cfg.SPATIAL_ATLAS_FULL),
                   help="Output path for the assembled C57BL6J-638850-raw-meta.h5ad "
                        f"(default: {cfg.SPATIAL_ATLAS_FULL}).")
    p.add_argument("--matrix", default=RAW_MATRIX,
                   help=f"Expression matrix file_name (default: {RAW_MATRIX}; "
                        "use 'C57BL6J-638850/log2' for log2-normalised).")
    p.add_argument("--pin-manifest", default=None,
                   help="Optional manifest path to pin a release for reproducibility, "
                        "e.g. 'releases/20250531/manifest.json'. Default: latest.")
    return p.parse_args()


def main():
    args = parse_args()
    args.download_base.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Imported here so the module's --help works without abc_atlas_access installed.
    from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

    # 1. Connect to the ABC Atlas S3-backed cache (no credentials needed).
    print(f"[1/5] Opening ABC Atlas cache at {args.download_base}")
    abc = AbcProjectCache.from_s3_cache(args.download_base)
    if args.pin_manifest:
        abc.load_manifest(args.pin_manifest)
    else:
        abc.load_latest_manifest()
    print("      manifest:", abc.current_manifest)

    # 2. Per-cell metadata WITH taxonomy + CCF parcellation annotations + coords.
    print("[2/5] Downloading cell metadata (taxonomy + region + coordinates)")
    cell = abc.get_metadata_dataframe(
        directory=CCF_DIR, file_name=CELL_PARC,
        dtype={"cell_label": str}, keep_default_na=False,
    ).set_index("cell_label")
    for axis in ("x", "y", "z"):
        cell[axis] = pd.to_numeric(cell[f"{axis}_section"], errors="coerce")
    UNASSIGNED_COLOR = "#d3d3d3"   # valid grey for empty color cells
    for col in CAT_COLS:
        if col not in cell.columns:
            continue
        if col.endswith("_color"):
            cell[col] = cell[col].replace({"": UNASSIGNED_COLOR,
                                           "unassigned": UNASSIGNED_COLOR}).astype("category")
        else:
            cell[col] = cell[col].replace("", "unassigned").astype("category")

    # 3. Gene panel metadata (gene_symbol, transcript_identifier, ...).
    print("[3/5] Downloading gene metadata")
    gene = abc.get_metadata_dataframe(directory=MERFISH_DIR, file_name=GENE_META)
    gene = gene.set_index(gene.columns[0])  # gene_identifier is the first column

    # 4. Raw expression matrix h5ad (~14 GB download on first run).
    print(f"[4/5] Downloading expression matrix '{args.matrix}' (large)")
    matrix_path = abc.get_file_path(directory=MERFISH_DIR, file_name=args.matrix)
    print("      reading", matrix_path)
    adata = anndata.read_h5ad(matrix_path)

    # 5. Align metadata to the matrix and assemble -raw-meta.h5ad.
    print("[5/5] Joining metadata onto expression and writing output")
    common = adata.obs_names.intersection(cell.index)
    if len(common) != adata.n_obs:
        print(f"      note: {adata.n_obs - len(common)} matrix cells lack CCF "
              "metadata; keeping them with region='unassigned'.")
    cell = cell.reindex(adata.obs_names)
    adata.obs = cell

    adata.obsm["spatial_grid"] = adata.obs[["x_section", "y_section"]].to_numpy(float)
    adata.obsm["spatial_reconstructed"] = adata.obs[
        ["x_reconstructed", "y_reconstructed", "z_reconstructed"]].apply(
        pd.to_numeric, errors="coerce").to_numpy(float)
    adata.obsm["spatial_ccf"] = adata.obs[
        ["x_ccf", "y_ccf", "z_ccf"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    g = gene.reindex(adata.var_names)
    for col in g.columns:
        adata.var[col] = g[col].values

    adata.uns["src"] = f"ABC Atlas {MERFISH_DIR} ({args.matrix}) + {CCF_DIR}/{CELL_PARC}"
    adata.uns["accessed_on"] = date.today().isoformat()

    adata.write_h5ad(args.out)
    print(f"\nDONE -> {args.out}")
    print(f"  {adata.n_obs:,} cells x {adata.n_vars} genes")
    print(f"  obsm: {list(adata.obsm.keys())}")


if __name__ == "__main__":
    main()
