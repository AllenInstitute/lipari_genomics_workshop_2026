#!/bin/bash
# =============================================================================
# Session 2 — FORWARD mapping
#   SpC snRNA query  ->  Allen mouse whole-brain (AIT21) SUBCLASS labels
#
# Uses the AllenInstitute cell_type_mapper (the engine behind MapMyCells) with
# the pre-built mouse-WB subclass reference shipped in the spatial tutorial
# bundle. Answers: "which mouse whole-brain cell types did our spinal cord
# capture?" — the forward half of the reciprocal mapping.
#
# Adapted from /data/mouse_wb_spatial_tutorial/scripts/1_run_mapping.sh.
#
# Run::  bash 01_map_spc_to_wb_subclass.sh
# =============================================================================
set -euo pipefail

# 0. Install the mapper (idempotent; pin a commit in the cloud for reproducibility).
pip install --quiet 'cell_type_mapper@git+https://github.com/AllenInstitute/cell_type_mapper'

# 1. Paths (kept in sync with 00_config.py).
QUERY_H5AD="${QUERY_H5AD:-/results/SpC_workshop_snRNA.h5ad}"
REF_PATH="${REF_PATH:-/data/mouse_wb_spatial_tutorial/reference/subclass_mapping_reference}"
OUT_PATH="${OUT_PATH:-/results/SpC_workshop_WB_SUBCLASS_MAPPING}"

# 2. Sanity checks.
if [[ ! -f "$QUERY_H5AD" ]]; then
    echo "ERROR: query h5ad not found: $QUERY_H5AD (run Session 1 first)" >&2; exit 1
fi
if [[ ! -f "$REF_PATH/precompute_stats.h5" || ! -f "$REF_PATH/reference_markers.h5" ]]; then
    echo "ERROR: mouse-WB subclass reference incomplete under $REF_PATH" >&2; exit 1
fi
mkdir -p "$OUT_PATH"
echo "Query     : $QUERY_H5AD"
echo "Reference : $REF_PATH (mouse WB, subclass level)"
echo "Output    : $OUT_PATH"

# 3. Select the reference markers usable for THIS query (genes present in both).
python -m cell_type_mapper.cli.query_markers \
  --reference_marker_path_list '["'"$REF_PATH"'/reference_markers.h5"]' \
  --search_for_stats_file True \
  --output_path "$OUT_PATH/query_markers.json"
echo "query_markers done"

# 4. Hierarchical, correlation-based assignment of every query cell.
#    --type_assignment.normalization raw : the workshop query .X holds RAW counts
#    (the mapper log2(CPM+1)-normalizes internally). Bootstrapping (on by default)
#    yields the per-cell probability we use downstream as a confidence score.
python -m cell_type_mapper.cli.from_specified_markers \
  --query_path "$QUERY_H5AD" \
  --query_markers.serialized_lookup "$OUT_PATH/query_markers.json" \
  --type_assignment.normalization raw \
  --type_assignment.rng_seed 0 \
  --precomputed_stats.path "$REF_PATH/precompute_stats.h5" \
  --extended_result_path "$OUT_PATH/hann_results.json" \
  > "$OUT_PATH/log_outputs.txt" 2>&1
echo "from_specified_markers done"

echo ""
echo "DONE. Forward mapping: $OUT_PATH/hann_results.json"
