#!/bin/bash
# =============================================================================
# Session 2 — Build the NEW reverse reference: OUR consensus spinal cord
#
# Builds a cell_type_mapper reference from the AIBS consensus spinal-cord
# taxonomy (hierarchy Class -> Subclass -> Group -> consensus_cluster) so that we
# can map the mouse whole-brain subclasses ONTO our spinal cord (the reverse half
# of the reciprocal mapping, run by 03_map_wb_to_spc.py).
#
# A pre-built copy already lives at $REF_PATH (/scratch/SpC_consensus_ref); this
# script SKIPS the heavy build if it is present, and otherwise rebuilds it from
# the source taxonomy h5ad. Mirrors the reference-build block in the tutorial's
# scripts/1_run_mapping.sh.
#
# Run::  bash 02_build_spc_reference.sh
# =============================================================================
set -euo pipefail

pip install --quiet 'cell_type_mapper@git+https://github.com/AllenInstitute/cell_type_mapper'

REF_PATH="${REF_PATH:-/scratch/SpC_consensus_ref}"
REF_H5AD="${REF_H5AD:-/data/SpinalCord/manuscript/RNA/AIBS_SpC_consensus_taxonomy_harmonized_AIT-pre-print.h5ad}"
# Coarse -> fine taxonomy hierarchy of the consensus spinal-cord atlas.
HIERARCHY='["Class","Subclass","Group","consensus_cluster"]'

if [[ -f "$REF_PATH/precompute_stats.h5" && -f "$REF_PATH/reference_markers.h5" ]]; then
    echo "SpC consensus reference already present under $REF_PATH — skipping rebuild."
    echo "  (delete it to force a rebuild, or set REF_PATH to a new location.)"
    exit 0
fi

if [[ ! -f "$REF_H5AD" ]]; then
    echo "ERROR: consensus SpC taxonomy h5ad not found: $REF_H5AD" >&2; exit 1
fi
mkdir -p "$REF_PATH/temp"
echo "Building SpC consensus reference under $REF_PATH from:"
echo "  $REF_H5AD   hierarchy=$HIERARCHY"

# 1. Per-node expression statistics over the taxonomy hierarchy.
python -m cell_type_mapper.cli.precompute_stats_scrattch \
  --h5ad_path "$REF_H5AD" \
  --hierarchy "$HIERARCHY" \
  --output_path "$REF_PATH/precompute_stats.h5" \
  --normalization raw \
  --tmp_dir "$REF_PATH/temp/"
echo "precompute_stats done"

# 2. Marker genes distinguishing sibling nodes in the taxonomy.
python -m cell_type_mapper.cli.reference_markers \
  --precomputed_path_list '["'"$REF_PATH"'/precompute_stats.h5"]' \
  --output_dir "$REF_PATH/" \
  --tmp_dir "$REF_PATH/temp/"
echo "reference_markers done"

echo ""
echo "DONE. SpC consensus reference ready: $REF_PATH"
