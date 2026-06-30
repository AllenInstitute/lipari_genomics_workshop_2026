"""Session 2 — REVERSE mapping: mouse-WB subclasses -> our consensus spinal cord.

The reverse half of the reciprocal mapping. For every Allen mouse whole-brain
(AIT21) subclass we take its mean expression profile and map it onto the NEW
consensus spinal-cord reference built by ``02_build_spc_reference.sh``, asking:
"which spinal-cord cell type does each mouse whole-brain subclass resemble?"

Steps
-----
1. Build a tiny query AnnData from the AIT21 subclass-means CSV (338 subclasses ×
   32285 genes), one row per subclass, and write it to ``WB_MEANS_H5AD``.
2. Run cell_type_mapper (query_markers + from_specified_markers) against the SpC
   consensus reference, with ``normalization=log2CPM`` because the means are
   already log-normalized (matching how the bundled full 4M-cell AIT21->SpC map
   under ``…_ABC_MAPPING`` was produced).
3. The result (``REV_RESULTS_JSON``) carries, per mouse subclass, an assignment
   at every SpC level (Class / Subclass / Group / consensus_cluster).

Run::  python 03_map_wb_to_spc.py
"""
import os
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from importlib import import_module

cfg = import_module('00_config')


def build_means_query(out_h5ad: str) -> ad.AnnData:
    """Build the per-subclass mean-expression query AnnData from the CSV."""
    print('Reading subclass means:', cfg.SUBCLASS_MEANS_CSV)
    df = pd.read_csv(cfg.SUBCLASS_MEANS_CSV, index_col=0)
    # Rows = mouse-WB subclasses, columns = genes.
    X = df.to_numpy(dtype=np.float32)
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=df.index.astype(str)),
        var=pd.DataFrame(index=df.columns.astype(str)),
    )
    adata.obs.index.name = None
    os.makedirs(os.path.dirname(out_h5ad), exist_ok=True)
    adata.write(out_h5ad)
    print(f'  wrote {out_h5ad}  ({adata.n_obs} subclasses x {adata.n_vars} genes)')
    return adata


def run_mapper(query_h5ad: str, ref_path: str, out_dir: str) -> None:
    """Run query_markers + from_specified_markers (reverse onto the SpC ref)."""
    os.makedirs(out_dir, exist_ok=True)
    qmarkers = os.path.join(out_dir, 'query_markers.json')
    results = os.path.join(out_dir, 'hann_results.json')

    print('Selecting query markers against the SpC reference...')
    subprocess.run([
        sys.executable, '-m', 'cell_type_mapper.cli.query_markers',
        '--reference_marker_path_list',
        '["%s/reference_markers.h5"]' % ref_path,
        '--search_for_stats_file', 'True',
        '--output_path', qmarkers,
    ], check=True)

    print('Assigning each mouse-WB subclass to a spinal-cord type...')
    # normalization=log2CPM: the subclass means are already log-normalized.
    with open(os.path.join(out_dir, 'log_outputs.txt'), 'w') as log:
        subprocess.run([
            sys.executable, '-m', 'cell_type_mapper.cli.from_specified_markers',
            '--query_path', query_h5ad,
            '--query_markers.serialized_lookup', qmarkers,
            '--type_assignment.normalization', 'log2CPM',
            '--type_assignment.rng_seed', str(cfg.SEED),
            '--precomputed_stats.path', '%s/precompute_stats.h5' % ref_path,
            '--extended_result_path', results,
        ], check=True, stdout=log, stderr=subprocess.STDOUT)
    print('  reverse mapping written to', results)


def main():
    cfg.set_all_seeds()
    if not (os.path.exists(os.path.join(cfg.SPC_CONSENSUS_REF, 'precompute_stats.h5'))
            and os.path.exists(os.path.join(cfg.SPC_CONSENSUS_REF, 'reference_markers.h5'))):
        sys.exit('SpC consensus reference missing — run 02_build_spc_reference.sh first: '
                 + cfg.SPC_CONSENSUS_REF)
    build_means_query(cfg.WB_MEANS_H5AD)
    run_mapper(cfg.WB_MEANS_H5AD, cfg.SPC_CONSENSUS_REF, cfg.REV_MAPPING_DIR)
    print('DONE.')


if __name__ == '__main__':
    main()
