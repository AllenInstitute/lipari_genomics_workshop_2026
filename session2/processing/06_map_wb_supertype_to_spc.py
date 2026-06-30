"""Session 2 — REVERSE mapping at the **supertype** level: mouse-WB supertypes
-> our consensus spinal cord.

Identical in spirit to ``03_map_wb_to_spc.py`` but one taxonomy level finer: the
query is the subsampled mouse whole-brain **supertype** mean profiles built by
``05_build_supertype_means.py`` (instead of the 338 subclass means). Each mouse-WB
supertype is assigned the spinal-cord Class / Subclass / Group it most resembles,
giving the reverse arm of the *supertype*-level reciprocal mapping.

Run::  python 06_map_wb_supertype_to_spc.py
"""
import os
import subprocess
import sys

sys.path.append(os.path.dirname(__file__))
from importlib import import_module

cfg = import_module('00_config')


def run_mapper(query_h5ad: str, ref_path: str, out_dir: str) -> None:
    """query_markers + from_specified_markers (reverse onto the SpC ref)."""
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

    print('Assigning each mouse-WB supertype to a spinal-cord type...')
    # normalization=log2CPM: the supertype means are already log2(CPM+1) profiles.
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
    print('  reverse supertype mapping written to', results)


def main():
    cfg.set_all_seeds()
    if not os.path.exists(cfg.WB_SUPERTYPE_MEANS_H5AD):
        sys.exit('Supertype means missing — run 05_build_supertype_means.py first: '
                 + cfg.WB_SUPERTYPE_MEANS_H5AD)
    if not (os.path.exists(os.path.join(cfg.SPC_REF, 'precompute_stats.h5'))
            and os.path.exists(os.path.join(cfg.SPC_REF, 'reference_markers.h5'))):
        sys.exit('SpC V2 reference missing — run 02b_build_spc_v2_reference.py first: '
                 + cfg.SPC_REF)
    run_mapper(cfg.WB_SUPERTYPE_MEANS_H5AD, cfg.SPC_REF,
               cfg.REV_SUPERTYPE_MAPPING_DIR)
    print('DONE.')


if __name__ == '__main__':
    main()
