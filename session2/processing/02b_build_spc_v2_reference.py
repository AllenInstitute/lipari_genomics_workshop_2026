"""Session 2 — build the V2 reverse reference: OUR Session-1 spinal-cord taxonomy.

The original reverse reference (``02_build_spc_reference.sh``) is the AIBS
*consensus* SpC taxonomy. Its **Group** (and Class) names do not match the
Session-1 **V2** taxonomy carried by the workshop query (only Subclass names line
up), so a Group-level reciprocal mapping against it would mostly mismatch. To
reciprocate mouse-WB **supertypes against Group_V2** (and subclasses against
Subclass_V2) we instead build a cell_type_mapper reference straight from the
workshop query, using its own ``Class_V2 → Subclass_V2 → Group_V2`` hierarchy.

The V2 labels are written into plain ``Class``/``Subclass``/``Group`` columns so
the rest of the pipeline's level names stay valid; the reverse results then carry
V2 values directly comparable to the forward ``Subclass_V2`` / ``Group_V2``.

A handful of cells carry a Group_V2 whose Subclass_V2 disagrees with the group's
majority (cell_type_mapper requires a strict tree), so we resolve each
``Group → Subclass → Class`` edge by **majority vote**; the canonical query is
left untouched.

Run::  python 02b_build_spc_v2_reference.py
"""
import os
import subprocess
import sys

import anndata as ad
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from importlib import import_module

cfg = import_module('00_config')


def build_reference_input(out_h5ad: str) -> None:
    """Write a strict-tree reference-input AnnData (raw counts + V2 hierarchy)."""
    print('Reading workshop query:', cfg.SPC_V2_SOURCE_H5AD, flush=True)
    a = ad.read_h5ad(cfg.SPC_V2_SOURCE_H5AD)
    obs = a.obs
    for col in ('Class_V2', 'Subclass_V2', 'Group_V2'):
        if col not in obs.columns:
            sys.exit(f'Query is missing required column {col!r}')

    # Majority-vote each edge so Group->Subclass->Class is a strict tree.
    g2s = obs.groupby('Group_V2', observed=True)['Subclass_V2'].agg(
        lambda s: s.value_counts().idxmax())
    s2c = obs.groupby('Subclass_V2', observed=True)['Class_V2'].agg(
        lambda s: s.value_counts().idxmax())
    group = obs['Group_V2'].astype(str)
    subclass = group.map(g2s).astype(str)
    klass = subclass.map(s2c).astype(str)
    n_fixed = int((subclass.values != obs['Subclass_V2'].astype(str).values).sum())
    print(f'  {len(g2s)} groups, {len(s2c)} subclasses, '
          f'{obs["Class_V2"].nunique()} classes; '
          f'{n_fixed} cells reassigned to the majority subclass', flush=True)

    ref = ad.AnnData(
        X=a.X,                                   # RAW counts
        obs=pd.DataFrame({'Class': klass.values,
                          'Subclass': subclass.values,
                          'Group': group.values}, index=obs.index),
        var=a.var.copy(),
    )
    os.makedirs(os.path.dirname(out_h5ad), exist_ok=True)
    ref.write(out_h5ad)
    print(f'  wrote {out_h5ad}  ({ref.n_obs} cells x {ref.n_vars} genes)', flush=True)


def build_reference(ref_path: str, ref_input_h5ad: str) -> None:
    """precompute_stats + reference_markers over the V2 hierarchy."""
    hierarchy = '[%s]' % ','.join('"%s"' % h for h in cfg.SPC_V2_HIERARCHY)
    os.makedirs(os.path.join(ref_path, 'temp'), exist_ok=True)
    print(f'Building V2 reference under {ref_path}  hierarchy={hierarchy}', flush=True)

    subprocess.run([
        sys.executable, '-m', 'cell_type_mapper.cli.precompute_stats_scrattch',
        '--h5ad_path', ref_input_h5ad,
        '--hierarchy', hierarchy,
        '--output_path', os.path.join(ref_path, 'precompute_stats.h5'),
        '--normalization', 'raw',
        '--tmp_dir', os.path.join(ref_path, 'temp'),
    ], check=True)
    print('precompute_stats done', flush=True)

    subprocess.run([
        sys.executable, '-m', 'cell_type_mapper.cli.reference_markers',
        '--precomputed_path_list', '["%s/precompute_stats.h5"]' % ref_path,
        '--output_dir', ref_path + '/',
        '--tmp_dir', os.path.join(ref_path, 'temp'),
    ], check=True)
    print('reference_markers done', flush=True)


def main():
    cfg.set_all_seeds()
    ref_path = cfg.SPC_V2_REF
    if (os.path.exists(os.path.join(ref_path, 'precompute_stats.h5'))
            and os.path.exists(os.path.join(ref_path, 'reference_markers.h5'))):
        print(f'V2 reference already present under {ref_path} — skipping rebuild.')
        print('  (delete it to force a rebuild.)')
        return
    ref_input = os.path.join(ref_path, 'reference_input.h5ad')
    build_reference_input(ref_input)
    build_reference(ref_path, ref_input)
    print(f'\nDONE. SpC V2 reference ready: {ref_path}')


if __name__ == '__main__':
    main()
