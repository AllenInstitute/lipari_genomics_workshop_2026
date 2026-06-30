"""Session 2 — distill the forward + reverse mappings into small notebook tables.

Reads the two cell_type_mapper result files produced by 01 (forward: SpC cell ->
mouse-WB subclass) and 03 (reverse: mouse-WB subclass -> SpC type) and writes
three compact CSVs to /results that the student notebook loads directly:

  reciprocal_forward_summary.csv  per SpC Group_V2: the mouse-WB subclass it most
                                  maps to (top_wb_subclass, top_frac, n_cells).
  reciprocal_reverse_summary.csv  per mouse-WB subclass: the SpC Class/Subclass/
                                  Group it maps back to (+ correlation).
  reciprocal_best_hits.csv        the join: which SpC groups <-> mouse subclasses
                                  are reciprocal best hits.

Run::  python 04_build_reciprocal_artifacts.py
"""
import os
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from importlib import import_module

cfg = import_module('00_config')
mio = import_module('mapping_io')


def read_obs_column(f, name):
    g = f['obs'][name]
    if isinstance(g, h5py.Group):
        cats = [c.decode() if isinstance(c, bytes) else c for c in g['categories'][:]]
        cats = np.array(cats, dtype=object)
        codes = g['codes'][:]
        return pd.Series(np.where(codes >= 0, cats[np.clip(codes, 0, None)], None))
    arr = g[:]
    return pd.Series([x.decode() if isinstance(x, bytes) else x for x in arr])


def main():
    cfg.set_all_seeds()

    # ── FORWARD: SpC cell -> mouse-WB subclass ─────────────────────────────────
    print('Loading forward mapping:', cfg.FWD_RESULTS_JSON)
    fwd = mio.load_mapping_results(
        cfg.FWD_RESULTS_JSON, levels=[cfg.WB_SUBCLASS_LEVEL, cfg.WB_CLASS_LEVEL])

    print('Loading query obs (', cfg.QUERY_GROUP_KEY, '/', cfg.QUERY_SUBCLASS_KEY, ')...')
    with h5py.File(cfg.QUERY_H5AD, 'r') as f:
        idx = [x.decode() if isinstance(x, bytes) else x
               for x in f['obs']['_index'][:]]
        qobs = pd.DataFrame({
            cfg.QUERY_GROUP_KEY: read_obs_column(f, cfg.QUERY_GROUP_KEY).values,
            cfg.QUERY_SUBCLASS_KEY: read_obs_column(f, cfg.QUERY_SUBCLASS_KEY).values,
        }, index=idx)

    forward_summary = mio.summarize_forward_by_query_level(
        fwd, qobs, cfg.QUERY_GROUP_KEY, cfg.WB_SUBCLASS_LEVEL)
    forward_summary.to_csv(cfg.FORWARD_SUMMARY_CSV)
    print('  wrote', cfg.FORWARD_SUMMARY_CSV, forward_summary.shape)

    # ── REVERSE: mouse-WB subclass -> SpC type ─────────────────────────────────
    print('Loading reverse mapping:', cfg.REV_RESULTS_JSON)
    rev = mio.load_mapping_results(
        cfg.REV_RESULTS_JSON,
        levels=[cfg.SPC_CLASS_LEVEL, cfg.SPC_SUBCLASS_LEVEL, cfg.SPC_GROUP_LEVEL])
    rev.index.name = cfg.WB_SUBCLASS_LEVEL
    rev.to_csv(cfg.REVERSE_SUMMARY_CSV)
    print('  wrote', cfg.REVERSE_SUMMARY_CSV, rev.shape)

    # ── RECIPROCAL correspondence (pivot on mouse-WB subclass, Subclass-anchored)
    reciprocal = mio.reciprocal_by_wb_subclass(
        fwd, qobs, rev,
        query_subclass_key=cfg.QUERY_SUBCLASS_KEY,
        wb_subclass_level=cfg.WB_SUBCLASS_LEVEL,
        rev_subclass_level=cfg.SPC_SUBCLASS_LEVEL,
        overlap_min=cfg.OVERLAP_MIN,
        z=cfg.WILSON_Z)
    reciprocal.to_csv(cfg.RECIPROCAL_CSV)
    n_recip = int(reciprocal['reciprocal'].sum())
    print('  wrote', cfg.RECIPROCAL_CSV, reciprocal.shape,
          f'({n_recip} reciprocal mouse-WB subclasses)')
    print('\nReciprocal mouse-WB subclasses (forward & reverse agree on spinal '
          'Subclass, support-discounted overlap_lb >= '
          f'{cfg.OVERLAP_MIN}), top 20 by SpC cells:')
    cols = ['n_spc_cells', 'fwd_spc_subclass', 'overlap', 'overlap_lb',
            'rev_spc_subclass', 'rev_spc_group']
    print(reciprocal[reciprocal['reciprocal']].head(20)[cols].to_string())


if __name__ == '__main__':
    main()
