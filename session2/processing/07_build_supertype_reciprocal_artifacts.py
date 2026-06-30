"""Session 2 — distill the **overlap-coefficient** reciprocity tables (subclass +
supertype) into small notebook CSVs.

Builds on the forward map (01), the subclass reverse map (03) and the supertype
reverse map (06). Writes:

  reciprocal_subclass_overlap.csv   per mouse-WB subclass: best spinal partner by
                                    overlap coefficient, reverse hit, reciprocal flag.
  reciprocal_supertype_hits.csv     per mouse-WB supertype: parent subclass, both
                                    directions, inherited overlap, reciprocal flag.

Run::  python 07_build_supertype_reciprocal_artifacts.py
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
    fwd = mio.load_mapping_results(
        cfg.FWD_RESULTS_JSON, levels=[cfg.WB_SUBCLASS_LEVEL])
    with h5py.File(cfg.QUERY_H5AD, 'r') as f:
        idx = [x.decode() if isinstance(x, bytes) else x
               for x in f['obs']['_index'][:]]
        qobs = pd.DataFrame({
            cfg.QUERY_SUBCLASS_KEY: read_obs_column(f, cfg.QUERY_SUBCLASS_KEY).values,
        }, index=idx)

    # ── REVERSE (subclass) ─────────────────────────────────────────────────────
    rev = mio.load_mapping_results(
        cfg.REV_RESULTS_JSON,
        levels=[cfg.SPC_CLASS_LEVEL, cfg.SPC_SUBCLASS_LEVEL, cfg.SPC_GROUP_LEVEL])

    sub_overlap = mio.reciprocal_subclass_overlap(
        fwd, qobs, rev,
        query_subclass_key=cfg.QUERY_SUBCLASS_KEY,
        wb_subclass_level=cfg.WB_SUBCLASS_LEVEL,
        rev_subclass_level=cfg.SPC_SUBCLASS_LEVEL,
        min_overlap=cfg.OVERLAP_MIN)
    sub_overlap.to_csv(cfg.SUBCLASS_OVERLAP_CSV)
    n_sub = int(sub_overlap['reciprocal'].sum())
    print(f'wrote {cfg.SUBCLASS_OVERLAP_CSV}  ({n_sub} reciprocal subclasses '
          f'at overlap >= {cfg.OVERLAP_MIN})')

    # ── REVERSE (supertype) + reciprocity ──────────────────────────────────────
    rev_st = mio.load_mapping_results(
        cfg.REV_SUPERTYPE_RESULTS_JSON,
        levels=[cfg.SPC_CLASS_LEVEL, cfg.SPC_SUBCLASS_LEVEL, cfg.SPC_GROUP_LEVEL])
    st2sc = pd.read_csv(cfg.WB_SUPERTYPE_TO_SUBCLASS_CSV, index_col=0)

    st_recip = mio.reciprocal_supertypes(
        rev_st, st2sc, sub_overlap,
        rev_subclass_level=cfg.SPC_SUBCLASS_LEVEL,
        min_overlap=cfg.OVERLAP_MIN)
    st_recip.to_csv(cfg.RECIPROCAL_SUPERTYPE_CSV)
    n_st = int(st_recip['reciprocal'].sum())
    print(f'wrote {cfg.RECIPROCAL_SUPERTYPE_CSV}  ({n_st} reciprocal supertypes '
          f'of {len(st_recip)} at overlap >= {cfg.OVERLAP_MIN})')

    cols = ['parent_wb_subclass', 'fwd_spc_subclass', 'rev_spc_subclass', 'overlap']
    print('\nTop reciprocal supertypes by overlap:')
    print(st_recip[st_recip['reciprocal']].head(15)[cols].to_string())


if __name__ == '__main__':
    main()
