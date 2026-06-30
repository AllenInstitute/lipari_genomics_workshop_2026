"""Session 2 — build a subsampled mouse-WB (AIT21 / ABC) snRNA dataset + supertype means.

The reverse arm of the reciprocal mapping (``03_map_wb_to_spc.py``) only resolved
the mouse whole-brain taxonomy to the **subclass** level, because that is the
granularity at which Allen ships pre-computed mean profiles
(``AIT21…_subclassmeans.csv``). To extend reciprocity to the finer **supertype**
level we subsample the full 224 GB ``AIT21`` matrix down to a small, student-sized
dataset and average per supertype.

Why a streaming subsample? The matrix is CSR on slow network storage where
*random* row reads cost ~1 s each (hopeless for 10⁵ cells), so we make a single
**sequential, parallel** pass over contiguous row-blocks, keeping up to
``N_PER_SUPERTYPE`` seeded cells per supertype.

Outputs (to ``/results`` — small enough to ship to students; the 224 GB original is not)
----------------------------------------------------------------------------------------
``wb_subsampled_ABC.h5ad``        ~N_PER_SUPERTYPE×1201 cells × 32285 genes, RAW counts,
                                  obs = class/subclass/supertype labels.
``wb_supertype_means.h5ad``       1201 supertypes × 32285 genes, log2(CPM+1) means.
``wb_supertype_to_subclass.csv``  supertype_label -> subclass_label -> class_label.

Run::  python 05_build_supertype_means.py            # 150 cells/supertype, 8 workers
       N_PER_SUPERTYPE=100 N_WORKERS=12 python 05_build_supertype_means.py
"""
import os
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.append(os.path.dirname(__file__))
from importlib import import_module

cfg = import_module('00_config')

N_PER_SUPERTYPE = int(os.environ.get('N_PER_SUPERTYPE', '150'))
N_WORKERS = int(os.environ.get('N_WORKERS', '8'))
ROW_BLOCK = int(os.environ.get('ROW_BLOCK', '40000'))


def _decode(arr):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in arr],
                    dtype=object)


def _read_categorical(g):
    return _decode(g['categories'][:]), g['codes'][:]


def _select_rows(st_codes, n_super, seed, n_per):
    """Seeded choice of <= n_per row indices per supertype; returns a boolean
    keep-mask over all cells."""
    rng = np.random.default_rng(seed)
    keep = np.zeros(st_codes.shape[0], dtype=bool)
    order = np.argsort(st_codes, kind='stable')       # group rows by supertype
    sorted_codes = st_codes[order]
    bounds = np.flatnonzero(np.diff(sorted_codes)) + 1
    for s, e in zip(np.r_[0, bounds], np.r_[bounds, sorted_codes.size]):
        idx = order[s:e]
        if idx.size > n_per:
            idx = rng.choice(idx, size=n_per, replace=False)
        keep[idx] = True
    return keep


def _extract_range(args):
    """Worker: stream one contiguous row-range, return RAW-count CSR of kept rows."""
    r0, r1, keep_path, tmp_dir = args
    keep = np.load(keep_path, mmap_mode='r')
    pieces = []
    with h5py.File(cfg.AIT21_H5AD, 'r') as f:
        indptr = f['X']['indptr']
        data_ds, indices_ds = f['X']['data'], f['X']['indices']
        n_genes = int(f['X'].attrs['shape'][1])
        for b0 in range(r0, r1, ROW_BLOCK):
            b1 = min(b0 + ROW_BLOCK, r1)
            a0, a1 = int(indptr[b0]), int(indptr[b1])
            local_ptr = indptr[b0:b1 + 1][:] - a0
            data = data_ds[a0:a1]
            cols = indices_ds[a0:a1].astype(np.int32)
            block = sparse.csr_matrix(
                (data, cols, local_ptr), shape=(b1 - b0, n_genes))
            m = np.asarray(keep[b0:b1])
            if m.any():
                pieces.append(block[m])
    out = (sparse.vstack(pieces, format='csr') if pieces
           else sparse.csr_matrix((0, n_genes)))
    tmp = os.path.join(tmp_dir, f'sub_{r0}.npz')
    sparse.save_npz(tmp, out)
    return r0, tmp, int(out.shape[0])


def main():
    from multiprocessing import Pool
    cfg.set_all_seeds()

    with h5py.File(cfg.AIT21_H5AD, 'r') as f:
        n_cells = f['obs']['supertype_label']['codes'].shape[0]
        var_names = _decode(f['var']['_index'][:])
        st_cats, st_codes = _read_categorical(f['obs']['supertype_label'])
        sc_cats, sc_codes = _read_categorical(f['obs']['subclass_label'])
        cl_cats, cl_codes = _read_categorical(f['obs']['class_label'])
    n_super = len(st_cats)
    print(f'{n_cells:,} cells; {n_super} supertypes; {len(var_names)} genes',
          flush=True)

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    tmp_dir = os.path.join(cfg.RESULTS_DIR, '_subsample_tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    keep = _select_rows(st_codes, n_super, cfg.SEED, N_PER_SUPERTYPE)
    keep_path = os.path.join(tmp_dir, 'keep.npy')
    np.save(keep_path, keep)
    print(f'keeping {int(keep.sum()):,} cells (<= {N_PER_SUPERTYPE}/supertype); '
          f'streaming with {N_WORKERS} workers', flush=True)

    bounds = np.linspace(0, n_cells, N_WORKERS + 1).astype(int)
    jobs = [(int(bounds[i]), int(bounds[i + 1]), keep_path, tmp_dir)
            for i in range(N_WORKERS)]
    with Pool(N_WORKERS) as p:
        results = p.map(_extract_range, jobs)

    results.sort(key=lambda t: t[0])
    Xsub = sparse.vstack([sparse.load_npz(t[1]) for t in results], format='csr')
    kept_rows = np.flatnonzero(keep)            # global indices, in row order
    assert Xsub.shape[0] == kept_rows.size, (Xsub.shape[0], kept_rows.size)
    print(f'assembled subsample: {Xsub.shape[0]:,} cells x {Xsub.shape[1]} genes',
          flush=True)

    # ── Student-facing subsampled ABC dataset (RAW counts) ─────────────────────
    obs = pd.DataFrame({
        'class_label': cl_cats[cl_codes[kept_rows]],
        'subclass_label': sc_cats[sc_codes[kept_rows]],
        'supertype_label': st_cats[st_codes[kept_rows]],
    }).astype(str)
    obs.index = pd.Index([f'AIT21_{i}' for i in kept_rows], name=None)
    adata = ad.AnnData(
        X=Xsub,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(var_names.astype(str), name=None)),
    )
    adata.write(cfg.WB_SUBSAMPLED_ABC_H5AD, compression='gzip')
    print(f'wrote {cfg.WB_SUBSAMPLED_ABC_H5AD}', flush=True)

    # ── Supertype mean profiles (log2(CPM+1) average over the subsample) ───────
    totals = np.asarray(Xsub.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    Xln = Xsub.multiply(1e6 / totals[:, None]).tocsr()
    Xln.data = np.log2(Xln.data + 1.0)
    codes = st_codes[kept_rows]
    present = np.unique(codes)
    ind = sparse.csr_matrix(
        (np.ones(codes.size), (np.searchsorted(present, codes),
                               np.arange(codes.size))),
        shape=(present.size, codes.size))
    counts = np.asarray(ind.sum(axis=1)).ravel()
    means = np.asarray((ind @ Xln).todense()) / counts[:, None]
    means_adata = ad.AnnData(
        X=means.astype(np.float32),
        obs=pd.DataFrame(index=pd.Index(st_cats[present].astype(str), name=None)),
        var=pd.DataFrame(index=pd.Index(var_names.astype(str), name=None)),
    )
    means_adata.write(cfg.WB_SUPERTYPE_MEANS_H5AD)
    print(f'wrote {cfg.WB_SUPERTYPE_MEANS_H5AD}  '
          f'({means_adata.n_obs} supertypes x {means_adata.n_vars} genes)',
          flush=True)

    # ── supertype -> subclass -> class lookup ──────────────────────────────────
    map_rows = []
    for code in present:
        m = codes == code
        sc_mode = np.bincount(sc_codes[kept_rows][m]).argmax()
        cl_mode = np.bincount(cl_codes[kept_rows][m]).argmax()
        map_rows.append((st_cats[code], sc_cats[sc_mode], cl_cats[cl_mode]))
    pd.DataFrame(map_rows,
                 columns=['supertype_label', 'subclass_label', 'class_label']
                 ).set_index('supertype_label').to_csv(
        cfg.WB_SUPERTYPE_TO_SUBCLASS_CSV)
    print(f'wrote {cfg.WB_SUPERTYPE_TO_SUBCLASS_CSV}', flush=True)

    for _, tmp, _ in results:
        os.remove(tmp)
    os.remove(keep_path)
    os.rmdir(tmp_dir)
    print('DONE.', flush=True)


if __name__ == '__main__':
    main()
