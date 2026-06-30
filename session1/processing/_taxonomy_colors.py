"""Carry the curated V2 taxonomy colors from the processed atlas into a workshop
object.

The atlas stores, for each categorical annotation, a colour per category in
``uns['<col>_colors']`` (aligned to ``obs[col].cat.categories``). Those colours
were hand-picked to reflect the taxonomy, so re-using them makes the workshop
plots far easier to read. This module reads the atlas colour map and applies it to
a (subsampled) workshop object, also re-ordering each column's categories to match
the atlas/taxonomy order so legends group sensibly.
"""
import h5py
import numpy as np
import pandas as pd

_GREY = '#808080'


def _decode(arr):
    return [x.decode() if isinstance(x, bytes) else x for x in arr]


def read_atlas_color_maps(atlas_h5ad, cols):
    """Return ``{col: (ordered_categories, {category: hex_colour})}`` for *cols*.

    Only columns that are categorical in the atlas *and* carry a ``*_colors`` entry
    in ``uns`` are returned.
    """
    out = {}
    with h5py.File(atlas_h5ad, 'r') as f:
        for col in cols:
            ckey = col + '_colors'
            grp = f['obs'].get(col)
            if ckey not in f['uns'] or not isinstance(grp, h5py.Group):
                continue
            cats = _decode(grp['categories'][:])
            colors = _decode(f['uns'][ckey][:])
            out[col] = (cats, dict(zip(cats, colors)))
    return out


def apply_v2_colors(adata, atlas_h5ad, cols):
    """In place: order each ``cols`` column by taxonomy and set ``uns`` colours.

    Categories present in *adata* are re-ordered to the atlas order, and
    ``uns['<col>_colors']`` is written with one colour per (present) category so
    scanpy/cellxgene render the curated taxonomy palette.
    """
    color_maps = read_atlas_color_maps(atlas_h5ad, cols)
    for col, (atlas_order, cat2color) in color_maps.items():
        if col not in adata.obs:
            continue
        s = adata.obs[col]
        if not isinstance(s.dtype, pd.CategoricalDtype):
            s = s.astype('category')
        present = set(map(str, s.dropna().unique()))
        ordered = [c for c in atlas_order if c in present]
        # keep any present categories the atlas didn't list (defensive)
        ordered += [c for c in present if c not in set(ordered)]
        adata.obs[col] = s.cat.set_categories(ordered)
        adata.uns[col + '_colors'] = np.array(
            [cat2color.get(c, _GREY) for c in ordered], dtype=object)
        print(f'    {col}: {len(ordered)} categories coloured from atlas')
    return adata
