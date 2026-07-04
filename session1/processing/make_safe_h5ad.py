#!/usr/bin/env python3
"""Sanitize an .h5ad so it loads cleanly in cellxgene.

cellxgene is strict about its schema: ``obs``/``var`` columns must be numeric or
categorical (no NaNs in numeric columns, bounded category counts), ``uns`` should
only carry ``*_colors`` palettes aligned to their categorical column, and stray
``raw``/``obsm``/``varm`` DataFrames or non-float ``X`` can trip it up. This script
coerces an object into that shape and writes a cleaned copy.

Usage:
    python make_safe_h5ad.py input.h5ad output.h5ad [--max-categories N]

Recommend copying the input to scratch first (the /uns prune step opens the input
file in r+ mode and mutates it in place).
"""

import argparse
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp


def stringify_with_nan(s: pd.Series) -> pd.Series:
    s = s.astype(object).copy()
    s[pd.isna(s)] = "nan"
    s[~pd.isna(s)] = s[~pd.isna(s)].map(str)
    return s


def process_df(df: pd.DataFrame, max_categories: int, axis_name: str) -> pd.DataFrame:
    out = {}
    dropped = []

    for col in df.columns:
        s = df[col]

        try:
            # categorical input -> convert to string categorical with "nan" for missing
            if pd.api.types.is_categorical_dtype(s):
                # Preserve the existing (possibly taxonomy-curated) category ORDER so
                # the matching uns['<col>_colors'] palette stays aligned.
                orig_order = [str(c) for c in s.cat.categories]
                s_str = stringify_with_nan(s)
                present = set(pd.Series(s_str).unique())
                n_unique = len(present)
                if n_unique == 0 or n_unique > max_categories:
                    dropped.append(col)
                    continue
                cats = [c for c in orig_order if c in present]
                cats += [c for c in present if c not in set(cats)]  # e.g. trailing "nan"
                out[col] = pd.Categorical(s_str, categories=cats)
                continue

            # booleans count as numeric; drop if missing, else float
            if pd.api.types.is_bool_dtype(s):
                if s.isna().any():
                    dropped.append(col)
                    continue
                out[col] = s.astype(np.float32)
                continue

            # native numeric -> drop if any NA, else float
            if pd.api.types.is_numeric_dtype(s):
                s_num = pd.to_numeric(s, errors="coerce")
                if s_num.isna().any():
                    dropped.append(col)
                    continue
                out[col] = s_num.astype(np.float32)
                continue

            # object/string: try exact numeric coercion first
            if s.dtype == object or pd.api.types.is_string_dtype(s):
                s_num = pd.to_numeric(s, errors="coerce")

                # treat as numeric only if every non-missing value parses as numeric
                orig_nonmissing = ~pd.isna(s)
                parsed_nonmissing = ~pd.isna(s_num)
                all_numeric = bool((parsed_nonmissing[orig_nonmissing]).all()) if orig_nonmissing.any() else False

                if all_numeric:
                    if s_num.isna().any():
                        dropped.append(col)
                        continue
                    out[col] = s_num.astype(np.float32)
                    continue

                s_str = stringify_with_nan(s)
                n_unique = int(pd.Series(s_str).nunique(dropna=False))
                if n_unique == 0 or n_unique > max_categories:
                    dropped.append(col)
                    continue
                out[col] = pd.Categorical(s_str)
                continue

            dropped.append(col)

        except Exception as e:
            warnings.warn(f"Dropping {axis_name}.{col!r} due to error: {e}")
            dropped.append(col)

    print(f"[{axis_name}] kept {len(out)} columns, dropped {len(dropped)}")
    if dropped:
        preview = dropped[:20]
        suffix = " ..." if len(dropped) > 20 else ""
        print(f"[{axis_name}] dropped columns: {preview}{suffix}")

    return pd.DataFrame(out, index=df.index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_h5ad")
    parser.add_argument("output_h5ad")
    parser.add_argument("--max-categories", type=int, default=10000)
    args = parser.parse_args()

    with h5py.File(args.input_h5ad, "r+") as f:
        if "uns" not in f:
            print("No /uns group found")
        else:
            uns = f["uns"]
            keys = list(uns.keys())  # copy to avoid mutation during iteration

            removed = []
            kept = []

            for k in keys:
                if "_colors" in k:
                    kept.append(k)
                else:
                    del uns[k]
                    removed.append(k)

            print(f"Removed {len(removed)} keys from /uns:")
            for k in removed:
                print(f"  - {k}")

            print(f"\nKept {len(kept)} keys:")
            for k in kept:
                print(f"  - {k}")

    adata = ad.read_h5ad(args.input_h5ad)

    # Capture the original <col>_colors as {category: hex} BEFORE we rebuild the
    # categoricals, so we can re-align palettes to the (possibly re-ordered) final
    # categories instead of relying on fragile positional alignment.
    orig_color_maps = {}
    for k in list(adata.uns.keys()):
        if isinstance(k, str) and k.endswith("_colors"):
            obs_key = k[:-7]
            if obs_key in adata.obs.columns and pd.api.types.is_categorical_dtype(
                adata.obs[obs_key]
            ):
                cats = [str(c) for c in adata.obs[obs_key].cat.categories]
                colors = list(adata.uns[k])
                if len(cats) == len(colors):
                    orig_color_maps[k] = dict(zip(cats, colors))

    obs_new = process_df(adata.obs.copy(), args.max_categories, "obs")
    var_new = process_df(adata.var.copy(), args.max_categories, "var")

    adata.obs = obs_new
    adata.var = var_new

    adata.obs_names = pd.Index([str(x) for x in adata.obs_names]).astype(str)
    adata.var_names = pd.Index([str(x) for x in adata.var_names]).astype(str)

    adata.obs.index = adata.obs_names
    adata.var.index = adata.var_names

    for k, v in list(adata.obsm.items()):
        if isinstance(v, pd.DataFrame) and len(v.index) == adata.n_obs:
            del adata.obsm[k]
            print("fixed obsm", k)

    for k, v in list(adata.varm.items()):
        if isinstance(v, pd.DataFrame) and len(v.index) == adata.n_vars:
            del adata.varm[k]
            print("fixed varm", k)

    if adata.raw is not None:
        del adata.raw

    if sp.issparse(adata.X):
        adata.X = adata.X.astype(np.float32).tocsc()
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)

    for k in list(adata.uns.keys()):
        if isinstance(k, str) and k.endswith("_colors"):
            obs_key = k[:-7]
            if (
                obs_key not in adata.obs.columns
                or not pd.api.types.is_categorical_dtype(adata.obs[obs_key])
            ):
                del adata.uns[k]
                continue

            # Re-align the palette to the FINAL category order (process_df may have
            # re-ordered, dropped, or appended a "nan" category). Missing categories
            # fall back to grey so lengths always match.
            cat2color = orig_color_maps.get(k)
            cats = [str(c) for c in adata.obs[obs_key].cat.categories]
            if cat2color is not None:
                adata.uns[k] = np.array(
                    [cat2color.get(c, "#808080") for c in cats], dtype=object
                )
            elif len(adata.uns[k]) != len(cats):
                del adata.uns[k]
        else:
            del adata.uns[k]

    adata.write_h5ad(args.output_h5ad, compression="gzip")

    print(f"Wrote: {args.output_h5ad}")


if __name__ == "__main__":
    main()
