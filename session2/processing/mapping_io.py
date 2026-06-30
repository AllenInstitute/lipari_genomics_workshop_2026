"""
mapping_io.py
=============
Load cell_type_mapper ("MapMyCells"-style) results and wire up a *reciprocal*
mapping between a query dataset and the Allen mouse whole-brain taxonomy.

Two directions:

  FORWARD  query cell  ->  mouse WB subclass label
           `load_mapping_results()` parses hann_results.json into a tidy
           DataFrame (one row per query cell, with assignment + confidence at
           every taxonomy level) and `attach_mapping_to_adata()` writes those
           columns onto a query AnnData's .obs.

  REVERSE  detected subclass  ->  WHERE it lives in the mouse brain
           Using the mouse WB MERFISH spatial atlas as a lookup, we ask: for
           each subclass the query "detected", where does that subclass sit in
           physical brain space? `build_spatial_reference()` +
           `representation_table()` + `attach_representation_to_spatial()`
           produce per-subclass representation scores and paint them onto the
           spatial atlas cells so you can plot the reciprocal mapping.

The pair of directions is what we mean by a "reciprocal mapping setup": the
query tells you which mouse types are present, and the spatial atlas tells you
where those types are anatomically.

Designed to be imported from notebooks/2_explore_and_plot.ipynb, but every
function is standalone and documented.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# FORWARD direction: load query -> taxonomy assignments
# ---------------------------------------------------------------------------
def load_mapping_results(
    results_path: str | Path,
    levels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Parse a cell_type_mapper ``hann_results.json`` into a tidy DataFrame.

    Each query cell becomes one row, indexed by ``cell_id``. For every taxonomy
    level present (e.g. ``class_label``, ``subclass_label``) we extract:

      * ``<level>``                 the assigned node (the ``assignment`` field)
      * ``<level>_prob``            bootstrapping_probability (confidence, 0-1)
      * ``<level>_corr``            avg_correlation, if present
      * ``<level>_runner_up``       top runner-up assignment, if present

    Parameters
    ----------
    results_path
        Path to ``hann_results.json`` produced by
        ``cell_type_mapper.cli.from_specified_markers``.
    levels
        Restrict to these level keys (e.g. ``["subclass_label"]``). If None,
        every per-cell dict field that looks like an assignment is used.

    Returns
    -------
    pandas.DataFrame indexed by ``cell_id``.
    """
    results_path = Path(results_path)
    with results_path.open() as f:
        payload = json.load(f)

    results = payload["results"]
    if not results:
        raise ValueError(f"No results in {results_path}")

    # Auto-detect level keys: per-cell fields whose value is a dict with an
    # "assignment" entry (cell_id is a plain string, so it is skipped).
    if levels is None:
        sample = results[0]
        levels = [
            k for k, v in sample.items()
            if isinstance(v, dict) and "assignment" in v
        ]
    if not levels:
        raise ValueError("Could not find any taxonomy levels in results.")

    rows = []
    for rec in results:
        row = {"cell_id": rec["cell_id"]}
        for lvl in levels:
            node = rec.get(lvl)
            if not isinstance(node, dict):
                continue
            row[lvl] = node.get("assignment")
            if "bootstrapping_probability" in node:
                row[f"{lvl}_prob"] = node["bootstrapping_probability"]
            if "avg_correlation" in node:
                row[f"{lvl}_corr"] = node["avg_correlation"]
            ru = node.get("runner_up_assignment")
            if isinstance(ru, (list, tuple)) and len(ru) > 0:
                row[f"{lvl}_runner_up"] = ru[0]
        rows.append(row)

    df = pd.DataFrame(rows).drop_duplicates("cell_id").set_index("cell_id")
    return df


def attach_mapping_to_adata(
    adata,
    mapping_df: pd.DataFrame,
    levels: Sequence[str] | None = None,
    as_categorical: bool = True,
):
    """Map columns of ``mapping_df`` onto ``adata.obs`` by matching obs index.

    Mirrors the cetacean notebook's
    ``adata.obs['subclass_label'] = adata.obs.index.map(mapping_df[...])``
    pattern, but copies assignment + confidence columns for every level.

    Returns the (modified, in place) ``adata`` for chaining.
    """
    if levels is None:
        # Treat any non-suffixed column as a level (the assignment columns).
        levels = [c for c in mapping_df.columns
                  if not c.endswith(("_prob", "_corr", "_runner_up"))]

    idx = adata.obs.index.to_series()
    for lvl in levels:
        for col in [lvl, f"{lvl}_prob", f"{lvl}_corr", f"{lvl}_runner_up"]:
            if col not in mapping_df.columns:
                continue
            mapped = idx.map(mapping_df[col])
            if as_categorical and col == lvl:
                adata.obs[col] = pd.Categorical(mapped)
            else:
                adata.obs[col] = mapped.values
    return adata


# ---------------------------------------------------------------------------
# REVERSE direction: project detected types onto the spatial atlas
# ---------------------------------------------------------------------------
def build_spatial_reference(
    atlas_h5ad: str | Path,
    subclass_key: str = "subclass",
    section_key: str = "brain_section_label",
):
    """Load the mouse WB MERFISH atlas and precompute spatial specificity.

    "Spatial breadth" of a subclass = fraction of brain sections it appears in.
    "Specificity" = 1 - breadth. Broad types (Oligo, Astro, Pvalb ...) are poor
    evidence that any particular region was sampled, so specificity lets us
    down-weight them when scoring regional coverage in the reverse direction.

    Returns
    -------
    (adata_atlas, specificity_series)
        adata_atlas         : the loaded spatial AnnData (full).
        specificity_series  : pandas.Series, index=subclass, values in [0, 1].
    """
    import scanpy as sc  # imported lazily so the module loads without scanpy

    adata = sc.read_h5ad(str(atlas_h5ad))
    n_sections = adata.obs[section_key].nunique()
    breadth = (
        adata.obs.groupby(subclass_key, observed=True)[section_key]
        .nunique()
        .div(n_sections)
    )
    specificity = (1.0 - breadth).rename("spatial_specificity")
    return adata, specificity


def representation_table(
    detected_counts: Mapping | pd.Series,
    reference_subclasses: Iterable[str],
    full_n: int = 500,
    specificity: pd.Series | None = None,
) -> pd.DataFrame:
    """Score, per reference subclass, how well the query "detected" it.

    Parameters
    ----------
    detected_counts
        Number of query cells assigned to each subclass (e.g.
        ``mapping_df['subclass_label'].value_counts()``).
    reference_subclasses
        The subclasses defined by the reference (used to align / fill zeros).
    full_n
        Cell count at which a subclass is considered fully "represented".
        ``representation_score = min(n_query / full_n, 1)``.
    specificity
        Optional per-subclass spatial specificity from
        ``build_spatial_reference``; merged in for downstream weighting.

    Returns
    -------
    DataFrame indexed by subclass with columns:
    ``n_query``, ``representation_score`` (and ``spatial_specificity`` if given).
    """
    ref_index = pd.Index(list(reference_subclasses)).unique()
    qry = pd.Series(detected_counts).reindex(ref_index, fill_value=0)
    score = (qry / float(full_n)).clip(upper=1.0)

    out = pd.DataFrame({
        "n_query": qry,
        "representation_score": score,
    })
    if specificity is not None:
        out["spatial_specificity"] = specificity.reindex(ref_index, fill_value=0.0)
    out.index.name = "subclass"
    return out


def attach_representation_to_spatial(
    adata_atlas,
    rep_df: pd.DataFrame,
    subclass_key: str = "subclass",
    score_col: str = "representation_score",
    out_col: str = "representation_score",
):
    """Paint per-subclass representation scores onto every spatial atlas cell.

    Lets you colour the MERFISH atlas by "did the query detect the type that
    lives here?" -- i.e. plot the reverse/reciprocal mapping in brain space.

    Returns the (modified, in place) ``adata_atlas``.
    """
    score_map = rep_df[score_col].to_dict()
    adata_atlas.obs[out_col] = (
        adata_atlas.obs[subclass_key].map(score_map).astype(float)
    )
    return adata_atlas


# ---------------------------------------------------------------------------
# CCF region boundaries -> vector polygons (for overlaying on a section)
# ---------------------------------------------------------------------------
def load_annotation_volume(abc_cache, directory="MERFISH-C57BL6J-638850-CCF",
                           file_name="resampled_annotation"):
    """Load the resampled CCF parcellation volume for the MERFISH dataset.

    Returns ``(array, spacing)`` where ``array`` is the SimpleITK view in
    ``[z, y, x]`` order (z = section axis) and ``spacing`` is ``(sx, sy, sz)``
    in mm. For ``MERFISH-C57BL6J-638850-CCF`` the volume is (76, 1100, 1100)
    with spacing (0.01, 0.01, 0.2) mm (10 µm in-plane, 200 µm between sections).

    ``abc_cache`` is an ``AbcProjectCache`` instance. SimpleITK is imported
    lazily so the rest of this module loads without it.
    """
    import SimpleITK as sitk

    path = abc_cache.get_file_path(directory=directory, file_name=file_name)
    img = sitk.ReadImage(str(path))
    # GetArrayFromImage COPIES (unlike GetArrayViewFromImage, whose buffer is
    # freed with `img` once this function returns). [z, y, x] order.
    arr = sitk.GetArrayFromImage(img)
    sx, sy, sz = img.GetSpacing()                  # ITK reports (x, y, z)
    return arr, (sx, sy, sz)


def section_zindex(z_reconstructed, z_spacing=0.2):
    """Map a section's ``z_reconstructed`` (mm) to its plane index in the volume."""
    return int(round(float(z_reconstructed) / z_spacing))


def extract_section_polygons(label_slice, xy_spacing=0.01, min_vertices=6,
                             drop_labels=(0,)):
    """Contour a 2-D parcellation label image into per-region vector polygons.

    For each region id present in ``label_slice`` we trace the region boundary
    with ``skimage.measure.find_contours`` on a binary mask, converting voxel
    (row, col) vertices to (x, y) millimetres via ``xy_spacing``.

    Parameters
    ----------
    label_slice
        2-D ndarray of parcellation_index values (one z-plane of the volume,
        i.e. ``volume[zindex]``).
    xy_spacing
        In-plane voxel size in mm (0.01 for this dataset).
    min_vertices
        Skip tiny contours with fewer than this many vertices.
    drop_labels
        Region ids to ignore (default: 0 = unassigned/background).

    Returns
    -------
    list of dicts: ``{"parcellation_index": int, "xy": (N, 2) ndarray in mm}``.
    One entry per contour (a region can yield several disjoint contours).
    """
    from skimage import measure

    polys = []
    # The volume marks out-of-brain voxels as NaN (not 0); skip non-finite and
    # any explicitly dropped labels.
    uniq = np.unique(np.asarray(label_slice))
    labels = [int(v) for v in uniq
              if np.isfinite(v) and int(v) not in drop_labels]
    for lab in labels:
        mask = (label_slice == lab)
        for contour in measure.find_contours(mask.astype(float), level=0.5):
            if len(contour) < min_vertices:
                continue
            # find_contours returns (row=y, col=x); convert to (x, y) mm.
            xy = np.column_stack([contour[:, 1], contour[:, 0]]) * xy_spacing
            polys.append({"parcellation_index": lab, "xy": xy})
    return polys


# ---------------------------------------------------------------------------
# Convenience: end-to-end reciprocal load in one call
# ---------------------------------------------------------------------------
def load_reciprocal(
    results_path: str | Path,
    atlas_h5ad: str | Path,
    subclass_level: str = "subclass_label",
    atlas_subclass_key: str = "subclass",
    full_n: int = 500,
):
    """Run the full reciprocal load and return everything a notebook needs.

    Returns a dict with:
      ``mapping_df``  forward query->taxonomy assignments
      ``adata_atlas`` spatial atlas with ``representation_score`` on .obs
      ``rep_df``      per-subclass representation + specificity table
      ``specificity`` per-subclass spatial specificity series
    """
    mapping_df = load_mapping_results(results_path, levels=[subclass_level])
    adata_atlas, specificity = build_spatial_reference(
        atlas_h5ad, subclass_key=atlas_subclass_key
    )
    rep_df = representation_table(
        detected_counts=mapping_df[subclass_level].value_counts(),
        reference_subclasses=adata_atlas.obs[atlas_subclass_key].cat.categories
        if hasattr(adata_atlas.obs[atlas_subclass_key], "cat")
        else adata_atlas.obs[atlas_subclass_key].unique(),
        full_n=full_n,
        specificity=specificity,
    )
    attach_representation_to_spatial(adata_atlas, rep_df,
                                     subclass_key=atlas_subclass_key)
    return {
        "mapping_df": mapping_df,
        "adata_atlas": adata_atlas,
        "rep_df": rep_df,
        "specificity": specificity,
    }


# ---------------------------------------------------------------------------
# REVERSE direction helpers: mouse-WB subclass -> our spinal-cord taxonomy
# ---------------------------------------------------------------------------
def normalize_label(s) -> str:
    """Collapse whitespace/underscores so two taxonomies' labels can be compared.

    The Session-1 query carries the HMBA ``*_V2`` spinal taxonomy while the
    reverse reference carries the AIBS *consensus* taxonomy; the same cell type
    may be written ``NN AIF1 Microglia`` vs ``NN_AIF1_Microglia`` (or ``Glut-D``
    identically). Normalizing lets us match the levels they share — in
    particular the neuronal **Subclass** vocabulary (Glut-D/M/V, GABA-D/M/V…).
    """
    import re
    return re.sub(r"[\s_]+", " ", str(s)).strip()


def summarize_forward_by_query_level(
    mapping_df: pd.DataFrame,
    query_obs: pd.DataFrame,
    query_level_key: str,
    subclass_level: str = "subclass_label",
) -> pd.DataFrame:
    """Per spinal-cord query label, the mouse-WB subclass it most often maps to.

    Joins the forward per-cell assignments (``mapping_df``) back to the query
    cells' own spinal-cord label (``query_obs[query_level_key]`` — a Group or a
    Subclass) and, for every value, reports the modal (top) detected mouse-WB
    subclass and the fraction of that label's cells assigned to it.

    Returns
    -------
    DataFrame indexed by the spinal-cord label with columns
    ``top_wb_subclass``, ``top_frac``, ``n_cells``.
    """
    df = query_obs[[query_level_key]].join(mapping_df[[subclass_level]], how="inner")
    df = df.dropna(subset=[query_level_key, subclass_level])
    rows = []
    for lab, sub in df.groupby(query_level_key, observed=True):
        vc = sub[subclass_level].value_counts()
        if vc.empty:
            continue
        rows.append({
            query_level_key: lab,
            "top_wb_subclass": vc.index[0],
            "top_frac": float(vc.iloc[0]) / float(len(sub)),
            "n_cells": int(len(sub)),
        })
    return pd.DataFrame(rows).set_index(query_level_key).sort_values(
        "n_cells", ascending=False)


def reciprocal_by_wb_subclass(
    mapping_df: pd.DataFrame,
    query_obs: pd.DataFrame,
    reverse_df: pd.DataFrame,
    query_subclass_key: str,
    wb_subclass_level: str = "subclass_label",
    rev_subclass_level: str = "Subclass",
    min_cells: int = 1,
    overlap_min: float = 0.0,
    z: float = 1.96,
) -> pd.DataFrame:
    """Reciprocal correspondence, pivoted on each mouse-WB **subclass** M.

    For every mouse-WB subclass M:

      * FORWARD — among our spinal cells that mapped to M, the modal spinal
        **Subclass** (``fwd_spc_subclass``); this is the spinal type that most
        "votes" for M.
      * REVERSE — the spinal **Subclass** that M maps back to when projected onto
        our V2 reference (``rev_spc_subclass``).

    M is a **reciprocal** hit when those agree (after ``normalize_label``) **and**
    the forward vote clears a support-discounted gate. We anchor on Subclass
    because it is the level the two spinal taxonomies share cleanly (Glut-D/M/V,
    GABA-D/M/V, Astro, Oligo, OPC…), unlike Group/Class.

    The forward signal is graded by ``overlap_lb`` — the Wilson lower confidence
    bound on the modal-vote fraction ``p = (cells voting for fwd_spc) /
    n_spc_cells`` with ``n = n_spc_cells``. This mirrors the overlap-coefficient
    arms: a mouse subclass backed by only a handful of spinal cells (or whose
    votes are diffuse) gets a low ``overlap_lb`` and is not called reciprocal when
    ``overlap_min > 0``.

    Returns
    -------
    DataFrame indexed by mouse-WB subclass with columns ``n_spc_cells``,
    ``fwd_spc_subclass``, ``overlap``, ``overlap_lb``, ``rev_spc_subclass``,
    ``rev_spc_group``, ``reciprocal`` (bool), sorted by ``n_spc_cells``
    descending.
    """
    df = query_obs[[query_subclass_key]].join(
        mapping_df[[wb_subclass_level]], how="inner")
    df = df.dropna(subset=[query_subclass_key, wb_subclass_level])
    rows = []
    rev_sub = reverse_df[rev_subclass_level]
    rev_grp = reverse_df["Group"] if "Group" in reverse_df.columns else None
    for m, sub in df.groupby(wb_subclass_level, observed=True):
        n = int(len(sub))
        if n < min_cells:
            continue
        vc = sub[query_subclass_key].value_counts()
        fwd_spc = vc.index[0]
        frac = float(vc.iloc[0]) / n
        overlap_lb = wilson_lower_bound(frac, n, z=z)
        rev_spc = rev_sub.get(m, None)
        agree = (rev_spc is not None
                 and normalize_label(fwd_spc) == normalize_label(rev_spc))
        recip = bool(agree and overlap_lb >= overlap_min)
        rows.append({
            wb_subclass_level: m,
            "n_spc_cells": n,
            "fwd_spc_subclass": fwd_spc,
            "overlap": round(frac, 4),
            "overlap_lb": round(overlap_lb, 4),
            "rev_spc_subclass": rev_spc,
            "rev_spc_group": (rev_grp.get(m, None) if rev_grp is not None else None),
            "reciprocal": recip,
        })
    return pd.DataFrame(rows).set_index(wb_subclass_level).sort_values(
        "n_spc_cells", ascending=False)


def _legacy_build_reciprocal_table(
    forward_summary: pd.DataFrame,
    reverse_df: pd.DataFrame,
    query_group_key: str,
    spc_group_level: str = "Group",
) -> pd.DataFrame:
    """Group-level reciprocal join (only matches where the two taxonomies share
    Group names — i.e. the non-neuronal groups). Kept for reference; the notebook
    uses :func:`reciprocal_by_wb_subclass` which anchors on the shared Subclass."""
    fwd = forward_summary.copy()
    rev_target = reverse_df[spc_group_level].map(normalize_label)
    fwd["reverse_spc_group"] = fwd["top_wb_subclass"].map(rev_target)
    fwd["reciprocal"] = (fwd["reverse_spc_group"]
                         == fwd.index.to_series().map(normalize_label))
    fwd.index.name = query_group_key
    cols = ["top_wb_subclass", "reverse_spc_group", "reciprocal"]
    cols += [c for c in ("top_frac", "n_cells") if c in fwd.columns]
    return fwd[cols]


# ---------------------------------------------------------------------------
# OVERLAP-COEFFICIENT reciprocity (subclass & supertype)
# ---------------------------------------------------------------------------
def wilson_lower_bound(p: float, n: int, z: float = 1.96) -> float:
    """Wilson score **lower** confidence bound of a binomial proportion.

    Treats an overlap value ``p`` as a proportion estimated from ``n`` trials
    (here ``n = min(|A|, |B|)``, the denominator of the overlap coefficient) and
    returns the lower end of its ``z``-confidence interval::

        LB = (p + z²/2n − z·√( p(1−p)/n + z²/4n² )) / (1 + z²/n)

    This shrinks small-``n`` estimates toward 0 (an overlap of 1.0 from a handful
    of cells is heavily discounted) while leaving well-supported overlaps almost
    unchanged, so it down-weights mouse-WB subclasses backed by few spinal cells.
    ``z = 1.96`` ≈ 95% confidence; larger ``z`` penalizes low ``n`` more.
    """
    if n <= 0:
        return 0.0
    p = min(max(float(p), 0.0), 1.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    margin = z * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return float(max(0.0, (centre - margin) / denom))


def overlap_coefficient_matrix(
    mapping_df: pd.DataFrame,
    query_obs: pd.DataFrame,
    query_subclass_key: str,
    wb_subclass_level: str = "subclass_label",
) -> pd.DataFrame:
    """Szymkiewicz–Simpson **overlap coefficient** between every spinal Subclass
    (rows) and every detected mouse-WB subclass (cols), from the forward map.

    For spinal subclass ``S`` (set ``A`` = its query cells) and mouse-WB subclass
    ``M`` (set ``B`` = query cells assigned to ``M``)::

        OC(S, M) = |A ∩ B| / min(|A|, |B|)

    OC = 1 means one set is contained in the other (a clean, exclusive
    correspondence); OC ≈ 0 means the two barely co-occur. Unlike a raw fraction
    it is symmetric and not diluted by very different set sizes, which is why it
    is a good "how reciprocally mapped" score.

    Returns a DataFrame ``OC[spinal_subclass, wb_subclass]`` (0–1).
    """
    df = query_obs[[query_subclass_key]].join(
        mapping_df[[wb_subclass_level]], how="inner")
    df = df.dropna(subset=[query_subclass_key, wb_subclass_level])
    ct = pd.crosstab(df[query_subclass_key], df[wb_subclass_level])
    a = ct.sum(axis=1).to_numpy()[:, None]        # |A| per spinal subclass
    b = ct.sum(axis=0).to_numpy()[None, :]        # |B| per mouse-WB subclass
    denom = np.minimum(a, b)
    denom[denom == 0] = 1
    oc = ct.to_numpy() / denom
    return pd.DataFrame(oc, index=ct.index, columns=ct.columns)


def forward_best_partner(
    mapping_df: pd.DataFrame,
    query_obs: pd.DataFrame,
    query_label_key: str,
    wb_subclass_level: str = "subclass_label",
    wilson_z: float = 1.96,
) -> pd.DataFrame:
    """Per mouse-WB subclass M: the spinal label (at ``query_label_key``
    granularity — e.g. ``Subclass_V2`` or ``Group_V2``) with the highest forward
    overlap coefficient, plus the raw overlap, its **Wilson lower bound**, and the
    support ``n = min(|A|, |B|)``.

    Returns a DataFrame indexed by ``wb_subclass_level`` with columns
    ``best_spc``, ``overlap``, ``overlap_lb``, ``n_spc_cells``, ``n_support``.
    """
    oc = overlap_coefficient_matrix(
        mapping_df, query_obs, query_label_key, wb_subclass_level)
    joined = query_obs[[query_label_key]].join(
        mapping_df[[wb_subclass_level]], how="inner").dropna(
        subset=[query_label_key, wb_subclass_level])
    a_sizes = joined[query_label_key].value_counts()   # |A| per spinal label
    b_sizes = joined[wb_subclass_level].value_counts()  # |B| per mouse-WB subclass
    rows = []
    for m in oc.columns:
        col = oc[m]
        best = col.idxmax()
        ov = float(col.max())
        n_support = int(min(a_sizes.get(best, 0), b_sizes.get(m, 0)))
        rows.append({
            wb_subclass_level: m,
            "best_spc": best,
            "overlap": ov,
            "overlap_lb": wilson_lower_bound(ov, n_support, z=wilson_z),
            "n_spc_cells": int(b_sizes.get(m, 0)),
            "n_support": n_support,
        })
    return pd.DataFrame(rows).set_index(wb_subclass_level)


def reciprocal_subclass_overlap(
    mapping_df: pd.DataFrame,
    query_obs: pd.DataFrame,
    reverse_df: pd.DataFrame,
    query_subclass_key: str,
    wb_subclass_level: str = "subclass_label",
    rev_subclass_level: str = "Subclass",
    rev_group_level: str = "Group",
    min_overlap: float = 0.20,
    wilson_z: float = 1.96,
) -> pd.DataFrame:
    """Per mouse-WB subclass M: its best spinal **subclass** partner by overlap
    coefficient, and whether the reverse map agrees (graded reciprocity).

      * ``fwd_spc_subclass`` / ``overlap`` — the spinal subclass with the highest
        OC to M (the forward "best partner") and that raw OC value.
      * ``overlap_lb`` — the **Wilson lower confidence bound** of ``overlap`` given
        ``n = min(|A|, |B|)`` cells of support. This discounts overlaps backed by
        few spinal cells, so a mouse-WB subclass that only a handful of spinal
        nuclei mapped to no longer scores ~1.0.
      * ``rev_spc_subclass`` — the spinal subclass M maps back to (reverse arm).
      * ``reciprocal`` — both arms name the same spinal subclass **and**
        ``overlap_lb >= min_overlap``.

    Returns a DataFrame indexed by mouse-WB subclass, sorted by ``overlap_lb``.
    """
    fwd = forward_best_partner(
        mapping_df, query_obs, query_subclass_key, wb_subclass_level, wilson_z)
    rev_sub = reverse_df[rev_subclass_level]
    rev_grp = reverse_df[rev_group_level] if rev_group_level in reverse_df.columns else None
    rows = []
    for m in fwd.index:
        fwd_spc = fwd.at[m, "best_spc"]
        ov = float(fwd.at[m, "overlap"])
        ov_lb = float(fwd.at[m, "overlap_lb"])
        rev_spc = rev_sub.get(m, None)
        recip = (rev_spc is not None
                 and normalize_label(fwd_spc) == normalize_label(rev_spc)
                 and ov_lb >= min_overlap)
        rows.append({
            wb_subclass_level: m,
            "n_spc_cells": int(fwd.at[m, "n_spc_cells"]),
            "fwd_spc_subclass": fwd_spc,
            "overlap": ov,
            "overlap_lb": ov_lb,
            "rev_spc_subclass": rev_spc,
            "rev_spc_group": (rev_grp.get(m, None) if rev_grp is not None else None),
            "reciprocal": bool(recip),
        })
    return pd.DataFrame(rows).set_index(wb_subclass_level).sort_values(
        "overlap_lb", ascending=False)


def reciprocal_supertypes(
    reverse_supertype_df: pd.DataFrame,
    supertype_to_subclass: pd.DataFrame,
    fwd_group_overlap: pd.DataFrame,
    rev_group_level: str = "Group",
    min_overlap: float = 0.20,
) -> pd.DataFrame:
    """Per mouse-WB **supertype** T: is it reciprocally mapped to a spinal
    **Group** (the finer ``Group_V2`` level)?

    The forward map only resolves mouse-WB to **subclass**, so a supertype T
    inherits its forward signal from its parent subclass M — specifically the
    spinal **Group** that M's cells overlap most (``fwd_group_overlap``, built by
    :func:`forward_best_partner` with the ``Group_V2`` query label). T carries its
    **own** reverse assignment at Group resolution (``reverse_supertype_df``, from
    ``06_map_wb_supertype_to_spc.py`` against the V2 reference). T is reciprocal
    when those agree and the forward overlap (after the Wilson lower-bound support
    discount) clears ``min_overlap``::

        reciprocal(T) = [ rev_spc_group(T) == fwd_spc_group(parent M) ]
                        and [ overlap_lb(parent M) >= min_overlap ]

    Parameters
    ----------
    fwd_group_overlap : DataFrame from :func:`forward_best_partner` at the
        ``Group_V2`` granularity, indexed by mouse-WB subclass (provides
        ``best_spc`` = the forward best spinal Group, ``overlap``, ``overlap_lb``).

    Returns a DataFrame indexed by supertype with the parent subclass, both
    directions (``fwd_spc_group`` / ``rev_spc_group``), ``overlap`` /
    ``overlap_lb`` (inherited from M), and ``reciprocal``.
    """
    parent = supertype_to_subclass["subclass_label"]
    rev_grp = reverse_supertype_df[rev_group_level]
    rows = []
    for t in reverse_supertype_df.index:
        m = parent.get(t, None)
        if m is None or m not in fwd_group_overlap.index:
            continue
        fwd_grp = fwd_group_overlap.at[m, "best_spc"]
        ov = float(fwd_group_overlap.at[m, "overlap"])
        ov_lb = float(fwd_group_overlap.at[m, "overlap_lb"])
        rev_grp_t = rev_grp.get(t, None)
        recip = (rev_grp_t is not None
                 and normalize_label(rev_grp_t) == normalize_label(fwd_grp)
                 and ov_lb >= min_overlap)
        rows.append({
            "supertype": t,
            "parent_wb_subclass": m,
            "fwd_spc_group": fwd_grp,
            "rev_spc_group": rev_grp_t,
            "overlap": ov,
            "overlap_lb": ov_lb,
            "reciprocal": bool(recip),
        })
    return pd.DataFrame(rows).set_index("supertype").sort_values(
        "overlap_lb", ascending=False)
