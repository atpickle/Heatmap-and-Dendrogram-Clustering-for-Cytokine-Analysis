"""Exploratory independent-group tests for raw cytokine heatmap observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy import stats


def analyze_heatmap_groups(
    observations: pd.DataFrame,
    samples: list[str],
    cytokines: list[str],
    control: str,
    alpha: float,
    censored: set[tuple[str, str | None, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Average technical rows, then run Kruskal-Wallis and Dunn vs control.

    Every included group needs at least two numeric biological replicates.
    Rep IDs identify repeats within groups, not pairs between groups. All
    requested comparisons are exported, including explicit not-tested rows.
    """
    selected = observations.loc[
        observations.Sample.isin(samples) & observations.Cytokine.isin(cytokines)
    ].copy()
    numeric = selected.Obs_Conc.dropna()
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite concentrations cannot be used for cytokine statistics.")
    points = selected.groupby(
        ["Cytokine", "Sample", "Rep"], sort=False, as_index=False, dropna=False
    ).agg(
        Value=("Obs_Conc", "mean"),
        Number_Values=("Obs_Conc", "count"),
        Number_Rows=("Obs_Conc", "size"),
    )
    others = [sample for sample in samples if sample != control]
    overall_rows, pair_rows = [], []
    for cytokine in cytokines:
        metric = points.loc[points.Cytokine.eq(cytokine)]
        groups = {
            sample: metric.loc[
                metric.Sample.eq(sample) & metric.Rep.notna(), "Value"
            ].dropna().to_numpy(dtype=float)
            for sample in samples
        }
        counts = {sample: len(values) for sample, values in groups.items()}
        inadequate = [sample for sample, count in counts.items() if count < 2]
        if len(samples) < 2:
            status = "not tested: at least two groups are required"
        elif metric.Rep.isna().any() or metric.Rep.fillna("").astype(str).str.strip().eq("").any():
            status = "not tested: missing biological replicate IDs"
        elif any(sample in samples and name == cytokine for sample, _, name in censored):
            status = "not tested: censored OOR < observations require an assay-specific handling policy"
        elif inadequate:
            status = f"not tested: fewer than two numeric biological replicates in {inadequate}"
        else:
            status = "tested"

        h_value = p_value = np.nan
        comparisons = []
        if status == "tested":
            pooled = np.concatenate(list(groups.values()))
            if np.ptp(pooled) == 0:
                h_value, p_value = 0.0, 1.0
            else:
                result = stats.kruskal(*groups.values())
                h_value, p_value = float(result.statistic), float(result.pvalue)
            ranks = stats.rankdata(pooled)
            boundaries = np.cumsum([0, *counts.values()])
            mean_ranks = {
                sample: ranks[boundaries[index]:boundaries[index + 1]].mean()
                for index, sample in enumerate(samples)
            }
            _, ties = np.unique(pooled, return_counts=True)
            total = len(pooled)
            variance = total * (total + 1) / 12 - np.sum(ties ** 3 - ties) / (12 * (total - 1))
            for sample in others:
                difference = mean_ranks[sample] - mean_ranks[control]
                se = np.sqrt(max(0.0, variance) * (1 / counts[sample] + 1 / counts[control]))
                z_value = difference / se if se > 0 else 0.0
                comparisons.append((sample, z_value, float(2 * stats.norm.sf(abs(z_value)))))
            adjusted = _holm(np.array([p for _, _, p in comparisons]))
        else:
            adjusted = np.full(len(others), np.nan)

        overall_rows.append({
            "Cytokine": cytokine, "Number_Groups": len(samples),
            "N_Total": sum(counts.values()), "DF": len(samples) - 1,
            "Replicates_Per_Group": "; ".join(f"{sample}: n={count}" for sample, count in counts.items()),
            "H_Statistic": h_value, "P_Value": p_value, "Q_BH": np.nan,
            "Status": status,
        })
        tested_pairs = {sample: (z_value, p) for sample, z_value, p in comparisons}
        for index, sample in enumerate(others):
            z_value, raw_p = tested_pairs.get(sample, (np.nan, np.nan))
            pair_rows.append({
                "Cytokine": cytokine, "Group": sample, "Control": control,
                "N_Group": counts[sample], "N_Control": counts[control],
                "Z_Statistic": z_value, "P_Unadjusted": raw_p,
                "P_Holm": adjusted[index],
                "Significant": bool(np.isfinite(adjusted[index]) and adjusted[index] < alpha),
                "Status": status,
            })
        print(f"{cytokine}: {status}; " + "; ".join(f"{s}: n={n}" for s, n in counts.items()))

    overall = pd.DataFrame(overall_rows)
    valid = overall.P_Value.dropna().sort_values()
    if len(valid):
        adjusted_bh = valid.to_numpy() * len(valid) / np.arange(1, len(valid) + 1)
        overall.loc[valid.index, "Q_BH"] = np.minimum.accumulate(adjusted_bh[::-1])[::-1].clip(0, 1)
    overall["FDR_Family_Size"] = len(valid)
    pairs = pd.DataFrame(pair_rows, columns=[
        "Cytokine", "Group", "Control", "N_Group", "N_Control",
        "Z_Statistic", "P_Unadjusted", "P_Holm", "Significant", "Status",
    ])
    return points, overall, pairs


def _holm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    adjusted[order] = np.maximum.accumulate(
        values[order] * np.arange(len(values), 0, -1)
    ).clip(0, 1)
    return adjusted


def export_heatmap_statistics(
    path: Path, observations: pd.DataFrame, points: pd.DataFrame,
    overall: pd.DataFrame, pairs: pd.DataFrame, control: str, alpha: float,
) -> None:
    notes = {
        "Control": control,
        "Alpha": str(alpha),
        "Minimum numeric biological replicates per group": "2; unequal group sizes allowed",
        "Unit of inference": "Technical rows averaged within cytokine/sample/Rep; groups independent, not paired",
        "Missing data": "Omit missing numeric values; do not impute or drop entire treatment groups",
        "Censored data": "Skip cytokine if OOR < values are present; descriptive zero is not used for inference",
        "Omnibus": "Kruskal-Wallis, tie-corrected chi-square approximation; BH FDR across valid selected cytokines",
        "Post-hoc": "Dunn, pooled tie-corrected ranks, two-sided; Holm within cytokine vs control; no omnibus gate",
        "Small sample warning": "Exploratory at n=2-3: asymptotic p-values may be inaccurate and power is limited",
        "Plots": "Raw row-weighted mean +/- SD and observations; tests use biological means, not heatmap z-scores",
    }
    with pd.ExcelWriter(path) as writer:
        overall.to_excel(writer, sheet_name="Kruskal_Wallis", index=False)
        pairs.to_excel(writer, sheet_name="Dunn_vs_Control", index=False)
        points.to_excel(writer, sheet_name="Biological_Replicates", index=False)
        observations.to_excel(writer, sheet_name="Input_Observations", index=False)
        pd.DataFrame(notes.items(), columns=["Setting", "Value"]).to_excel(
            writer, sheet_name="Analysis_Notes", index=False,
        )


def annotate_heatmap_bar(
    ax: Axes, cytokine: str, samples: list[str], overall: pd.DataFrame,
    pairs: pd.DataFrame, control: str,
) -> None:
    """Label skipped tests and Holm-adjusted Dunn significance without clipping."""
    row = overall.loc[overall.Cytokine.eq(cytokine)].iloc[0]
    if row.Status != "tested":
        ax.set_title(f"{cytokine}\n{row.Status}", fontsize=9, wrap=True)
        return
    ax.set_title(
        f"{cytokine}\nKruskal-Wallis p={row.P_Value:.3g}; BH q={row.Q_BH:.3g}\n"
        f"Dunn vs {control} (Holm); exploratory, small n", fontsize=9,
    )
    lower, upper = ax.get_ylim()
    span = upper - lower
    significant = pairs.loc[pairs.Cytokine.eq(cytokine) & pairs.Significant]
    for row in significant.itertuples():
        p_value = row.P_Holm
        stars = "****" if p_value < 0.0001 else "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
        ax.text(samples.index(row.Group), upper + 0.03 * span, stars, ha="center", va="bottom")
    if not significant.empty:
        ax.set_ylim(lower, upper + 0.15 * span)
