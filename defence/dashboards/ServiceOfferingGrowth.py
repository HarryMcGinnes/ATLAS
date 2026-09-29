"""
Build an Accenture-addressable capability growth dashboard from AusTender data.

Purpose
-------
This is a separate dashboard from the main TAM/market-share page. It focuses on
one deliverable: showing whether each Accenture-addressable capability segment is
growing or shrinking over time.

The main visual is an interactive category selector:
- choose a capability/category from the dropdown
- bars show addressable segment value by financial year
- line shows year-on-year growth % for that selected capability
- summary cards show total value, latest FY value, CAGR, peak FY and Accenture share

Scope
-----
Unless you change the code, this dashboard only analyses contracts classified as
addressable to Accenture. That keeps the analysis focused on the market Accenture
can realistically compete for: transformation, IT services, cyber, data, AI,
cloud, advisory, enterprise platforms, systems integration, managed services and
program delivery.

Usage
-----
  python CapabilityGrowth.py --input "master_classification_output/master_defence_contracts.parquet"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio



VALUE_COL = "Value"
ANNUALISED_VALUE_COL = "Value Per Year"
SUPPLIER_COL = "Supplier Name"
DESCRIPTION_COL = "Description"
CATEGORY_COL = "Category"
CATEGORY_TYPE_COL = "Category Type"
FIN_YEAR_COL = "Financial Year"
CN_ID_COL = "CN ID"
AGENCY_COL = "Agency"
AGENCY_DIVISION_COL = "Agency Division"
AGENCY_BRANCH_COL = "Agency Branch"

REQUIRED_CLASSIFICATION_COLS = [
    "capability",
    "is_addressable",
]

OPTIONAL_CLASSIFICATION_COLS = [
    "service_line",
    "confidence",
    "matched_terms",
    "supplier_group",
    "is_accenture",
    "is_white_space",
]



MASTER_REQUIRED_COLUMNS = {
    "capability",
    "ReinventionPartner",
    "ReinventionEngine",
    "is_addressable",
    "supplier_group",
    "is_accenture",
    "defence_domain",
    "Value",
}

SEVEN_DOMAINS = {
    "Maritime",
    "Land",
    "Air",
    "Joint",
    "Cyber",
    "Space",
    "Capability Enabler",
    "Unmapped",
}

STRICT_DEFENCE_AGENCIES = {
    "department of defence",
    "australian signals directorate",
    "australian submarine agency",
}


def _normalise_dashboard_agency(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\u00a0", " ").strip().lower())


def _parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def load_master_dataset(path: Path) -> pd.DataFrame:
    """Load the canonical master Defence parquet without reclassification.

    The dashboard layer is intentionally read-only with respect to capability,
    addressability, supplier grouping and Defence-domain mapping. Those fields
    are created only by build_master.py.
    """
    if not path.exists():
        raise SystemExit(f"Master Defence parquet not found: {path}")
    if path.suffix.lower() not in {".parquet", ".pq"}:
        raise SystemExit(
            "Dashboard input must be the canonical master Defence parquet, not "
            "a raw parquet or classified CSV. Expected master_defence_contracts.parquet."
        )

    df = pd.read_parquet(path)
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        raise SystemExit(
            "Master parquet contains duplicate column names: "
            + ", ".join(sorted(set(duplicate_columns)))
        )

    missing = sorted(MASTER_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise SystemExit(
            "Input is not a valid canonical master Defence dataset. Missing columns: "
            + ", ".join(missing)
            + ". Run build_master.py first."
        )

    if AGENCY_COL not in df.columns:
        raise SystemExit(
            f"Canonical master dataset is missing the mandatory procurer field: {AGENCY_COL}"
        )
    agency_norm = df[AGENCY_COL].map(_normalise_dashboard_agency)
    invalid_agencies = sorted(
        df.loc[~agency_norm.isin(STRICT_DEFENCE_AGENCIES), AGENCY_COL]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if invalid_agencies:
        raise SystemExit(
            "Canonical master dataset contains non-Defence procurers and must be rebuilt. "
            "Allowed agencies are Department of Defence, Australian Signals Directorate, "
            "and Australian Submarine Agency. Unexpected agencies include: "
            + "; ".join(invalid_agencies[:20])
        )

    out = df.copy()
    out["is_addressable"] = _parse_bool_series(out["is_addressable"])
    out["is_accenture"] = _parse_bool_series(out["is_accenture"])
    if "is_market_relevant" in out.columns:
        out["is_market_relevant"] = _parse_bool_series(out["is_market_relevant"])
    if "is_defence_scope" in out.columns:
        out["is_defence_scope"] = _parse_bool_series(out["is_defence_scope"])
        out = out[out["is_defence_scope"]].copy()

    out["capability"] = out["capability"].fillna("Unclassified").astype(str)
    out["ReinventionPartner"] = out["ReinventionPartner"].fillna("Unclassified").replace("", "Unclassified").astype(str)
    out["ReinventionEngine"] = out["ReinventionEngine"].fillna("Unclassified").replace("", "Unclassified").astype(str)
    out["supplier_group"] = out["supplier_group"].fillna("Unknown").replace("", "Unknown").astype(str)
    out["defence_domain"] = out["defence_domain"].fillna("Unmapped").replace("", "Unmapped").astype(str)

    invalid_domains = sorted(set(out["defence_domain"]) - SEVEN_DOMAINS)
    if invalid_domains:
        raise SystemExit(
            "Master parquet contains unsupported Defence domains: "
            + ", ".join(invalid_domains)
        )

    out["is_competitor_market"] = out["is_addressable"] & ~out["is_accenture"]
    out["is_white_space"] = out["is_competitor_market"]

    print(f"Using canonical master Defence dataset without reclassification: {path}")
    print(f"Master Defence rows loaded: {len(out):,}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="master_output/master_defence_contracts.parquet")
    parser.add_argument("--output-dir", default="ServiceGrowthDashboard_Output")
    parser.add_argument(
        "--value-mode",
        choices=["total", "annualised"],
        default="total",
        help="total = full contract value. annualised = Value Per Year / run-rate.",
    )
    parser.add_argument("--top-n-capabilities", type=int, default=18)
    parser.add_argument(
        "--include-all-agencies",
        action="store_true",
        help="Compatibility option only. The canonical master input is already Defence-only.",
    )
    parser.add_argument(
        "--min-year-value",
        type=float,
        default=0,
        help="Optional minimum annual capability value required to include in the chart data.",
    )
    return parser.parse_args()


def short_money(value: float) -> str:
    """Human friendly money formatting for labels."""
    if pd.isna(value):
        return "$0"
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000_000:
        return f"{sign}${value / 1_000_000_000:,.2f}B"
    if value >= 100_000_000:
        return f"{sign}${value / 1_000_000:,.0f}M"
    if value >= 10_000_000:
        return f"{sign}${value / 1_000_000:,.1f}M"
    if value >= 1_000_000:
        return f"{sign}${value / 1_000_000:,.2f}M"
    if value >= 1_000:
        return f"{sign}${value / 1_000:,.0f}K"
    return f"{sign}${value:,.0f}"


def pct(value: float) -> str:
    if pd.isna(value):
        return "0.0%"
    return f"{float(value) * 100:,.1f}%"


def pct_number(value: float) -> str:
    if pd.isna(value):
        return "0.0%"
    return f"{float(value):,.1f}%"


def supplier_group(value: object) -> str:
    """Deprecated: dashboards must use canonical supplier_group from the master."""
    raise RuntimeError(
        "ServiceGrowth must not normalise supplier names. Use the canonical "
        "supplier_group column written by build_master.py."
    )


DEFENCE_AGENCY_TERMS = [
    "department of defence",
    "australian signals directorate",
    "australian geospatial",
    "defence intelligence",
    "defence housing australia",
]

DEFENCE_ORG_CODES = {
    "ADHQ", "AIRF", "ARMY", "ASA", "ASD", "ASSEC", "CASG", "CDG", "CFO", "CIOG",
    "DDG", "DFG", "DIG", "DIO", "DPG", "DSTG", "GWEO", "JCG", "JOC", "NAVY",
    "NPS", "NSSG", "SEG", "SP&I", "SPI", "VCDF",
}

DEFENCE_TEXT_TERMS = [
    "air force", "army", "navy", "naval", "maritime", "casg", "ciog", "ddg",
    "joint capabilities", "aerospace systems", "land systems", "maritime systems",
    "defence digital", "defence intelligence", "defence science", "guided weapons",
    "capability acquisition", "security and estate", "defence people", "defence finance",
]


def _norm_code(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper().replace(" ", "")


def _contains_defence_text(value: object) -> bool:
    text = str(value or "").lower()
    return any(term in text for term in DEFENCE_TEXT_TERMS)


def filter_defence_scope(df: pd.DataFrame, include_all_agencies: bool = False) -> pd.DataFrame:
    """Return the canonical Defence dataset unchanged.

    Defence scope is established only in build_master.py by
    exact Agency allow-list before domain or capability classification.
    """
    if include_all_agencies:
        print("Warning: --include-all-agencies is ignored for canonical master inputs.")
    return df.copy()


def export_defence_scope_audit(original_df: pd.DataFrame, filtered_df: pd.DataFrame, value_col: str, output_dir: Path) -> None:
    """Write audit files showing which agencies/divisions were included/excluded."""
    original = original_df.copy()
    original["_scope"] = "Excluded"
    original.loc[filtered_df.index.intersection(original.index), "_scope"] = "Included Defence scope"

    if AGENCY_COL in original.columns:
        (
            original.groupby(["_scope", AGENCY_COL], dropna=False)
            .agg(value=(value_col, "sum"), contracts=(value_col, "size"))
            .reset_index()
            .sort_values(["_scope", "value"], ascending=[True, False])
            .to_csv(output_dir / "defence_scope_audit_by_agency.csv", index=False)
        )

    group_cols = [c for c in [AGENCY_COL, AGENCY_DIVISION_COL, AGENCY_BRANCH_COL, "branch_clean", "div_clean"] if c in original.columns]
    if group_cols:
        (
            original.groupby(["_scope"] + group_cols, dropna=False)
            .agg(value=(value_col, "sum"), contracts=(value_col, "size"))
            .reset_index()
            .sort_values(["_scope", "value"], ascending=[True, False])
            .head(1000)
            .to_csv(output_dir / "defence_scope_audit_by_org.csv", index=False)
        )


def load_raw_data(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise SystemExit(
            "Unable to read Parquet because no Parquet engine is installed. "
            "Install pyarrow or fastparquet, then rerun: pip install pyarrow"
        ) from exc
    except FileNotFoundError as exc:
        raise SystemExit(f"Input file not found: {path}") from exc


def load_or_classify(args: argparse.Namespace) -> pd.DataFrame:
    return load_master_dataset(Path(args.input))


def get_value_col(df: pd.DataFrame, value_mode: str) -> str:
    if value_mode == "annualised":
        if ANNUALISED_VALUE_COL not in df.columns:
            raise SystemExit(f"--value-mode annualised requested but column is missing: {ANNUALISED_VALUE_COL}")
        return ANNUALISED_VALUE_COL
    return VALUE_COL


def clean_value_columns(df: pd.DataFrame, value_cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in value_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def financial_year_start_year(value: object) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    import re
    match = re.search(r"(20\d{2}|19\d{2})", text)
    if not match:
        return None
    return int(match.group(1))


def prepare_growth_data(df: pd.DataFrame, value_col: str, top_n: int, min_year_value: float = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    if FIN_YEAR_COL not in df.columns:
        raise SystemExit(f"Missing required column: {FIN_YEAR_COL}")

    data = df[df["is_addressable"]].copy()
    data["capability"] = data["capability"].fillna("Unclassified").replace("", "Unclassified")
    data["_fy_start_year"] = data[FIN_YEAR_COL].map(financial_year_start_year)
    data = data.dropna(subset=["_fy_start_year"])
    data["_fy_start_year"] = data["_fy_start_year"].astype(int)
    data[FIN_YEAR_COL] = data[FIN_YEAR_COL].astype(str)

    top_caps = (
        data.groupby("capability")[value_col]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    data = data[data["capability"].isin(top_caps)].copy()

    annual = (
        data.groupby(["capability", FIN_YEAR_COL, "_fy_start_year"], as_index=False)
        .agg(
            segment_value=(value_col, "sum"),
            accenture_wins=(value_col, lambda s: s[data.loc[s.index, "is_accenture"]].sum()),
            contracts=(CN_ID_COL, "nunique") if CN_ID_COL in data.columns else (value_col, "size"),
        )
        .sort_values(["capability", "_fy_start_year"])
    )
    annual = annual[annual["segment_value"] >= min_year_value].copy()

    # Re-index every selected capability against every selected FY so zero years are visible.
    years = (
        data[[FIN_YEAR_COL, "_fy_start_year"]]
        .drop_duplicates()
        .sort_values("_fy_start_year")
    )
    full_index = pd.MultiIndex.from_product(
        [top_caps, years[FIN_YEAR_COL].tolist()],
        names=["capability", FIN_YEAR_COL],
    ).to_frame(index=False)
    full_index = full_index.merge(years, on=FIN_YEAR_COL, how="left")
    annual = full_index.merge(annual, on=["capability", FIN_YEAR_COL, "_fy_start_year"], how="left")
    annual[["segment_value", "accenture_wins", "contracts"]] = annual[["segment_value", "accenture_wins", "contracts"]].fillna(0)
    annual = annual.sort_values(["capability", "_fy_start_year"])

    annual["segment_value_b"] = annual["segment_value"] / 1_000_000_000
    annual["accenture_wins_b"] = annual["accenture_wins"] / 1_000_000_000
    annual["accenture_share_pct"] = annual["accenture_wins"] / annual["segment_value"].replace(0, pd.NA) * 100
    annual["yoy_growth_pct"] = annual.groupby("capability")["segment_value"].pct_change() * 100
    annual.loc[annual.groupby("capability")["segment_value"].shift(1).fillna(0).eq(0), "yoy_growth_pct"] = pd.NA

    summary_rows = []
    for cap, g in annual.groupby("capability", sort=False):
        g = g.sort_values("_fy_start_year")
        nonzero = g[g["segment_value"] > 0]
        first_value = float(nonzero["segment_value"].iloc[0]) if not nonzero.empty else 0
        last_value = float(g["segment_value"].iloc[-1]) if not g.empty else 0
        first_year = str(nonzero[FIN_YEAR_COL].iloc[0]) if not nonzero.empty else "n/a"
        latest_year = str(g[FIN_YEAR_COL].iloc[-1]) if not g.empty else "n/a"
        periods = max(0, int(nonzero["_fy_start_year"].iloc[-1] - nonzero["_fy_start_year"].iloc[0])) if len(nonzero) >= 2 else 0
        cagr = ((last_value / first_value) ** (1 / periods) - 1) * 100 if first_value > 0 and last_value > 0 and periods > 0 else pd.NA
        peak_idx = g["segment_value"].idxmax() if not g.empty else None
        peak_year = str(g.loc[peak_idx, FIN_YEAR_COL]) if peak_idx is not None else "n/a"
        peak_value = float(g.loc[peak_idx, "segment_value"]) if peak_idx is not None else 0
        total_value = float(g["segment_value"].sum())
        accenture_total = float(g["accenture_wins"].sum())
        share = accenture_total / total_value * 100 if total_value else 0

        # Accenture share CAGR is calculated from Accenture's annual share of
        # the selected capability market, not from Accenture dollar value.
        share_nonzero = g[g["segment_value"].gt(0)].copy()
        share_nonzero["_share_for_cagr"] = (
            share_nonzero["accenture_wins"] / share_nonzero["segment_value"].replace(0, pd.NA) * 100
        )
        share_nonzero = share_nonzero.dropna(subset=["_share_for_cagr"])
        share_nonzero = share_nonzero[share_nonzero["_share_for_cagr"].gt(0)]
        if len(share_nonzero) >= 2:
            first_share = float(share_nonzero["_share_for_cagr"].iloc[0])
            last_share = float(share_nonzero["_share_for_cagr"].iloc[-1])
            share_periods = max(0, int(share_nonzero["_fy_start_year"].iloc[-1] - share_nonzero["_fy_start_year"].iloc[0]))
            accenture_share_cagr = ((last_share / first_share) ** (1 / share_periods) - 1) * 100 if first_share > 0 and last_share > 0 and share_periods > 0 else pd.NA
        else:
            accenture_share_cagr = pd.NA

        summary_rows.append({
            "capability": cap,
            "total_value": total_value,
            "latest_year": latest_year,
            "latest_value": last_value,
            "first_year": first_year,
            "first_value": first_value,
            "cagr_pct": cagr,
            "peak_year": peak_year,
            "peak_value": peak_value,
            "accenture_wins": accenture_total,
            "accenture_share_pct": share,
            "accenture_share_cagr_pct": accenture_share_cagr,
        })

    summary = pd.DataFrame(summary_rows).sort_values("total_value", ascending=False)
    return annual, summary


def prepare_reinvention_growth_data(df: pd.DataFrame, value_col: str, min_year_value: float = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare annual growth data for every Reinvention Partner x Reinvention Engine combination."""
    if FIN_YEAR_COL not in df.columns:
        raise SystemExit(f"Missing required column: {FIN_YEAR_COL}")

    required = {"ReinventionPartner", "ReinventionEngine", "is_addressable", "is_accenture"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(
            "Master parquet is missing Reinvention Model columns: " + ", ".join(missing)
            + ". Rebuild the master with the Reinvention Model classifier first."
        )

    data = df[df["is_addressable"]].copy()
    data["ReinventionPartner"] = data["ReinventionPartner"].fillna("Unclassified").replace("", "Unclassified")
    data["ReinventionEngine"] = data["ReinventionEngine"].fillna("Unclassified").replace("", "Unclassified")
    data = data[~data["ReinventionPartner"].eq("Non-addressable")].copy()
    data = data[~data["ReinventionEngine"].eq("Non-addressable")].copy()
    data["reinvention_combination"] = data["ReinventionPartner"] + " | " + data["ReinventionEngine"]
    data["_fy_start_year"] = data[FIN_YEAR_COL].map(financial_year_start_year)
    data = data.dropna(subset=["_fy_start_year"])
    data["_fy_start_year"] = data["_fy_start_year"].astype(int)
    data[FIN_YEAR_COL] = data[FIN_YEAR_COL].astype(str)

    combinations = (
        data.groupby(["ReinventionPartner", "ReinventionEngine"], as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)
    )
    combination_order = (
        combinations["ReinventionPartner"] + " | " + combinations["ReinventionEngine"]
    ).tolist()

    annual = (
        data.groupby(["ReinventionPartner", "ReinventionEngine", "reinvention_combination", FIN_YEAR_COL, "_fy_start_year"], as_index=False)
        .agg(
            segment_value=(value_col, "sum"),
            accenture_wins=(value_col, lambda s: s[data.loc[s.index, "is_accenture"]].sum()),
            contracts=(CN_ID_COL, "nunique") if CN_ID_COL in data.columns else (value_col, "size"),
        )
        .sort_values(["reinvention_combination", "_fy_start_year"])
    )
    annual = annual[annual["segment_value"] >= min_year_value].copy()

    years = data[[FIN_YEAR_COL, "_fy_start_year"]].drop_duplicates().sort_values("_fy_start_year")
    full_index = pd.MultiIndex.from_product(
        [combination_order, years[FIN_YEAR_COL].tolist()],
        names=["reinvention_combination", FIN_YEAR_COL],
    ).to_frame(index=False)
    parts = full_index["reinvention_combination"].str.split(" | ", n=1, expand=True, regex=False)
    full_index["ReinventionPartner"] = parts[0]
    full_index["ReinventionEngine"] = parts[1]
    full_index = full_index.merge(years, on=FIN_YEAR_COL, how="left")
    annual = full_index.merge(
        annual,
        on=["ReinventionPartner", "ReinventionEngine", "reinvention_combination", FIN_YEAR_COL, "_fy_start_year"],
        how="left",
    )
    annual[["segment_value", "accenture_wins", "contracts"]] = annual[["segment_value", "accenture_wins", "contracts"]].fillna(0)
    annual = annual.sort_values(["reinvention_combination", "_fy_start_year"])
    annual["segment_value_b"] = annual["segment_value"] / 1_000_000_000
    annual["accenture_wins_b"] = annual["accenture_wins"] / 1_000_000_000
    annual["accenture_share_pct"] = annual["accenture_wins"] / annual["segment_value"].replace(0, pd.NA) * 100
    annual["yoy_growth_pct"] = annual.groupby("reinvention_combination")["segment_value"].pct_change() * 100
    annual.loc[annual.groupby("reinvention_combination")["segment_value"].shift(1).fillna(0).eq(0), "yoy_growth_pct"] = pd.NA

    summary_rows = []
    for combo, g in annual.groupby("reinvention_combination", sort=False):
        g = g.sort_values("_fy_start_year")
        nonzero = g[g["segment_value"] > 0]
        first_value = float(nonzero["segment_value"].iloc[0]) if not nonzero.empty else 0
        last_value = float(g["segment_value"].iloc[-1]) if not g.empty else 0
        first_year = str(nonzero[FIN_YEAR_COL].iloc[0]) if not nonzero.empty else "n/a"
        latest_year = str(g[FIN_YEAR_COL].iloc[-1]) if not g.empty else "n/a"
        periods = max(0, int(nonzero["_fy_start_year"].iloc[-1] - nonzero["_fy_start_year"].iloc[0])) if len(nonzero) >= 2 else 0
        cagr = ((last_value / first_value) ** (1 / periods) - 1) * 100 if first_value > 0 and last_value > 0 and periods > 0 else pd.NA
        peak_idx = g["segment_value"].idxmax() if not g.empty else None
        peak_year = str(g.loc[peak_idx, FIN_YEAR_COL]) if peak_idx is not None else "n/a"
        peak_value = float(g.loc[peak_idx, "segment_value"]) if peak_idx is not None else 0
        total_value = float(g["segment_value"].sum())
        accenture_total = float(g["accenture_wins"].sum())
        share = accenture_total / total_value * 100 if total_value else 0

        share_nonzero = g[g["segment_value"].gt(0)].copy()
        share_nonzero["_share_for_cagr"] = share_nonzero["accenture_wins"] / share_nonzero["segment_value"].replace(0, pd.NA) * 100
        share_nonzero = share_nonzero.dropna(subset=["_share_for_cagr"])
        share_nonzero = share_nonzero[share_nonzero["_share_for_cagr"].gt(0)]
        if len(share_nonzero) >= 2:
            first_share = float(share_nonzero["_share_for_cagr"].iloc[0])
            last_share = float(share_nonzero["_share_for_cagr"].iloc[-1])
            share_periods = max(0, int(share_nonzero["_fy_start_year"].iloc[-1] - share_nonzero["_fy_start_year"].iloc[0]))
            accenture_share_cagr = ((last_share / first_share) ** (1 / share_periods) - 1) * 100 if first_share > 0 and last_share > 0 and share_periods > 0 else pd.NA
        else:
            accenture_share_cagr = pd.NA

        summary_rows.append({
            "reinvention_combination": combo,
            "ReinventionPartner": str(g["ReinventionPartner"].iloc[0]),
            "ReinventionEngine": str(g["ReinventionEngine"].iloc[0]),
            "total_value": total_value,
            "latest_year": latest_year,
            "latest_value": last_value,
            "first_year": first_year,
            "first_value": first_value,
            "cagr_pct": cagr,
            "peak_year": peak_year,
            "peak_value": peak_value,
            "accenture_wins": accenture_total,
            "accenture_share_pct": share,
            "accenture_share_cagr_pct": accenture_share_cagr,
        })

    summary = pd.DataFrame(summary_rows).sort_values("total_value", ascending=False)
    return annual, summary


def build_reinvention_growth_chart(annual: pd.DataFrame, summary: pd.DataFrame, value_mode: str) -> str:
    """Build a second explorer using linked Reinvention Partner and Engine selectors."""
    partners = summary["ReinventionPartner"].drop_duplicates().tolist()
    datasets = []
    for _, s in summary.iterrows():
        combo = s["reinvention_combination"]
        g = annual[annual["reinvention_combination"].eq(combo)].sort_values("_fy_start_year")
        datasets.append({
            "combination": combo,
            "partner": s["ReinventionPartner"],
            "engine": s["ReinventionEngine"],
            "financial_years": g[FIN_YEAR_COL].astype(str).tolist(),
            "values_b": g["segment_value_b"].fillna(0).round(6).tolist(),
            "competitor_values_b": ((g["segment_value"] - g["accenture_wins"]).clip(lower=0) / 1_000_000_000).fillna(0).round(6).tolist(),
            "accenture_wins_b": g["accenture_wins_b"].fillna(0).round(6).tolist(),
            "yoy": [None if pd.isna(x) else round(float(x), 2) for x in g["yoy_growth_pct"].tolist()],
            "contracts": [int(x) for x in g["contracts"].fillna(0).tolist()],
            "accenture_share": [None if pd.isna(x) else round(float(x), 2) for x in g["accenture_share_pct"].tolist()],
        })

    partner_options = "\n".join(f'<option value="{p}">{p}</option>' for p in partners)
    chart_data = json.dumps(datasets)

    return f"""
<div class="growth-app reinvention-growth-app">
  <div class="growth-header">
    <div>
      <h2>Reinvention Model Growth Explorer</h2>
      <p>Select a Reinvention Partner and Reinvention Engine to see annual addressable value, Accenture wins and growth across that combination.</p>
    </div>
    <div class="reinvention-selectors">
      <div class="selector-box">
        <label for="rpSelect">Reinvention Partner</label>
        <select id="rpSelect">{partner_options}</select>
      </div>
      <div class="selector-box">
        <label for="reSelect">Reinvention Engine</label>
        <select id="reSelect"></select>
      </div>
    </div>
  </div>

  <div class="mini-cards">
    <div class="mini-card"><span>Total selected-period value</span><b id="rpCardTotal">-</b></div>
    <div class="mini-card"><span>Latest FY value</span><b id="rpCardLatest">-</b></div>
    <div class="mini-card"><span>Combination CAGR</span><b id="rpCardCagr">-</b></div>
    <div class="mini-card accenture-mini-card"><span>Accenture share CAGR</span><b id="rpCardShareCagr">-</b></div>
    <div class="mini-card"><span>Peak FY</span><b id="rpCardPeak">-</b></div>
    <div class="mini-card"><span>Accenture share</span><b id="rpCardShare">-</b></div>
  </div>

  <div class="range-controls growth-range-controls">
    <label class="range-pill"><input type="radio" name="reinventionRange" value="all" checked> All years</label>
    <label class="range-pill"><input type="radio" name="reinventionRange" value="last5"> Last 5 FY</label>
    <label class="range-pill"><input type="radio" name="reinventionRange" value="last3"> Last 3 FY</label>
  </div>

  <div id="reinventionGrowthChart" style="height:670px; width:100%;"></div>
</div>

<script>
(function() {{
  const datasets = {chart_data};
  const rpSelect = document.getElementById('rpSelect');
  const reSelect = document.getElementById('reSelect');
  const chartId = 'reinventionGrowthChart';

  function rangeMode() {{
    const checked = document.querySelector('input[name="reinventionRange"]:checked');
    return checked ? checked.value : 'all';
  }}
  function parseFyStart(fy) {{ const m = String(fy || '').match(/(20[0-9]{{2}}|19[0-9]{{2}})/); return m ? Number(m[1]) : null; }}
  function moneyFromB(v) {{
    const dollars = Number(v || 0) * 1000000000, abs = Math.abs(dollars), sign = dollars < 0 ? '-' : '';
    if (abs >= 1000000000) return sign + '$' + (abs / 1000000000).toFixed(2) + 'B';
    if (abs >= 100000000) return sign + '$' + (abs / 1000000).toFixed(0) + 'M';
    if (abs >= 10000000) return sign + '$' + (abs / 1000000).toFixed(1) + 'M';
    if (abs >= 1000000) return sign + '$' + (abs / 1000000).toFixed(2) + 'M';
    if (abs >= 1000) return sign + '$' + (abs / 1000).toFixed(0) + 'K';
    return sign + '$' + abs.toFixed(0);
  }}
  function pctText(v) {{ return isFinite(Number(v)) ? Number(v).toFixed(1) + '%' : 'n/a'; }}
  function cagr(rows, key) {{
    const valid = rows.filter(r => Number(r[key] || 0) > 0 && r.fyStart !== null);
    if (valid.length < 2) return null;
    const first = valid[0], last = valid[valid.length - 1];
    const periods = Math.max(0, last.fyStart - first.fyStart);
    if (!periods) return null;
    return (Math.pow(Number(last[key]) / Number(first[key]), 1 / periods) - 1) * 100;
  }}
  function updateEngineOptions() {{
    const partner = rpSelect.value;
    const engines = [...new Set(datasets.filter(d => d.partner === partner).map(d => d.engine))];
    reSelect.innerHTML = engines.map(e => '<option value="' + e + '">' + e + '</option>').join('');
  }}
  function currentDataset() {{ return datasets.find(d => d.partner === rpSelect.value && d.engine === reSelect.value) || datasets[0]; }}
  function filtered(d) {{
    const mode = rangeMode();
    const n = mode === 'last3' ? 3 : (mode === 'last5' ? 5 : d.financial_years.length);
    const start = Math.max(0, d.financial_years.length - n);
    const rows = d.financial_years.map((fy, i) => ({{
      fy, fyStart: parseFyStart(fy), totalB: Number(d.values_b[i] || 0), accB: Number(d.accenture_wins_b[i] || 0),
      compB: Math.max(0, Number(d.competitor_values_b[i] || 0)), contracts: Number(d.contracts[i] || 0)
    }})).slice(start);
    rows.forEach(r => r.share = r.totalB > 0 ? r.accB / r.totalB * 100 : null);
    const totalB = rows.reduce((a,r)=>a+r.totalB,0), accB = rows.reduce((a,r)=>a+r.accB,0);
    return {{rows, totalB, accB, latest: rows[rows.length-1] || null,
      peak: rows.reduce((b,r)=>!b||r.totalB>b.totalB?r:b,null),
      growthCagr: cagr(rows,'totalB'),
      shareCagr: cagr(rows.map(r=>({{fyStart:r.fyStart, shareValue:r.share||0}})),'shareValue'),
      shareTotal: totalB > 0 ? accB/totalB*100 : 0,
      label: mode === 'last3' ? 'Last 3 FY' : (mode === 'last5' ? 'Last 5 FY' : 'All years')
    }};
  }}
  function render() {{
    const d = currentDataset(); if (!d) return;
    const fd = filtered(d);
    document.getElementById('rpCardTotal').textContent = moneyFromB(fd.totalB);
    document.getElementById('rpCardLatest').textContent = fd.latest ? fd.latest.fy + ': ' + moneyFromB(fd.latest.totalB) : 'n/a';
    document.getElementById('rpCardCagr').textContent = fd.growthCagr === null ? 'n/a' : pctText(fd.growthCagr);
    document.getElementById('rpCardShareCagr').textContent = fd.shareCagr === null ? 'n/a' : pctText(fd.shareCagr);
    document.getElementById('rpCardPeak').textContent = fd.peak ? fd.peak.fy + ': ' + moneyFromB(fd.peak.totalB) : 'n/a';
    document.getElementById('rpCardShare').textContent = pctText(fd.shareTotal);
    const maxBar = Math.max(0.1, ...fd.rows.map(r => r.totalB));
    const competitor = {{type:'bar', name:'Competitor-owned segment value', x:fd.rows.map(r=>r.fy), y:fd.rows.map(r=>r.compB), marker:{{color:'#D7DEE9'}}, customdata:fd.rows.map(r=>[moneyFromB(r.totalB),moneyFromB(r.compB),moneyFromB(r.accB),r.contracts,r.share]), hovertemplate:'<b>%{{x}}</b><br>Total value: %{{customdata[0]}}<br>Competitor-owned value: %{{customdata[1]}}<br>Accenture wins: %{{customdata[2]}}<br>Contracts: %{{customdata[3]}}<br>Accenture share: %{{customdata[4]:,.1f}}%<extra></extra>'}};
    const acc = {{type:'bar', name:'Accenture wins', x:fd.rows.map(r=>r.fy), y:fd.rows.map(r=>r.accB), marker:{{color:'#A100FF'}}, customdata:fd.rows.map(r=>[moneyFromB(r.accB),r.share,moneyFromB(r.totalB)]), hovertemplate:'<b>%{{x}}</b><br>Accenture wins: %{{customdata[0]}}<br>Accenture share: %{{customdata[1]:,.1f}}%<br>Total value: %{{customdata[2]}}<extra></extra>'}};
    const annotations = [{{text:'Grey = competitor-owned segment value. Purple = Accenture wins. Only Accenture-addressable contracts are included.',x:0,y:1.15,xref:'paper',yref:'paper',showarrow:false,xanchor:'left',font:{{size:12,color:'#526070'}}}}];
    fd.rows.forEach(r => {{
      const pad=maxBar*0.055, y=r.compB>pad*1.8?r.compB-pad:Math.max(r.compB*0.5,maxBar*0.018);
      annotations.push({{x:r.fy,y,xref:'x',yref:'y',text:moneyFromB(r.totalB),showarrow:false,font:{{size:11,color:'#1f2937'}},bgcolor:'rgba(0,0,0,0)',bordercolor:'rgba(0,0,0,0)',borderpad:0}});
      if (r.accB>0.0005 && r.share!==null) annotations.push({{x:r.fy,y:r.totalB+maxBar*0.06,xref:'x',yref:'y',text:'<b>'+pctText(r.share)+'</b>',showarrow:false,font:{{size:12,color:'#A100FF'}},bgcolor:'rgba(255,255,255,0.90)',borderpad:3}});
    }});
    const layout={{title:d.partner+' + '+d.engine+' - annual addressable value<br><sup>'+fd.label+'</sup>',template:'plotly_white',height:670,margin:{{t:95,l:70,r:90,b:100}},legend:{{orientation:'h',y:-0.18}},barmode:'stack',bargap:fd.rows.length<=3?0.48:0.34,xaxis:{{title:'Financial year',tickangle:-35}},yaxis:{{title:'$B',range:[0,maxBar*1.46],showgrid:true}},annotations}};
    Plotly.react(chartId,[competitor,acc],layout,{{responsive:true,displayModeBar:false,staticPlot:false,scrollZoom:false,doubleClick:false}});
  }}
  rpSelect.addEventListener('change',()=>{{updateEngineOptions();render();}});
  reSelect.addEventListener('change',render);
  document.querySelectorAll('input[name="reinventionRange"]').forEach(i=>i.addEventListener('change',render));
  updateEngineOptions(); render();
}})();
</script>
"""


def build_interactive_growth_chart(annual: pd.DataFrame, summary: pd.DataFrame, value_mode: str) -> str:
    capabilities = summary["capability"].tolist()
    datasets = []
    for cap in capabilities:
        g = annual[annual["capability"].eq(cap)].sort_values("_fy_start_year")
        s = summary[summary["capability"].eq(cap)].iloc[0]
        datasets.append({
            "capability": cap,
            "financial_years": g[FIN_YEAR_COL].astype(str).tolist(),
            "values_b": g["segment_value_b"].fillna(0).round(6).tolist(),
            "competitor_values_b": ((g["segment_value"] - g["accenture_wins"]).clip(lower=0) / 1_000_000_000).fillna(0).round(6).tolist(),
            "accenture_wins_b": g["accenture_wins_b"].fillna(0).round(6).tolist(),
            "values_label": g["segment_value"].map(short_money).tolist(),
            "accenture_wins_label": g["accenture_wins"].map(short_money).tolist(),
            "competitor_values_label": (g["segment_value"] - g["accenture_wins"]).clip(lower=0).map(short_money).tolist(),
            "yoy": [None if pd.isna(x) else round(float(x), 2) for x in g["yoy_growth_pct"].tolist()],
            "contracts": [int(x) for x in g["contracts"].fillna(0).tolist()],
            "accenture_share": [None if pd.isna(x) else round(float(x), 2) for x in g["accenture_share_pct"].tolist()],
            "summary": {
                "total_value": short_money(s["total_value"]),
                "latest_year": str(s["latest_year"]),
                "latest_value": short_money(s["latest_value"]),
                "cagr": "n/a" if pd.isna(s["cagr_pct"]) else f"{float(s['cagr_pct']):,.1f}%",
                "peak_year": str(s["peak_year"]),
                "peak_value": short_money(s["peak_value"]),
                "accenture_share": f"{float(s['accenture_share_pct']):,.1f}%",
                "accenture_share_cagr": "n/a" if pd.isna(s.get("accenture_share_cagr_pct", pd.NA)) else f"{float(s['accenture_share_cagr_pct']):,.1f}%",
            },
        })

    options_html = "\n".join(
        f'<option value="{i}">{cap}</option>' for i, cap in enumerate(capabilities)
    )
    chart_data = json.dumps(datasets)
    basis_label = "Total contract value" if value_mode == "total" else "Annualised contract value / run-rate"

    return f"""
<div class="growth-app">
  <div class="growth-header">
    <div>
      <h2>Service Offering Growth Explorer</h2>
      <p>
        Select an Accenture-addressable Service Offering to see its annual segment value and YoY movement across the financial-year history.
        Stacked bars show segment value and Accenture wins.</b>.
      </p>
    </div>
    <div class="selector-box">
      <label for="capabilitySelect">Service Offering</label>
      <select id="capabilitySelect">{options_html}</select>
    </div>
  </div>

  <div class="mini-cards">
    <div class="mini-card"><span>Total selected-period value</span><b id="cardTotal">-</b></div>
    <div class="mini-card"><span>Latest FY value</span><b id="cardLatest">-</b></div>
    <div class="mini-card"><span>Service Offering CAGR</span><b id="cardCagr">-</b></div>
    <div class="mini-card accenture-mini-card"><span>Accenture share CAGR</span><b id="cardShareCagr">-</b></div>
    <div class="mini-card"><span>Peak FY</span><b id="cardPeak">-</b></div>
    <div class="mini-card"><span>Accenture share</span><b id="cardShare">-</b></div>
  </div>

  <div class="range-controls growth-range-controls">
    <label class="range-pill"><input type="radio" name="growthRange" value="all" checked> All years</label>
    <label class="range-pill"><input type="radio" name="growthRange" value="last5"> Last 5 FY</label>
    <label class="range-pill"><input type="radio" name="growthRange" value="last3"> Last 3 FY</label>
  </div>

  <div id="growthChart" style="height:670px; width:100%;"></div>
</div>

<script>
(function() {{
  const datasets = {chart_data};
  const select = document.getElementById('capabilitySelect');
  const chartId = 'growthChart';

  function rangeMode() {{
    const checked = document.querySelector('input[name="growthRange"]:checked');
    return checked ? checked.value : 'all';
  }}

  function parseFyStart(fy) {{
    const m = String(fy || '').match(/(20[0-9]{{2}}|19[0-9]{{2}})/);
    return m ? Number(m[1]) : null;
  }}

  function moneyFromB(v) {{
    const dollars = Number(v || 0) * 1000000000;
    const abs = Math.abs(dollars);
    const sign = dollars < 0 ? '-' : '';
    if (abs >= 1000000000) return sign + '$' + (abs / 1000000000).toFixed(2) + 'B';
    if (abs >= 100000000) return sign + '$' + (abs / 1000000).toFixed(0) + 'M';
    if (abs >= 10000000) return sign + '$' + (abs / 1000000).toFixed(1) + 'M';
    if (abs >= 1000000) return sign + '$' + (abs / 1000000).toFixed(2) + 'M';
    if (abs >= 1000) return sign + '$' + (abs / 1000).toFixed(0) + 'K';
    return sign + '$' + abs.toFixed(0);
  }}

  function pctText(v) {{
    return isFinite(Number(v)) ? Number(v).toFixed(1) + '%' : 'n/a';
  }}

  function cagrFromPairs(pairs, valueKey) {{
    const valid = pairs.filter(r => Number(r[valueKey] || 0) > 0 && r.fyStart !== null);
    if (valid.length < 2) return null;
    const first = valid[0];
    const last = valid[valid.length - 1];
    const periods = Math.max(0, Number(last.fyStart) - Number(first.fyStart));
    const firstVal = Number(first[valueKey] || 0);
    const lastVal = Number(last[valueKey] || 0);
    if (periods <= 0 || firstVal <= 0 || lastVal <= 0) return null;
    return (Math.pow(lastVal / firstVal, 1 / periods) - 1) * 100;
  }}

  function filteredDataset(d) {{
    const mode = rangeMode();
    const n = mode === 'last3' ? 3 : (mode === 'last5' ? 5 : d.financial_years.length);
    const start = Math.max(0, d.financial_years.length - n);
    const idx = d.financial_years.map((_, i) => i).slice(start);
    const rows = idx.map(i => {{
      const total = Number(d.values_b[i] || 0);
      const acc = Number(d.accenture_wins_b[i] || 0);
      const comp = Math.max(0, Number(d.competitor_values_b[i] || 0));
      const share = total > 0 ? (acc / total * 100) : null;
      return {{
        fy: d.financial_years[i],
        fyStart: parseFyStart(d.financial_years[i]),
        totalB: total,
        accB: acc,
        compB: comp,
        yoy: d.yoy[i],
        contracts: d.contracts[i],
        share: share,
        totalLabel: moneyFromB(total),
        accLabel: moneyFromB(acc),
        compLabel: moneyFromB(comp),
        shareLabel: share === null ? '' : pctText(share)
      }};
    }});

    const totalB = rows.reduce((a, r) => a + r.totalB, 0);
    const accB = rows.reduce((a, r) => a + r.accB, 0);
    const latest = rows[rows.length - 1] || null;
    const peak = rows.reduce((best, r) => !best || r.totalB > best.totalB ? r : best, null);
    const capabilityCagr = cagrFromPairs(rows, 'totalB');
    const shareRows = rows.map(r => ({{fyStart: r.fyStart, shareValue: r.share === null ? 0 : r.share}}));
    const shareCagr = cagrFromPairs(shareRows, 'shareValue');
    const shareTotal = totalB > 0 ? (accB / totalB * 100) : 0;

    return {{
      mode,
      label: mode === 'last3' ? 'Last 3 FY' : (mode === 'last5' ? 'Last 5 FY' : 'All years'),
      rows,
      totalB,
      accB,
      latest,
      peak,
      capabilityCagr,
      shareCagr,
      shareTotal
    }};
  }}

  function buildBarAnnotations(fd, maxBar) {{
    const annotations = [{{
      text: 'Grey = competitor-owned segment value. Purple = Accenture wins. Only Accenture-addressable contracts are included.',
      x: 0, y: 1.15, xref: 'paper', yref: 'paper', showarrow: false, xanchor: 'left',
      font: {{size: 12, color: '#526070'}}
    }}];

    fd.rows.forEach(row => {{
      const total = Number(row.totalB || 0);
      const acc = Number(row.accB || 0);
      const share = row.share;
      const competitor = Number(row.compB || 0);
      const tinyTotal = total < maxBar * 0.055;
      const hasAccenture = acc > 0.0005;

      // Keep the total-value label fully inside the grey section.
      // The fixed gap below the grey/purple boundary prevents the label
      // from touching or clipping into even very thin purple segments.
      const labelPadding = maxBar * 0.055;
      const greyLabelY = competitor > labelPadding * 1.8
        ? competitor - labelPadding
        : Math.max(competitor * 0.50, maxBar * 0.018);

      annotations.push({{
        x: row.fy,
        y: greyLabelY,
        xref: 'x',
        yref: 'y',
        text: row.totalLabel,
        showarrow: false,
        xanchor: 'center',
        yanchor: 'middle',
        font: {{size: 11, color: '#1f2937'}},
        bgcolor: 'rgba(0,0,0,0)',
        bordercolor: 'rgba(0,0,0,0)',
        borderpad: 0
      }});

      if (hasAccenture && share !== null && isFinite(share)) {{
        annotations.push({{
          x: row.fy,
          y: total + (maxBar * 0.06),
          xref: 'x',
          yref: 'y',
          text: '<b>' + pctText(share) + '</b>',
          showarrow: false,
          xanchor: 'center',
          yanchor: 'bottom',
          font: {{size: 12, color: '#A100FF'}},
          bgcolor: 'rgba(255,255,255,0.90)',
          borderpad: 3
        }});
      }}
    }});
    return annotations;
  }}

  function render(index) {{
    const d = datasets[index] || datasets[0];
    if (!d) return;
    const fd = filteredDataset(d);

    document.getElementById('cardTotal').textContent = moneyFromB(fd.totalB);
    document.getElementById('cardLatest').textContent = fd.latest ? (fd.latest.fy + ': ' + fd.latest.totalLabel) : 'n/a';
    document.getElementById('cardCagr').textContent = fd.capabilityCagr === null ? 'n/a' : pctText(fd.capabilityCagr);
    document.getElementById('cardShareCagr').textContent = fd.shareCagr === null ? 'n/a' : pctText(fd.shareCagr);
    document.getElementById('cardPeak').textContent = fd.peak ? (fd.peak.fy + ': ' + fd.peak.totalLabel) : 'n/a';
    document.getElementById('cardShare').textContent = pctText(fd.shareTotal);

    const competitorBar = {{
      type: 'bar',
      name: 'Competitor-owned segment value',
      x: fd.rows.map(r => r.fy),
      y: fd.rows.map(r => r.compB),
      marker: {{color: '#D7DEE9'}},
      text: [],
      cliponaxis: false,
      customdata: fd.rows.map(r => [r.totalLabel, r.compLabel, r.accLabel, r.contracts, r.share]),
      hovertemplate: '<b>%{{x}}</b><br>Total segment value: %{{customdata[0]}}<br>Competitor-owned value: %{{customdata[1]}}<br>Accenture wins: %{{customdata[2]}}<br>Contracts: %{{customdata[3]}}<br>Accenture share: %{{customdata[4]:,.1f}}%<extra></extra>'
    }};

    const accentureBar = {{
      type: 'bar',
      name: 'Accenture wins',
      x: fd.rows.map(r => r.fy),
      y: fd.rows.map(r => r.accB),
      marker: {{color: '#A100FF'}},
      text: [],
      cliponaxis: false,
      customdata: fd.rows.map(r => [r.accLabel, r.share, r.totalLabel]),
      hovertemplate: '<b>%{{x}}</b><br>Accenture wins: %{{customdata[0]}}<br>Accenture share: %{{customdata[1]:,.1f}}%<br>Total segment value: %{{customdata[2]}}<extra></extra>'
    }};
    const maxBar = Math.max(0.1, ...fd.rows.map(r => r.totalB));
    const layout = {{
      title: d.capability + ' - annual addressable Service Offering value<br><sup>' + fd.label + '</sup>',
      template: 'plotly_white',
      height: 670,
      margin: {{t: 95, l: 70, r: 90, b: 100}},
      legend: {{orientation: 'h', y: -0.18}},
      barmode: 'stack',
      bargap: fd.rows.length <= 3 ? 0.48 : 0.34,
      xaxis: {{title: 'Financial year', tickangle: -35}},
      yaxis: {{title: '$B', range: [0, maxBar * 1.46], showgrid: true}},

      annotations: buildBarAnnotations(fd, maxBar)
    }};

    Plotly.react(chartId, [competitorBar, accentureBar], layout, {{responsive:true, displayModeBar:false, staticPlot:false, scrollZoom:false, doubleClick:false}});
  }}

  select.addEventListener('change', () => render(Number(select.value)));
  document.querySelectorAll('input[name="growthRange"]').forEach(input => {{
    input.addEventListener('change', () => render(Number(select.value)));
  }});
  render(0);
}})();
</script>
"""


def build_ranked_growth_table(summary: pd.DataFrame) -> str:
    table = summary.copy()
    table["Total value"] = table["total_value"].map(short_money)
    table["Latest FY value"] = table["latest_value"].map(short_money)
    table["CAGR"] = table["cagr_pct"].map(lambda x: "n/a" if pd.isna(x) else f"{float(x):,.1f}%")
    table["Peak FY"] = table.apply(lambda r: f"{r['peak_year']} ({short_money(r['peak_value'])})", axis=1)
    table["Accenture share"] = table["accenture_share_pct"].map(lambda x: f"{float(x):,.1f}%")
    show = table[["capability", "Total value", "Accenture share", "Latest FY value", "CAGR", "Peak FY"]].rename(columns={
        "capability": "Service Offering",
    })
    rows = []
    for _, row in show.iterrows():
        cells = "".join(f"<td>{row[col]}</td>" for col in show.columns)
        rows.append(f"<tr>{cells}</tr>")
    header = "".join(f"<th>{col}</th>" for col in show.columns)
    return f"""
<div class="table-panel">
  <h3>Ranked growth summary</h3>
  <p>Use this table to identify which addressable segments are largest, which are peaking, and which have the strongest CAGR.</p>
  <div class="table-scroll">
    <table class="summary-table">
      <thead><tr>{header}</tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</div>
"""


def build_html(annual: pd.DataFrame, summary: pd.DataFrame, reinvention_annual: pd.DataFrame, reinvention_summary: pd.DataFrame, output_html: Path, value_mode: str) -> None:
    total_value = float(summary["total_value"].sum()) if not summary.empty else 0
    latest_total = float(summary["latest_value"].sum()) if not summary.empty else 0
    top_cap = summary.iloc[0]["capability"] if not summary.empty else "n/a"
    top_cap_value = float(summary.iloc[0]["total_value"]) if not summary.empty else 0.0

    fastest = summary.dropna(subset=["cagr_pct"]).sort_values("cagr_pct", ascending=False).head(1)
    fastest_label = fastest.iloc[0]["capability"] if not fastest.empty else "n/a"
    fastest_cagr = float(fastest.iloc[0]["cagr_pct"]) if not fastest.empty else None
    fastest_cagr_label = f"{fastest_cagr:,.1f}%" if fastest_cagr is not None else "n/a"
    basis_label = "Total contract value" if value_mode == "total" else "Annualised contract value / run-rate"
    years = annual[FIN_YEAR_COL].dropna().astype(str).unique().tolist()
    period = f"{years[0]} to {years[-1]}" if years else "n/a"

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Accenture Defence Service Offering Growth Explorer</title>
  <script src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; background: #f6f8fb; color: #1f2937; }}
    .wrap {{ max-width:1480px; margin:0 auto; padding:28px; }}
    h1 {{ margin:0 0 8px 0; font-size:30px; }}
    .subtitle {{ margin:0 0 20px 0; color:#526070; font-size:15px; line-height:1.45; }}
    .executive-summary {{
  background:#ffffff;
  border:1px solid #e5e7eb;
  border-left:5px solid #A100FF;
  border-radius:14px;
  padding:18px 20px;
  margin:0 0 18px 0;
  box-shadow:0 1px 2px rgba(0,0,0,0.04);
}}
.executive-summary h2 {{ margin:0 0 9px 0; font-size:20px; color:#111827; }}
.executive-summary p {{ margin:0 0 10px 0; color:#475467; line-height:1.55; }}
.executive-summary p:last-child {{ margin-bottom:0; }}

    .executive-summary strong[data-tooltip] {{
      color: #111827;
      border-bottom: 1px dotted #667085;
      cursor: help;
      outline: none;
    }}
    .exec-tooltip {{
      position: fixed;
      left: 0;
      top: 0;
      max-width: min(360px, calc(100vw - 24px));
      padding: 10px 12px;
      border-radius: 9px;
      background: #101828;
      color: #ffffff;
      font-size: 12px;
      font-weight: 400;
      line-height: 1.4;
      text-align: left;
      white-space: normal;
      box-shadow: 0 8px 24px rgba(16,24,40,.22);
      opacity: 0;
      visibility: hidden;
      pointer-events: none;
      z-index: 2147483647;
      transition: opacity .12s ease;
    }}
    .exec-tooltip.is-visible {{
      opacity: 1;
      visibility: visible;
    }}
    .cards {{ display: grid; grid-template-columns: repeat(5, minmax(190px, 1fr)); gap: 14px; margin: 18px 0 22px 0; }}
    .card {{ background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 17px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); min-width: 0; overflow: hidden; }}
    .card-title {{ font-size: clamp(11px, 0.9vw, 13px); text-transform: uppercase; letter-spacing: .04em; color: #667085; margin-bottom: 8px; line-height: 1.25; }}
    .card-value {{ font-size: clamp(20px, 1.8vw, 28px); font-weight: 700; color: #111827; margin-bottom: 5px; line-height: 1.08; overflow-wrap: anywhere; }}
    .card-subtitle {{ font-size: clamp(11px, 0.9vw, 13px); color: #667085; line-height: 1.3; overflow-wrap: anywhere; }}
    .card-subtitle b {{ color: #667085; font-weight: 400; font-family: inherit; }}
    .panel {{ background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 18px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); margin-bottom: 18px; }}
    .growth-header {{ display: flex; justify-content: space-between; gap: 18px; align-items: flex-start; }}
    .growth-header h2 {{ margin: 0 0 6px 0; font-size: 22px; }}
    .growth-header p {{ margin: 0; color: #526070; line-height: 1.45; }}
    .selector-box {{ min-width: 320px; }}
    .reinvention-selectors {{ display:flex; gap:12px; align-items:flex-end; flex-wrap:wrap; }}
    .reinvention-selectors .selector-box {{ min-width:260px; }}
    .selector-box label {{ display: block; font-size: 13px; color: #667085; margin-bottom: 6px; }}
    .selector-box select {{ width: 100%; border: 1px solid #d0d5dd; border-radius: 10px; padding: 10px 12px; font-size: 14px; background: #fff; }}
    .mini-cards {{ display: grid; grid-template-columns: repeat(6, minmax(140px, 1fr)); gap: 10px; margin: 16px 0 8px 0; }}
    .mini-card {{ background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; }}
    .mini-card span {{ display: block; font-size: 12px; color: #667085; margin-bottom: 6px; }}
    .mini-card b {{ font-size: 18px; color: #111827; }}
    .accenture-mini-card {{ border-color: #d9b8ff; background: #fbf7ff; }}
    .accenture-mini-card b {{ color: #A100FF; }}
    .range-controls {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 6px 0; }}
    .range-pill {{ display: inline-flex; align-items: center; gap: 6px; border: 1px solid #d0d5dd; border-radius: 999px; padding: 7px 12px; background: #ffffff; font-size: 12px; cursor: pointer; white-space: nowrap; }}
    .range-pill:has(input:checked) {{ background: #eef2ff; border-color: #636efa; color: #243b9f; font-weight: 600; }}
    .table-panel h3 {{ margin: 0 0 4px 0; font-size: 18px; }}
    .table-panel p {{ margin: 0 0 12px 0; color: #526070; font-size: 13px; }}
    .table-scroll {{ max-height: 520px; overflow: auto; border: 1px solid #e5e7eb; border-radius: 10px; }}
    .summary-table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    .summary-table th {{ position: sticky; top: 0; background: #f2f4f7; text-align: left; padding: 9px; border-bottom: 1px solid #e5e7eb; }}
    .summary-table td {{ padding: 8px 9px; border-bottom: 1px solid #eef2f7; }}
    .summary-table tr:hover {{ background: #fafafa; }}
    .note {{ color: #526070; font-size: 13px; line-height: 1.45; margin-top: 14px; }}
    @media (max-width: 1000px) {{ .cards, .mini-cards {{ grid-template-columns: 1fr; }} .growth-header {{ flex-direction: column; }} .selector-box {{ width: 100%; }} }}
  </style>
</head>
<body>
<div class="wrap">
  <h1>Accenture Defence Service Offering Growth Explorer</h1>
  <p class="subtitle">
     </p>

  <section class="executive-summary" aria-labelledby="executiveSummaryTitle">
    <h2 id="executiveSummaryTitle">Executive Summary</h2>
    <p>
      This dashboard provides an executive view of how
      <strong tabindex="0" data-tooltip="The Defence procurement segments classified as relevant to Accenture's consulting, technology, engineering, cloud, cyber, data and managed-services capabilities.">Accenture-addressable service offerings</strong>
      have evolved over the AusTender history. It highlights which service-offering areas are growing, contracting, peaking or becoming more volatile, helping identify where future demand may be emerging.
    </p>
    <p>
      The <strong tabindex="0" data-tooltip="Headline measures covering the selected period, total addressable value, latest financial-year value, the largest service offering and the fastest-growing service offering.">Headline KPI cards</strong>
      summarise the scale and direction of the market. The largest-service-offering card shows the highest-value segment across the selected period, while the fastest-growing card identifies the service offering with the strongest compound annual growth rate.
    </p>
    <p>
      The <strong tabindex="0" data-tooltip="An interactive financial-year chart where grey bars show competitor-owned addressable value, purple bars show Accenture wins and the line shows year-on-year segment growth.">Service Offering Growth Explorer</strong>
      allows each service offering to be analysed individually. Use the Service Offering selector and financial-year controls to compare long-term performance with the latest, last 3 and last 5 Financial Year periods.
    </p>
    <p>
      Together, these views support capability investment decisions, strategic workforce planning, go-to-market prioritisation and identification of Defence service-offering areas experiencing sustained growth.
      Results use Total Contract Value and exclude non-addressable Defence spend.
    </p>
  </section>

  <div class="cards">
    <div class="card"><div class="card-title">Selected period</div><div class="card-value">{period}</div><div class="card-subtitle">Financial years detected</div></div>
    <div class="card"><div class="card-title">Top-N segment total</div><div class="card-value">{short_money(total_value)}</div><div class="card-subtitle">Across selected Service Offerings</div></div>
    <div class="card"><div class="card-title">Latest FY total</div><div class="card-value">{short_money(latest_total)}</div><div class="card-subtitle">Sum of latest year values</div></div>
    <div class="card"><div class="card-title">Largest Service Offering</div><div class="card-value">{short_money(top_cap_value)}</div><div class="card-subtitle"><b>{top_cap}</b></div></div>
    <div class="card"><div class="card-title">Fastest growing</div><div class="card-value">{fastest_cagr_label}</div><div class="card-subtitle"><b>{fastest_label}</b></div></div>
  </div>

  <div class="panel">
    {build_interactive_growth_chart(annual, summary, value_mode)}
  </div>

  <div class="panel">
    {build_reinvention_growth_chart(reinvention_annual, reinvention_summary, value_mode)}
  </div>

  <div class="panel">
    {build_ranked_growth_table(summary)}
  </div>

  <p class="note">
    Note: YoY growth is blank where the prior-year value is zero. Use <code>--value-mode annualised</code> for run-rate style forecasting, or the default total value mode for award-flow analysis.
  </p>
</div>

<div id="execTooltip" class="exec-tooltip" role="tooltip" aria-hidden="true"></div>
<script>
(function() {{
  const tooltip = document.getElementById('execTooltip');
  if (!tooltip) return;

  let activeTarget = null;

  function placeTooltip(target) {{
    if (!activeTarget || activeTarget !== target) return;

    const rect = target.getBoundingClientRect();
    const tipRect = tooltip.getBoundingClientRect();
    const gap = 10;
    const edge = 12;

    let left = rect.left + (rect.width / 2) - (tipRect.width / 2);
    left = Math.max(edge, Math.min(left, window.innerWidth - tipRect.width - edge));

    let top = rect.top - tipRect.height - gap;
    if (top < edge) {{
      top = rect.bottom + gap;
    }}
    top = Math.max(edge, Math.min(top, window.innerHeight - tipRect.height - edge));

    tooltip.style.left = Math.round(left) + 'px';
    tooltip.style.top = Math.round(top) + 'px';
  }}

  function showTooltip(target) {{
    const text = target.getAttribute('data-tooltip');
    if (!text) return;

    activeTarget = target;
    tooltip.textContent = text;
    tooltip.classList.add('is-visible');
    tooltip.setAttribute('aria-hidden', 'false');

    requestAnimationFrame(() => placeTooltip(target));
  }}

  function hideTooltip(target) {{
    if (target && activeTarget !== target) return;
    activeTarget = null;
    tooltip.classList.remove('is-visible');
    tooltip.setAttribute('aria-hidden', 'true');
  }}

  document.querySelectorAll('.executive-summary strong[data-tooltip]').forEach(target => {{
    target.addEventListener('mouseenter', () => showTooltip(target));
    target.addEventListener('mouseleave', () => hideTooltip(target));
    target.addEventListener('focus', () => showTooltip(target));
    target.addEventListener('blur', () => hideTooltip(target));
  }});

  window.addEventListener('scroll', () => {{
    if (activeTarget) placeTooltip(activeTarget);
  }}, {{ passive: true }});

  window.addEventListener('resize', () => {{
    if (activeTarget) placeTooltip(activeTarget);
  }});
}})();
</script>

</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_all = load_or_classify(args)
    value_col = get_value_col(df_all, args.value_mode)
    df_all = clean_value_columns(df_all, [VALUE_COL, ANNUALISED_VALUE_COL])
    df = filter_defence_scope(df_all, include_all_agencies=args.include_all_agencies)
    annual, summary = prepare_growth_data(df, value_col, args.top_n_capabilities, args.min_year_value)
    reinvention_annual, reinvention_summary = prepare_reinvention_growth_data(df, value_col, args.min_year_value)

    annual_path = output_dir / "capability_growth_by_year.csv"
    summary_path = output_dir / "capability_growth_summary.csv"
    reinvention_annual_path = output_dir / "reinvention_growth_by_year.csv"
    reinvention_summary_path = output_dir / "reinvention_growth_summary.csv"
    dashboard_path = output_dir / "ServiceGrowthDashboard.html"

    export_defence_scope_audit(df_all, df, value_col, output_dir)

    annual.to_csv(annual_path, index=False)
    slim_cols = [c for c in [AGENCY_COL, AGENCY_DIVISION_COL, AGENCY_BRANCH_COL, FIN_YEAR_COL, CN_ID_COL, SUPPLIER_COL, DESCRIPTION_COL, "capability", "ReinventionPartner", "ReinventionEngine", "reinvention_mapping_confidence", "reinvention_mapping_reason", "is_addressable", "is_accenture", value_col] if c in df.columns]
    df[slim_cols].to_csv(output_dir / "defence_filtered_contracts_slim.csv", index=False)
    summary.to_csv(summary_path, index=False)
    reinvention_annual.to_csv(reinvention_annual_path, index=False)
    reinvention_summary.to_csv(reinvention_summary_path, index=False)
    build_html(annual, summary, reinvention_annual, reinvention_summary, dashboard_path, args.value_mode)

    print("Service Offering growth dashboard complete")
    print(f"Value mode: {args.value_mode} / column used: {value_col}")
    print(f"Service Offerings shown: {len(summary)}")
    print(f"Reinvention combinations shown: {len(reinvention_summary)}")
    print(f"Rows after Defence filter: {len(df):,}")
    print(f"Defence filter active: {not args.include_all_agencies}")
    print(f"Filtered Defence addressable value: {short_money(summary['total_value'].sum() if not summary.empty else 0)}")
    print(f"Wrote: {dashboard_path}")
    print(f"Wrote: {annual_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {reinvention_annual_path}")
    print(f"Wrote: {reinvention_summary_path}")
    print(f"Wrote: {output_dir / 'defence_filtered_contracts_slim.csv'}")
    print(f"Wrote: {output_dir / 'defence_scope_audit_by_agency.csv'}")
    print(f"Wrote: {output_dir / 'defence_scope_audit_by_org.csv'}")


if __name__ == "__main__":
    main()