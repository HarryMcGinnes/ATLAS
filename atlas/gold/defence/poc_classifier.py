from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# =============================================================================
# FRESH MASTER BUILD - BASE-HEADER FIRST
# =============================================================================
# Design principle:
#   * Never rename or overwrite source columns from the raw parquet.
#   * Read the original AusTender headers exactly as supplied.
#   * Add derived/canonical columns only.
#   * Addressability is decided in this order:
#       1) Category Type
#       2) Category
#       3) Description
#       4) Winning supplier as weak supporting context only
#
# The dashboards continue to receive:
#   master_output/master_defence_contracts.parquet
# with their expected derived fields.
# =============================================================================

# -------------------------------
# Raw source headers - DO NOT RENAME
# -------------------------------
CN_ID = "CN ID"
AGENCY = "Agency"
CONTRACT_TYPE = "Contract Type"
FINANCIAL_YEAR = "Financial Year"
DESCRIPTION = "Description"
CATEGORY_TYPE = "Category Type"
CATEGORY_CODE = "Category Code"
CATEGORY = "Category"
SUPPLIER_NAME = "Supplier Name"
SUPPLIER_ABN = "Supplier ABN"
RAW_DIVISION = "Agency Divison"  # exact source spelling
RAW_BRANCH = "Agency Branch"
VALUE = "Value"
VALUE_PER_YEAR = "Value Per Year"
START_DATE = "Start Date"
END_DATE = "End Date"
SOURCE_SUPPLIER_DISPLAY = "supplier_display"
SOURCE_DIV_CLEAN = "div_clean"
SOURCE_BRANCH_CLEAN = "branch_clean"

# Derived compatibility fields used by existing dashboards.
CANON_DIVISION = "Agency Division"
CANON_BRANCH = "Agency Branch"  # already exists in source; do not overwrite

FINAL_SERVICE_OFFERINGS = [
    "Strategy, Transformation & Advisory",
    "SI & Engineering",
    "Data, AI & Automation",
    "Cloud Infrastructure & Cyber",
    "Managed Services & Operations",
]

DEFENCE_AGENCIES = {
    "department of defence",
    "australian signals directorate",
    "australian submarine agency",
}

MANUAL_NON_ADDRESSABLE_CN_IDS = {
    "CN3840723", "CN4050683", "CN359557", "CN2953962", "CN1384831",
    "CN3486107", "CN3658983", "CN3667078", "CN3296931", "CN1926632",
    "CN4179707",

}

AUTHORITATIVE_SERVICE_OFFERING_OVERRIDES = {
    "CN4066214": "Data, AI & Automation",
}

# -----------------------------------------------------------------------------
# Text helpers
# -----------------------------------------------------------------------------
def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9+#./ -]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def upper_cn(value: object) -> str:
    return str(value or "").strip().upper()


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def normalise_abn(value: object) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    return digits if len(digits) == 11 else ""


def normalise_supplier(value: object) -> str:
    text = clean(value).upper()
    return re.sub(r"\s+", " ", text).strip()


# Supplier identity is deliberately conservative.
#
# IMPORTANT INVARIANT:
#   supplier consolidation is a LABEL transformation only. It must never add,
#   remove, duplicate or revalue a contract. The raw Supplier Name and Supplier
#   ABN columns are never modified.
#
# The source parquet already contains a useful supplier_display field. We use it
# as the baseline when available, then apply only explicit, reviewed family rules
# for major suppliers. We DO NOT propagate a family across every row sharing an
# ABN: that was too aggressive and could move large values between suppliers.
#
# Patterns are intentionally specific to known legal/trading-name variants seen
# in the source extract. Add new aliases here only when they are clearly the same
# supplier family.
CANONICAL_SUPPLIER_FAMILIES: list[tuple[str, list[str]]] = [
    ("Accenture", [
        r"\bACCENTURE\b",
    ]),
    ("Deloitte", [
        r"\bDELOITTE\b", r"\bDELOITTE TOUCHE TOHMATSU\b",
    ]),
    ("EY", [
        r"\bERNST\s*(?:&|AND)\s*YOUNG\b",
        r"^EY(?:\s+(?:AUSTRALIA|OCEANIA|PTY|LIMITED|LLP))?\b",
    ]),
    ("KPMG", [r"\bKPMG\b"]),
    ("PwC", [
        r"\bPRICEWATERHOUSECOOPERS\b", r"\bPRICEWATERHOUSE\s+COOPERS\b", r"\bPWC\b",
    ]),
    ("Lockheed Martin", [r"\bLOCKHEED\s+MARTIN\b"]),
    ("Leidos", [r"\bLEIDOS\b"]),
    ("Fujitsu", [r"\bFUJITSU\b"]),
    ("Thales", [r"\bTHALES\b"]),
    ("IBM", [r"\bIBM\b", r"\bINTERNATIONAL\s+BUSINESS\s+MACHINES\b"]),
    ("BAE Systems", [r"\bBAE\s+SYSTEMS\b"]),
    ("Boeing", [
        r"\bBOEING\b", r"\bJEPPESEN\b.*\bBOEING\b",
    ]),
    ("Northrop Grumman", [r"\bNORTHROP\s+GRUMMAN\b"]),
    ("Raytheon", [r"\bRAYTHEON\b"]),
    ("Rheinmetall", [r"\bRHEINMETALL\b"]),
    ("Saab", [r"\bSAAB\b"]),
    ("Airbus", [r"\bAIRBUS\b", r"\bEADS\b.*\bCASA\b"]),
    ("DXC Technology", [r"\bDXC\s+TECHNOLOGY\b", r"^DXC\b"]),
    ("Capgemini", [r"\bCAPGEMINI\b"]),
    ("Atos / Eviden", [r"\bATOS\b", r"\bEVIDEN\b"]),
    ("Microsoft", [r"\bMICROSOFT\b"]),
    ("Amazon Web Services", [r"\bAMAZON\s+WEB\s+SERVICES\b", r"^AWS\b"]),
    ("Oracle", [r"\bORACLE\b"]),
    ("SAP", [r"\bSAP\b"]),
    ("Unisys", [r"\bUNISYS\b"]),
    ("Kinetic IT", [r"\bKINETIC\s+IT\b"]),
    ("Data#3", [r"\bDATA#?3\b"]),
    ("Telstra", [r"\bTELSTRA\b", r"\bCOBS\s+TELSTRA\b"]),
    ("Optus", [r"\bOPTUS\b"]),
    ("ASC", [r"^ASC\s+(?:PTY\s+LTD|SHIPBUILDING\s+PTY\s+LTD)$"]),
    ("Lendlease", [r"\bLEND\s+LEASE\b", r"\bLENDLEASE\b"]),
    ("Jacobs", [r"\bJACOBS\b"]),
    ("Downer", [r"\bDOWNER\b"]),
    ("Ventia", [r"\bVENTIA\b"]),
    ("Nova Systems", [r"\bNOVA\s+SYSTEMS\b"]),
    ("Qinetiq", [r"\bQINETIQ\b"]),
    ("CAE", [r"\bCAE\s+AUSTRALIA\b"]),
    ("Elbit Systems", [r"\bELBIT\s+SYSTEMS\b"]),
    ("RPS", [r"\bRPS\s+AAP\s+CONSULTING\b"]),
]


def _family_match(text: str) -> str:
    """Return one reviewed canonical family or an empty string.

    Family rules are evaluated against a single normalised text string. The first
    match wins; patterns are ordered from specific/high-value families to more
    general ones. No ABN-based propagation is performed here.
    """
    for canonical, patterns in CANONICAL_SUPPLIER_FAMILIES:
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            return canonical
    return ""


def _source_supplier_baseline(out: pd.DataFrame) -> pd.Series:
    raw = out[SUPPLIER_NAME].fillna("").astype(str).str.strip()
    if SOURCE_SUPPLIER_DISPLAY in out.columns:
        source_display = out[SOURCE_SUPPLIER_DISPLAY].fillna("").astype(str).str.strip()
        return source_display.where(source_display.ne(""), raw).where(lambda x: x.ne(""), "Unknown Supplier")
    return raw.where(raw.ne(""), "Unknown Supplier")


def _accenture_source_mask(out: pd.DataFrame) -> pd.Series:
    """Detect Accenture from raw/source identity, before derived grouping exists."""
    raw = out[SUPPLIER_NAME].fillna("").astype(str).map(normalise_supplier)
    mask = raw.str.contains(r"\bACCENTURE\b", regex=True, na=False)
    if SOURCE_SUPPLIER_DISPLAY in out.columns:
        src = out[SOURCE_SUPPLIER_DISPLAY].fillna("").astype(str).map(normalise_supplier)
        mask = mask | src.str.fullmatch(r"ACCENTURE", na=False) | src.str.contains(r"\bACCENTURE\b", regex=True, na=False)
    return mask


def _validate_supplier_label_invariants(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Fail loudly if supplier grouping changes the population or contract value."""
    if len(before) != len(after):
        raise RuntimeError(
            f"Supplier grouping changed row count: {len(before):,} -> {len(after):,}. "
            "Supplier consolidation must be label-only."
        )
    before_value = pd.to_numeric(before[VALUE], errors="coerce").fillna(0).sum()
    after_value = pd.to_numeric(after[VALUE], errors="coerce").fillna(0).sum()
    if abs(float(before_value) - float(after_value)) > 0.01:
        raise RuntimeError(
            f"Supplier grouping changed Defence value: ${before_value:,.2f} -> ${after_value:,.2f}."
        )

    # Every source-identified Accenture record must still be Accenture afterwards.
    acc_source = _accenture_source_mask(before)
    if acc_source.any():
        mapped = after.loc[acc_source, "supplier_group"].fillna("").astype(str).str.strip()
        bad = ~mapped.eq("Accenture")
        if bad.any():
            cols = [c for c in [CN_ID, SUPPLIER_NAME, SUPPLIER_ABN, SOURCE_SUPPLIER_DISPLAY] if c in before.columns]
            sample = before.loc[acc_source].loc[bad.values, cols].head(10).to_dict("records")
            raise RuntimeError(f"Accenture supplier reconciliation failed; examples: {sample}")


def consolidate_major_supplier_families(out: pd.DataFrame) -> pd.DataFrame:
    """Conservatively canonicalise reviewed supplier families.

    Raw Supplier Name / ABN are never modified. We first trust the source
    supplier_display as the baseline, then canonicalise only when either the raw
    name or baseline display independently matches a reviewed family rule.
    """
    before = out.copy()
    baseline = _source_supplier_baseline(out)
    raw_norm = out[SUPPLIER_NAME].fillna("").astype(str).map(normalise_supplier)
    base_norm = baseline.map(normalise_supplier)

    out["supplier_group_pre_family"] = baseline
    out["supplier_group"] = baseline
    out["supplier_mapping_method"] = "SOURCE_SUPPLIER_DISPLAY" if SOURCE_SUPPLIER_DISPLAY in out.columns else "RAW_SUPPLIER_NAME"

    raw_family = raw_norm.map(_family_match)
    base_family = base_norm.map(_family_match)

    # Prefer raw-name family evidence. If raw name is not recognised, a curated
    # source supplier_display may still identify the family.
    chosen_family = raw_family.where(raw_family.ne(""), base_family)
    family_mask = chosen_family.ne("")
    out.loc[family_mask, "supplier_group"] = chosen_family[family_mask]
    out.loc[family_mask, "supplier_mapping_method"] = "REVIEWED_FAMILY_ALIAS"

    # Accenture is a hard identity invariant because supplier performance depends
    # on it and raw/source names are unambiguous in this dataset.
    acc_source = _accenture_source_mask(out)
    out.loc[acc_source, "supplier_group"] = "Accenture"
    out.loc[acc_source, "supplier_mapping_method"] = "ACCENTURE_IDENTITY_GUARANTEE"

    out["supplier_display"] = out["supplier_group"]
    out["supplier_id"] = out["supplier_group"].map(
        lambda x: re.sub(r"[^A-Z0-9]+", "_", str(x).upper()).strip("_") or "UNKNOWN"
    )
    out["is_accenture"] = out["supplier_group"].eq("Accenture")

    _validate_supplier_label_invariants(before, out)
    return out



# Supplier cohorts are used only as context for ambiguous Goods records.
# They never override a clearly physical/military Category.
DEFENCE_PRIME_SUPPLIERS = {
    "BAE Systems", "Lockheed Martin", "Northrop Grumman", "Boeing",
    "Thales", "Raytheon", "Rheinmetall", "Navantia", "Austal",
    "Babcock", "ASC", "Elbit Systems",
}

CONSULTING_SUPPLIERS = {
    "Accenture", "Deloitte", "EY", "KPMG", "PwC", "Capgemini",
}

TECH_SERVICES_SUPPLIERS = {
    "IBM", "Fujitsu", "Leidos", "DXC Technology", "Atos / Eviden", "Microsoft",
    "Amazon Web Services", "Oracle", "SAP", "Kinetic IT", "Unisys", "Data#3",
    "Esri-Australia Pty Ltd",
}

DEFENCE_PRIME_PATTERNS = [
    r"\bBAE SYSTEMS\b", r"\bLOCKHEED MARTIN\b", r"\bNORTHROP GRUMMAN\b",
    r"\bBOEING\b", r"\bTHALES\b", r"\bRAYTHEON\b", r"\bRHEINMETALL\b",
    r"\bNAVANTIA\b", r"\bAUSTAL\b", r"\bBABCOCK\b", r"\bASC\b",
    r"\bELBIT SYSTEMS\b",
]

CONSULTING_PATTERNS = [
    r"\bACCENTURE\b", r"\bDELOITTE\b", r"\bERNST(?:\s*&\s*|\s+AND\s+)YOUNG\b",
    r"\bEY\b", r"\bKPMG\b", r"\bPWC\b", r"\bPRICEWATERHOUSECOOPERS\b",
    r"\bCAPGEMINI\b",
]

TECH_SERVICES_PATTERNS = [
    r"\bIBM\b", r"\bFUJITSU\b", r"\bLEIDOS\b", r"\bDXC\b",
    r"\bATOS\b", r"\bEVIDEN\b", r"\bMICROSOFT\b",
    r"\bAMAZON WEB SERVICES\b", r"\bAWS\b", r"\bORACLE\b", r"\bSAP\b",
    r"\bKINETIC IT\b", r"\bUNISYS\b", r"\bDATA#?3\b", r"\bESRI(?:-AUSTRALIA)?\b",
]

# Physical construction / facilities delivery suppliers are used only to resolve
# otherwise-generic project-management/support descriptions. Supplier identity
# alone never excludes a contract; an explicit physical/infrastructure context
# is also required, and explicit digital/advisory subject matter can rescue it.
PHYSICAL_FACILITIES_CONSTRUCTION_SUPPLIER_PATTERNS = [
    r"\bLEND(?:\s+)?LEASE\b",
    r"\bLENDLEASE\b",
    r"\bSPOTLESS FACILITY SERVICES\b",
    r"\bRIVERINA REDEVELOPMENT\b",
    r"\bHUTCHINSON BUILDERS?\b",
    r"\bWATPAC CONSTRUCTION\b",
    r"\bRMS ENGINEERING (?:AND|&) CONSTRUCTION\b",
    r"\bCPB CONTRACTORS\b",
]


# Generic physical data-centre facility delivery is outside the TAM when the
# supplier is acting as a data-centre/colocation facility operator. This is
# intentionally narrow: explicit hosting, cloud, network, support, project,
# specialist or advisory work is still handled by the normal digital rules.
DATA_CENTRE_FACILITY_OPERATOR_PATTERNS = [
    r"\bCANBERRA DATA CENTRES?\b",
    r"\bNEXTDC\b",
    r"\bEQUINIX\b",
    r"\bGLOBAL SWITCH\b",
]

GENERIC_DATA_CENTRE_FACILITY_DESCRIPTION_PATTERNS = [
    r"^data cent(?:re|er) services?$",
    r"^data cent(?:re|er) facilit(?:y|ies)(?: services?)?$",
]

PHYSICAL_FACILITIES_DESCRIPTION_PATTERNS = [
    r"^(?:construction services?|professional construction services?|civil works?|design and construct services?)$",
    r"\b(?:building|facility|facilities|estate|garrison|base)\b.*\b(?:construction|redevelopment|upgrade|works|maintenance|refurbishment|fitout|fit out)\b",
    r"\b(?:construction|redevelopment|upgrade|works|maintenance|refurbishment|fitout|fit out)\b.*\b(?:building|facility|facilities|estate|garrison|base)\b",
    r"\bredevelopment project support services?\b",
    # Defence capability facilities projects are physical estate/infrastructure work
    # unless the Description explicitly identifies a digital/advisory subject.
    # Example: "Air Combat Capability Facilities Project".
    r"\b(?:air combat|combat|aviation|aircraft|naval|maritime|army|land|military|defence)\b.*\b(?:capability\s+)?facilit(?:y|ies)\s+project\b",
    r"\b(?:capability\s+)?facilit(?:y|ies)\s+project\b.*\b(?:air combat|combat|aviation|aircraft|naval|maritime|army|land|military|defence)\b",
    r"\bbase services? re[- ]?tender(?:ing)?\b",
    r"\bfacilit(?:y|ies) management services?\b",
]

GENERIC_PROJECT_DELIVERY_PATTERNS = [
    r"^management costs? and services?$",
    r"^additional funding(?: for .+)?$",
    r"^project support services?$",
    r"^project management services?$",
    r"^project management$",
    r"^contract services?$",
]

INFRASTRUCTURE_ORG_PATTERNS = [
    r"\binfrastructure division\b",
    r"\bestate and infrastructure\b",
    r"\bservice delivery division\b",
    r"\bgarrison\b",
    r"\bestate\b",
    r"\bfacilit(?:y|ies)\b",
]

FACILITIES_DIGITAL_OR_ADVISORY_RESCUE_PATTERNS = [
    r"\b(?:software|digital|ict|cloud|data|analytics|cyber|application|database|erp|sap|servicenow|technology)\b",
    r"\b(?:information system|management system|visualisation tool|visualization tool|digital twin)\b",
    r"\b(?:strategy|strategic advisory|advisory services?|governance|business case|feasibility|transformation)\b",
]

def supplier_cohort(value: object) -> str:
    raw = str(value or "").strip()
    if raw in DEFENCE_PRIME_SUPPLIERS:
        return "Defence prime / platform OEM"
    if raw in CONSULTING_SUPPLIERS:
        return "Consulting / professional services"
    if raw in TECH_SERVICES_SUPPLIERS:
        return "Technology / managed services"
    text = normalise_supplier(raw)
    if any(re.search(p, text, flags=re.IGNORECASE) for p in DEFENCE_PRIME_PATTERNS):
        return "Defence prime / platform OEM"
    if any(re.search(p, text, flags=re.IGNORECASE) for p in CONSULTING_PATTERNS):
        return "Consulting / professional services"
    if any(re.search(p, text, flags=re.IGNORECASE) for p in TECH_SERVICES_PATTERNS):
        return "Technology / managed services"
    return "Other / unknown"


# =============================================================================
# 1. SOURCE VALIDATION
# =============================================================================
REQUIRED_SOURCE_COLUMNS = {
    CN_ID, AGENCY, DESCRIPTION, CATEGORY_TYPE, CATEGORY, SUPPLIER_NAME,
    RAW_DIVISION, RAW_BRANCH, VALUE,
}


def validate_source_headers(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_SOURCE_COLUMNS - set(df.columns))
    if missing:
        raise RuntimeError(
            "Fresh source parquet is missing expected base columns: " + ", ".join(missing)
        )
    dupes = list(df.columns[df.columns.duplicated()])
    if dupes:
        raise RuntimeError("Source parquet contains duplicate headers: " + ", ".join(map(str, dupes)))


# =============================================================================
# 2. DEFENCE SCOPE
# =============================================================================
def strict_defence_filter(df: pd.DataFrame) -> pd.DataFrame:
    key = df[AGENCY].fillna("").astype(str).str.strip().str.lower()
    out = df.loc[key.isin(DEFENCE_AGENCIES)].copy()
    if out.empty:
        raise RuntimeError("Strict Defence Agency filter returned no rows.")
    out["is_defence_scope"] = True
    return out


# =============================================================================
# 3. SUPPLIER CONSOLIDATION - EXACT MAPPING ONLY
# =============================================================================
def apply_supplier_mapping(df: pd.DataFrame, mapping_path: Path) -> pd.DataFrame:
    """Build supplier_group without changing the source population.

    The old persistent workbook is no longer used as a source of truth for family
    propagation because stale or over-broad mappings can move large contract values.
    It is retained only as an optional audit reference. The production identity is:

      raw Supplier Name / source supplier_display -> reviewed family aliases

    This makes the build upgradeable to new extracts: unknown suppliers simply keep
    their source identity until a reviewed alias is added.
    """
    out = df.copy()
    out["supplier_name_normalized"] = out[SUPPLIER_NAME].map(normalise_supplier)
    out["supplier_abn_normalized"] = out.get(
        SUPPLIER_ABN, pd.Series("", index=out.index)
    ).map(normalise_abn)

    # Preserve any workbook mapping for comparison only; never let it silently
    # override the source identity in production.
    out["supplier_workbook_reference"] = ""
    if mapping_path.exists():
        try:
            mapping = pd.read_excel(mapping_path, sheet_name="Supplier Mapping", dtype=str).fillna("")
            key_cols = ["supplier_name_normalized", "supplier_abn_normalized"]
            if set(key_cols + ["canonical_supplier"]).issubset(mapping.columns):
                ref = mapping.drop_duplicates(key_cols, keep="first")[key_cols + ["canonical_supplier"]]
                ref = ref.rename(columns={"canonical_supplier": "supplier_workbook_reference"})
                out = out.drop(columns=["supplier_workbook_reference"]).merge(
                    ref, on=key_cols, how="left", validate="many_to_one", sort=False
                )
                out["supplier_workbook_reference"] = out["supplier_workbook_reference"].fillna("")
        except Exception as exc:
            print(f"Supplier workbook audit reference skipped: {exc}")

    out = consolidate_major_supplier_families(out)
    return out


# =============================================================================
# 4. DIVISION / BRANCH - PRESERVE RAW HEADERS
# =============================================================================
def add_canonical_org_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Never modify Agency Divison. Add exactly one canonical alias for dashboard use.
    if SOURCE_DIV_CLEAN in out.columns:
        canonical = out[SOURCE_DIV_CLEAN].fillna("").astype(str).str.strip()
        raw = out[RAW_DIVISION].fillna("").astype(str).str.strip()
        out[CANON_DIVISION] = canonical.where(canonical.ne(""), raw)
    else:
        out[CANON_DIVISION] = out[RAW_DIVISION].fillna("").astype(str).str.strip()

    # Branch already has the correct base header. Preserve it exactly.
    # If branch_clean exists, keep it as source metadata only; do not overwrite Agency Branch.

    # Confirmed DDG resolution applies only to the derived canonical Division.
    raw_div = out[RAW_DIVISION].fillna("").astype(str).str.upper()
    canon_div = out[CANON_DIVISION].fillna("").astype(str).str.upper()
    branch = out[RAW_BRANCH].fillna("").astype(str).str.upper()
    ddg = (
        raw_div.str.fullmatch(r"\s*DDG\s*", na=False)
        | canon_div.str.contains("DDG - UNRESOLVED", na=False)
        | canon_div.str.contains("DEFENCE DIGITAL GROUP", na=False)
        | branch.str.startswith("DDG", na=False)
    ) & ~raw_div.str.contains("DEFENCE DELIVERY GROUP", na=False)
    out.loc[ddg, CANON_DIVISION] = "DDG - Defence Digital Group"
    out["division_mapping_status"] = "Source-derived"
    out.loc[ddg, "division_mapping_status"] = "Confirmed DDG - Defence Digital Group"
    return out


# =============================================================================
# 5. DEFENCE DOMAIN
# =============================================================================
def load_domain_lookup(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    lookup = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    rename = {}
    for c in lookup.columns:
        compact = re.sub(r"[^a-z]", "", str(c).lower())
        if compact in {"agencydivision", "agencydivison", "division"}:
            rename[c] = "division"
        elif compact in {"agencybranch", "branch"}:
            rename[c] = "branch"
        elif compact == "domain":
            rename[c] = "domain"
    lookup = lookup.rename(columns=rename)
    if not {"division", "branch", "domain"}.issubset(lookup.columns):
        raise RuntimeError("Domain lookup must contain Division, Branch and Domain columns.")
    def key(v: object) -> str:
        return re.sub(r"[^A-Z0-9]+", "", str(v or "").upper())
    result = {}
    for d, b, domain in lookup[["division", "branch", "domain"]].itertuples(index=False, name=None):
        result[(key(d), key(b))] = str(domain).strip() or "Unmapped"
    return result


def add_defence_domain(df: pd.DataFrame, lookup_path: Path) -> pd.DataFrame:
    out = df.copy()
    mapping = load_domain_lookup(lookup_path)
    def key(v: object) -> str:
        return re.sub(r"[^A-Z0-9]+", "", str(v or "").upper())

    domains = []
    sources = []
    confidences = []
    for agency, raw_div, branch in zip(out[AGENCY], out[RAW_DIVISION], out[RAW_BRANCH]):
        agency_l = clean(agency)
        if agency_l == "australian signals directorate":
            domains.append("Cyber"); sources.append("Authoritative procuring agency"); confidences.append(100); continue
        if agency_l == "australian submarine agency":
            domains.append("Maritime"); sources.append("Authoritative procuring agency"); confidences.append(100); continue
        domain = mapping.get((key(raw_div), key(branch)), "Unmapped")
        domains.append(domain)
        sources.append("Division + Branch lookup" if domain != "Unmapped" else "No lookup match")
        confidences.append(100 if domain != "Unmapped" else 0)

    out["defence_domain"] = domains
    out["domain_mapping_source"] = sources
    out["domain_confidence"] = confidences
    out["domain_mapping_reason"] = out["domain_mapping_source"]
    return out


# =============================================================================
# 6. SERVICE OFFERING CLASSIFIER
# =============================================================================
# First gate: Category Type = Goods.
# Goods use a three-stage hierarchy:
#   1) hard physical / military Category -> non-addressable, regardless of Description;
#   2) explicitly digital Category -> continue to normal Service Offering scoring;
#   3) otherwise ambiguous Goods -> Description may rescue only with strong, explicit
#      digital/service evidence. Generic words such as support, system, project,
#      management, training or engineering are never enough on their own.

HARD_NON_ADDRESSABLE_GOODS_CATEGORY_PATTERNS = [
    r"\bcomponents?\b",
    r"\bequipment\b",
    r"\baccessories\b",
    r"\bdevices?\b",
    r"\bparts?\b",
    r"\bassembl(?:y|ies)\b",
    r"\bsubassembl(?:y|ies)\b",
    r"\bhardware\b",
    r"\bmachinery\b",
    r"\bvehicles?\b",
    r"\bwatercraft\b",
    r"\baircraft\b",
    r"\baerospace systems?\b",
    r"\bweapons?\b",
    r"\bmissiles?\b",
    r"\bmunitions?\b",
    r"\bammunition\b",
    r"\bexplosives?\b",
    r"\bfirearms?\b",
    r"\btelecommunications equipment\b",
    r"\bcommunications devices?\b",
    r"\bcomputer equipment\b",
    r"\belectronic components?\b",
    r"\btest(?:ing)? equipment\b",
    r"\bdiagnostic equipment\b",
    r"\bmeasurement equipment\b",
    r"\bprotective equipment\b",
    r"\bclothing\b",
    r"\buniforms?\b",
    r"\bfuel\b",
    r"\blaunchers?\b",
    r"\brockets?\b",
    r"\bsubsystems?\b",
    r"\bguided weapons?\b",
    r"\bcombat systems?\b",
    r"\bradar(?:s)?\b",
    r"\bsonar\b",
    r"\bfire control systems?\b",
    r"\bcomputer servers?\b",
    r"\bserver hardware\b",
    r"\bracks?\b",
    r"\bprinters?\b",
]

# Software/licence-style goods can remain in scope despite Category Type = Goods.
DIGITAL_GOODS_EXCEPTIONS = [
    r"\blicen[cs]e\b",
    r"\bsubscription\b",
    r"\bcloud\b",
    r"\bsaas\b",
]

# Only these specific Description signals can rescue an otherwise ambiguous Goods
# Category. They are intentionally narrow and subject-specific.
STRONG_ADDRESSABLE_GOODS_DESCRIPTION_PATTERNS = [
    r"\bsoftware (?:licen[cs]e|subscription|development|engineering|implementation|migration|support|maintenance)\b",
    r"\b(?:saas|software as a service|cloud services?|cloud subscription|cloud platform)\b",
    r"\b(?:data platform|data engineering|data migration|data analytics|business intelligence|power bi)\b",
    r"\b(?:cybersecurity|cyber security|security operations|siem|soc|identity and access management)\b",
    r"\b(?:application development|application implementation|systems integration|system integration)\b",
]

NON_ADDRESSABLE_SERVICE_CATEGORY_PATTERNS = [
    r"\breal estate\b", r"\bproperty\b", r"\bconstruction\b", r"\bbuilding\b",
    r"\bfacility maintenance\b", r"\bfacilities management\b", r"\bcleaning\b",
    r"\bcatering\b", r"\bwaste\b", r"\btransportation\b", r"\bfreight\b",
    r"\bhotel rooms?\b", r"\bhotels and lodging and meeting facilities\b",
    r"\btaxi services?\b", r"\bpassenger transport(?:ation)?\b", r"\btravel facilitation\b",
    r"\brelocation\b", r"\bhealth services?\b", r"\bmedical services?\b",
    r"\brecruitment\b", r"\bpersonnel recruitment\b", r"\belectricity\b",
    r"\benergy supply\b", r"\butilities\b", r"\bsecurity guard\b",
    r"\bguarding\b", r"\bshipbuilding\b",

    # Physical platform / industrial service categories. These are hard exclusions
    # because Category describes the actual subject of the work more reliably than
    # generic words such as support, assurance, sustainment or management.
    r"\baircraft maintenance and repair services?\b",
    r"\bmarine craft maintenance and repair services?\b",
    r"\bvehicle maintenance and repair services?\b",
    r"\bweapon(?:s)? maintenance\b",
    r"\bmanufacturing technologies\b",
    r"\bhuman resources services?\b",
]

CATEGORY_RULES: dict[str, list[tuple[str, float, str]]] = {
    "Data, AI & Automation": [
        (r"\bdata management\b", 16, "data management category"),
        (r"\bdatabase\b", 15, "database category"),
        (r"\banalytics\b", 16, "analytics category"),
        (r"\bbusiness intelligence\b", 17, "business intelligence category"),
        (r"\bstatistical services?\b", 14, "statistical services category"),
        (r"\bdata services?\b", 15, "data services category"),
        (r"\bartificial intelligence\b", 18, "AI category"),
    ],
    "Cloud Infrastructure & Cyber": [
        (r"\bcloud computing\b", 18, "cloud computing category"),
        (r"\bhosting\b", 16, "hosting category"),
        (r"\bnetwork services?\b", 15, "network services category"),
        (r"\bcyber security\b", 18, "cyber security category"),
        (r"\binformation security\b", 17, "information security category"),
        (r"\btelecommunications services?\b", 14, "telecommunications services category"),
    ],
    "SI & Engineering": [
        (r"^software$", 14, "software category"),
        (r"\bsoftware or hardware engineering\b", 15, "software/hardware engineering category"),
        (r"\bapplication implementation services?\b", 12, "application implementation category"),
        (r"\bcomputer services?\b", 8, "computer services category - generic"),
        (r"\bsoftware development\b", 17, "software development category"),
        (r"\bapplication development\b", 17, "application development category"),
        (r"\bsystems? integration\b", 17, "systems integration category"),
        (r"\benterprise software\b", 16, "enterprise software category"),
        (r"\binformation technology services?\b", 10, "IT services category"),
        (r"\bprofessional engineering services?\b", 7, "engineering category - generic"),
    ],
    "Strategy, Transformation & Advisory": [
        (r"\bstrategic planning consultation services?\b", 17, "strategic planning consultation category"),
        (r"\binformation technology consultation services?\b", 8, "IT consultation category - generic"),
        (r"\bproject administration or planning\b", 14, "project administration / planning category"),
        (r"\bfeasibility studies? or screening of project ideas\b", 17, "feasibility study category"),
        (r"\bmanagement advisory\b", 17, "management advisory category"),
        (r"\bproject management\b", 14, "project management category"),
        (r"\bconsulting services?\b", 11, "consulting category"),
    ],
    "Managed Services & Operations": [
        (r"\bsystem administrators?\b", 17, "system administration category"),
        (r"\bsoftware maintenance and support\b", 17, "software maintenance/support category"),
        (r"\blearning management\b", 15, "learning management category"),
        (r"\bcustomer relationship management\b", 16, "CRM category"),
        (r"\bservice management\b", 15, "service management category"),
        (r"\bworkflow management\b", 15, "workflow management category"),
    ],
}


# Category-context classifier. This is deliberately separate from Service Offering
# scoring: first decide what kind of procurement Category describes, then decide
# whether Description/Supplier evidence is allowed to refine the outcome.
NEUTRAL_DEFENCE_CATEGORY_PATTERNS = [
    r"\bmilitary services and national defence\b",
    r"\bdefence services\b",
    r"\bnational defence\b",
]


def classify_category_context(cat_type: object, category_value: object) -> tuple[str, str, str]:
    cat_type_l = clean(cat_type)
    category_l = clean(category_value)

    # Broad Defence umbrella categories are intentionally neutral. They tell us
    # who/what environment the work relates to, not whether the underlying work
    # is digital, advisory, engineering, or physical materiel.
    neutral_match = any_pattern(category_l, NEUTRAL_DEFENCE_CATEGORY_PATTERNS)
    if neutral_match:
        return (
            "Neutral Defence umbrella",
            neutral_match,
            "Broad Defence Category is non-dispositive; inspect Description and supplier context for the actual work subject.",
        )

    if cat_type_l == "goods":
        physical_match = any_pattern(category_l, HARD_NON_ADDRESSABLE_GOODS_CATEGORY_PATTERNS)
        if physical_match:
            return (
                "Specific physical/materiel Goods",
                physical_match,
                "Specific Goods Category identifies physical/platform/materiel procurement and takes precedence over generic service wording.",
            )
        digital_match = any_pattern(category_l, DIGITAL_GOODS_EXCEPTIONS)
        if digital_match:
            return (
                "Explicit digital Goods",
                digital_match,
                "Goods Category explicitly describes software/licence/cloud/digital procurement and may proceed to Service Offering classification.",
            )
        return (
            "Ambiguous Goods",
            "",
            "Goods Category is neither clearly physical nor explicitly digital; require strong Description evidence before inclusion.",
        )

    nonaddr_match = any_pattern(category_l, NON_ADDRESSABLE_SERVICE_CATEGORY_PATTERNS)
    if nonaddr_match:
        return (
            "Specific non-addressable service Category",
            nonaddr_match,
            "Category explicitly describes a non-addressable physical/business service subject.",
        )

    # Determine whether Category itself contains a recognised Service Offering signal.
    category_scores, _ = collect_scores(category_l, CATEGORY_RULES, 1.0)
    if max(category_scores.values(), default=0) > 0:
        top = max(category_scores, key=category_scores.get)
        return (
            "Recognised addressable service Category",
            top,
            f"Category contains recognised addressable evidence for {top}; Description may refine or override when more specific.",
        )

    return (
        "Neutral / ambiguous service Category",
        "",
        "Category does not establish addressability by itself; inspect Description and supplier context.",
    )

DESCRIPTION_RULES: dict[str, list[tuple[str, float, str]]] = {
    "Data, AI & Automation": [
        (r"\bdata (?:platform|integration|migration|engineering|management|governance|services?|warehouse|lake|quality|remediation)\b", 14, "explicit data work"),
        (r"\bdata remediation\b", 16, "data remediation"),
        (r"\b(?:analytics|business intelligence|power bi|tableau|databricks|snowflake)\b", 14, "analytics / BI"),
        (r"\b(?:artificial intelligence|machine learning|generative ai|genai|automation|rpa)\b", 15, "AI / automation"),
    ],
    "Cloud Infrastructure & Cyber": [
        (r"\bcloud services?\b", 18, "explicit cloud services"),
        (r"\btelecommunications services?\b", 16, "explicit telecommunications services"),
        (r"\b(?:infrastructure hosting services?|hosting services?)\b", 16, "infrastructure hosting"),
        (r"\b(?:security accreditation|security grading|certification and accreditation|certification, accreditation)\b", 15, "security accreditation / grading"),
        (r"\b(?:aws|amazon web services|azure|gcp|cloud migration|cloud hosting|cloud infrastructure)\b", 15, "cloud platform / infrastructure"),
        (r"\b(?:cybersecurity|cyber security|security operations|siem|soc|zero trust|identity and access management|irap)\b", 15, "cybersecurity"),
        (r"\bict security (?:uplift|upgrade|services?|support)\b", 17, "ICT security uplift / support"),
        (r"\b(?:network architecture|network engineering|network implementation|network infrastructure|telecommunications network|network design services?)\b", 12, "network infrastructure"),
        (r"\b(?:data centre services?|data center services?|data backup storage services?|ict hosting platform|storage replacement and implementation)\b", 16, "data-centre / hosting infrastructure"),
    ],
    "SI & Engineering": [
        (r"^(?:ict services?|information technology services?|information and communications? technology services?|digital technology services?)\.?$", 16, "explicit ICT / technology services"),
        (r"^ict capability$", 15, "explicit ICT capability"),
        (r"\b(?:integration and development services?|system development|systems development|software developer services?|software engineers?)\b", 15, "system/software development"),
        (r"\b(?:open system architecture|open systems architecture)\b", 14, "open systems architecture"),
        (r"\b(?:simulation capability|core simulation capability|live virtual constructive)\b", 14, "simulation capability"),
        (r"\b(?:pilot training system)\b", 13, "training system"),
        (r"\b(?:systems? integration(?: services?)?|systems? integrator services?|platform integration|application integration|integration services?)\b", 18, "systems / platform integration"),
        (r"\b(?:software development|application development|software engineering|devops|devsecops)\b", 14, "software engineering"),
        (r"\b(?:systems? engineering|mbse|digital engineering|requirements engineering)\b", 13, "systems engineering"),
        (r"\b(?:sap|s/4hana|oracle erp|erp|servicenow|salesforce)\b", 13, "enterprise platform"),
        (r"\b(?:solution architecture|enterprise architecture|technical architecture|ict architecture)\b", 12, "architecture"),
        (r"\b(?:mission systems?|command and control|c4isr|battle management|simulation system)\b", 13, "mission / simulation systems"),
    ],
    "Strategy, Transformation & Advisory": [
        (r"\b(?:program management|project administration services?|business analysis services?|requirements workshop|scoping study|project definition study|solutions study|definition study)\b", 13, "program / analysis / study"),
        (r"\b(?:strategy|strategic advisory|advisory services?|business advisory)\b", 12, "strategy / advisory"),
        (r"\b(?:business case|feasibility study|options analysis|concept assessment|independent review|assurance)\b", 13, "assessment / assurance"),
        (r"\b(?:business transformation|operating model|organisational design|process improvement|change management)\b", 13, "transformation / change"),
        (r"\b(?:training services?|course development|training delivery|coaching|mentoring)\b", 13, "training / learning"),
        (r"\b(?:pmo|project management office|program management office|portfolio management|project support services?|project management services?|project management support(?: services?)?)\b", 12, "PMO / project support"),
    ],
    "Managed Services & Operations": [
        (r"\b(?:applications? managed services? partner arrangement|applications? managed services?|managed services?)\b", 20, "explicit managed services"),
        (r"\b(?:end user support services?|ict service desk|computer support services?|hardware support services?)\b", 16, "end-user / service-desk support"),
        (r"\b(?:ict|information technology|it)\s+(?:management and )?support services?\b", 18, "ICT management / support services"),
        (r"^ict support$", 17, "explicit ICT support"),
        (r"^ict personnel services$", 14, "ICT personnel services"),
        (r"\bapplication sustainment support\b", 17, "application sustainment support"),
        (r"^software maintenance$", 15, "software maintenance"),
        (r"\b(?:ict|information technology|it)\s+(?:operations?|operational support|managed support)\b", 17, "ICT operations / managed support"),
        (r"\b(?:systems? operations? and sustainment|systems? sustainment services?|systems? administration services?|systems? operations?|ict services for non-integrated capabilities|centralised processing services?|centralized processing services?|central processing services?)\b", 15, "systems operations / sustainment"),
        (r"\b(?:software sustainment and support|software maintenance and support)\b", 15, "software sustainment / support"),
        (r"\b(?:managed services?|application managed services?|service desk|help desk)\b", 12, "managed service"),
        (r"\b(?:application support|system support|software support|application maintenance|system maintenance)\b", 11, "application/system support"),
        (r"\b(?:hris|hcm|workday|successfactors|learning management system|lms|crm|salesforce|servicenow|itsm)\b.*\b(?:support|maintenance|operations?|uplift|upgrade|managed)\b", 16, "enterprise platform run/uplift"),
    ],
}

SUPPLIER_HINTS = [
    (r"\b(?:amazon web services|aws)\b", "Cloud Infrastructure & Cyber", 4, "AWS supplier"),
    (r"\b(?:microsoft)\b", "Cloud Infrastructure & Cyber", 2, "Microsoft supplier"),
    (r"\b(?:databricks|snowflake|tableau|informatica|alteryx|palantir)\b", "Data, AI & Automation", 4, "data-platform supplier"),
    (r"\b(?:sap)\b", "SI & Engineering", 3, "SAP supplier"),
    (r"\b(?:servicenow|salesforce|workday)\b", "Managed Services & Operations", 2, "enterprise-platform supplier"),
]


def any_pattern(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return m.group(0)
    return None


def collect_scores(text: str, rules: dict[str, list[tuple[str, float, str]]], multiplier: float) -> tuple[dict[str, float], dict[str, list[str]]]:
    scores = {o: 0.0 for o in FINAL_SERVICE_OFFERINGS}
    evidence = {o: [] for o in FINAL_SERVICE_OFFERINGS}
    for offering, patterns in rules.items():
        for pattern, base, label in patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                score = base * multiplier
                scores[offering] += score
                evidence[offering].append(f"{label}: {m.group(0)} (+{score:.1f})")
    return scores, evidence


def merge_score_dicts(*dicts: dict[str, float]) -> dict[str, float]:
    return {o: sum(float(d.get(o, 0.0)) for d in dicts) for o in FINAL_SERVICE_OFFERINGS}


def confidence_from_scores(winner_score: float, second_score: float, source_count: int) -> tuple[int, str]:
    margin = winner_score - second_score
    if winner_score <= 0:
        return 25, "Low"
    if winner_score >= 24 and margin >= 10 and source_count >= 2:
        return 97, "Very high"
    if winner_score >= 18 and margin >= 8:
        return 92, "High"
    if winner_score >= 13 and margin >= 5:
        return 84, "Medium"
    if winner_score >= 9 and margin >= 3:
        return 72, "Medium"
    return 50, "Low"


def classify_service_offering(row: pd.Series) -> dict[str, object]:
    cn = upper_cn(row.get(CN_ID, ""))
    cat_type = clean(row.get(CATEGORY_TYPE, ""))
    category = clean(row.get(CATEGORY, ""))
    description = clean(row.get(DESCRIPTION, ""))
    supplier_raw = row.get("supplier_group", row.get(SUPPLIER_NAME, ""))
    supplier = clean(supplier_raw)
    supplier_source_text = clean(row.get(SUPPLIER_NAME, supplier_raw))
    division_text = clean(row.get(RAW_DIVISION, ""))
    branch_text = clean(row.get(RAW_BRANCH, ""))
    organisation_text = f"{division_text} {branch_text}".strip()
    cohort = supplier_cohort(supplier_raw)
    category_context, category_context_signal, category_context_reason = classify_category_context(cat_type, category)

    # Authoritative known contract-level override.
    if cn in AUTHORITATIVE_SERVICE_OFFERING_OVERRIDES:
        offering = AUTHORITATIVE_SERVICE_OFFERING_OVERRIDES[cn]
        return mapping_result(
            True, offering, 100, "Authoritative", "Authoritative CN override", "",
            100.0, f"Confirmed project-level override for {cn}", {}, {}, {},
            "Authoritative known project classification."
        )

    # Authoritative non-addressable CN exclusions.
    if cn in MANUAL_NON_ADDRESSABLE_CN_IDS:
        return non_addressable_result(
            100, "Authoritative CN exclusion",
            "Contract is explicitly retained in total Defence procurement but excluded from Accenture TAM."
        )

    # ------------------------------------------------------------------
    # GATE 1: CATEGORY TYPE = GOODS
    # ------------------------------------------------------------------
    if cat_type == "goods":
        # Stage 1 - Category is the primary evidence for Goods. A clearly physical
        # or military procurement is locked out of the TAM. Even an apparently
        # digital phrase in Description cannot rescue a missile, launcher, platform,
        # component, equipment or accessories procurement.
        hard_physical_match = category_context_signal if category_context == "Specific physical/materiel Goods" else ""
        if hard_physical_match:
            strong_desc = any_pattern(description, STRONG_ADDRESSABLE_GOODS_DESCRIPTION_PATTERNS)
            if strong_desc and cohort in {"Consulting / professional services", "Technology / managed services"}:
                return non_addressable_result(
                    70,
                    "Goods supplier/category conflict - manual review",
                    f"Category Type is Goods and Category contains specific physical/military signal '{hard_physical_match}', so the contract remains outside TAM. "
                    f"However, Description contains strong digital/service evidence '{strong_desc}' and supplier cohort is Consulting / professional services or Technology / managed services. "
                    "This contradiction is flagged for manual review rather than automatically included."
                )
            extra = (
                f" Description also contains digital-looking evidence '{strong_desc}', but the physical/military Category takes precedence."
                if strong_desc else ""
            )
            prime_note = (
                " Supplier is a Defence prime/platform OEM, which is consistent with the physical Category."
                if cohort == "Defence prime / platform OEM" else ""
            )
            return non_addressable_result(
                100,
                "Goods hard physical Category gate",
                f"Category Type is Goods and Category contains hard physical/military procurement signal '{hard_physical_match}'." + extra + prime_note
            )

        # Stage 2 - if the Category itself says software/licence/cloud/SaaS, allow
        # the row to continue to the normal Category -> Description -> Supplier
        # classifier. These are the only Goods categories presumed addressable.
        digital_exception = category_context_signal if category_context == "Explicit digital Goods" else ""
        if digital_exception:
            pass
        else:
            # Stage 3 - ambiguous Goods may only be rescued by strong, explicit
            # digital subject matter. Generic terms such as support, system,
            # engineering, project management or training do not qualify.
            strong_desc = any_pattern(description, STRONG_ADDRESSABLE_GOODS_DESCRIPTION_PATTERNS)
            if not strong_desc:
                return non_addressable_result(
                    95,
                    "Goods ambiguous Category gate",
                    "Category Type is Goods, Category is not explicitly digital, and Description contains no strong addressable digital subject. "
                    "Generic service wording is insufficient to override a Goods classification."
                )
            if cohort == "Defence prime / platform OEM":
                return non_addressable_result(
                    70,
                    "Goods ambiguous Category + Defence prime - manual review",
                    f"Category Type is Goods and Category is ambiguous. Description contains strong addressable evidence '{strong_desc}', "
                    "but the supplier is a Defence prime/platform OEM. The contract is kept outside TAM pending manual review because the digital wording may describe embedded platform work."
                )
            # Professional/technology-services suppliers materially strengthen a strong
            # digital Description when the Goods Category itself is ambiguous. Other
            # suppliers can still proceed, but receive no cohort-based confidence benefit.

    # ------------------------------------------------------------------
    # GATE 1B: Consulting/professional-services work miscoded to generic HR.
    # Human-resources procurement is normally outside TAM, but a recognised
    # consulting supplier delivering professional/project/advisory services is
    # treated as a consulting engagement unless a physical exclusion above fired.
    # This is a general evidence rule, not a CN-specific inclusion.
    # ------------------------------------------------------------------
    consulting_hr_service = (
        cat_type == "services"
        and cohort == "Consulting / professional services"
        and bool(any_pattern(category, [r"^human resources services?$"]))
        and bool(any_pattern(description, [
            r"^professional services?$",
            r"^project management support(?: services?)?$",
            r"^project management services?$",
            r"\b(?:advisory|consulting|transformation|change management|program management)\b",
        ]))
    )
    if consulting_hr_service:
        return mapping_result(
            True, "Strategy, Transformation & Advisory", 92, "High",
            "Consulting services miscoded to HR Category", "", 18.0,
            "Recognised consulting supplier is delivering professional/project/advisory services; the generic Human resources services Category does not describe the actual engagement subject.",
            {}, {}, {},
            "Supplier cohort + Description establish a consulting engagement; hard physical exclusions still take precedence."
        )

    # ------------------------------------------------------------------
    # GATE 2: CATEGORY says clearly non-addressable service
    # ------------------------------------------------------------------
    nonaddr_service = category_context_signal if category_context == "Specific non-addressable service Category" else ""
    if nonaddr_service:
        return non_addressable_result(
            98,
            "Non-addressable Category gate",
            f"Category contains non-addressable procurement signal '{nonaddr_service}'."
        )


    # ------------------------------------------------------------------
    # GATE 2A1: Generic physical data-centre facility delivery.
    #
    # A plain "Data Centre Services" / "Data Centre Facility Services"
    # engagement from a specialist facility/colocation operator is treated as
    # physical infrastructure/facility delivery, not as addressable cloud or SI.
    # More explicit digital work (hosting, cloud, network, support, project
    # services, specialists, strategy, etc.) does not hit this gate.
    # ------------------------------------------------------------------
    data_centre_facility_operator = any_pattern(
        f"{supplier_source_text} {supplier}",
        DATA_CENTRE_FACILITY_OPERATOR_PATTERNS,
    )
    generic_data_centre_facility = any_pattern(
        description,
        GENERIC_DATA_CENTRE_FACILITY_DESCRIPTION_PATTERNS,
    )
    if data_centre_facility_operator and generic_data_centre_facility:
        return non_addressable_result(
            98,
            "Physical data-centre facility delivery gate",
            "Generic data-centre facility/service delivery by a specialist data-centre operator is physical infrastructure/facility procurement rather than an Accenture-addressable digital service."
        )

    # ------------------------------------------------------------------
    # GATE 2A2: Physical construction / facilities delivery.
    #
    # This is rule-based rather than CN-specific. It catches physical estate,
    # redevelopment, civil works and construction delivery even when AusTender
    # uses generic project-management / project-administration categories.
    # A construction/facilities supplier is only used to resolve ambiguous
    # project/support wording; supplier identity by itself is never enough.
    # Explicit digital, systems, data, transformation or advisory subject matter
    # can rescue a facilities-related contract.
    # ------------------------------------------------------------------
    physical_facilities_desc = any_pattern(description, PHYSICAL_FACILITIES_DESCRIPTION_PATTERNS)
    physical_supplier = any_pattern(
        f"{supplier_source_text} {supplier}",
        PHYSICAL_FACILITIES_CONSTRUCTION_SUPPLIER_PATTERNS,
    )
    generic_project_delivery = any_pattern(description, GENERIC_PROJECT_DELIVERY_PATTERNS)
    infrastructure_org = any_pattern(organisation_text, INFRASTRUCTURE_ORG_PATTERNS)
    # Rescue must be explicit in the Description. A generic or miscoded Category
    # (for example Data services on a base-services tender) must not override
    # clear physical facilities/construction subject matter.
    facilities_rescue = any_pattern(
        description,
        FACILITIES_DIGITAL_OR_ADVISORY_RESCUE_PATTERNS,
    )
    generic_project_category = any_pattern(
        category,
        [
            r"^project management$",
            r"^project administration or planning$",
            r"^management support services$",
            r"^management advisory services$",
        ],
    )

    physical_facilities_context = bool(physical_facilities_desc) or bool(
        physical_supplier
        and generic_project_delivery
        and (generic_project_category or infrastructure_org)
    )

    if physical_facilities_context and not facilities_rescue:
        evidence_bits = []
        if physical_facilities_desc:
            evidence_bits.append(f"Description='{physical_facilities_desc}'")
        if physical_supplier:
            evidence_bits.append(f"physical delivery supplier='{physical_supplier}'")
        if infrastructure_org:
            evidence_bits.append(f"infrastructure/estate organisation='{infrastructure_org}'")
        if generic_project_category:
            evidence_bits.append(f"generic project Category='{generic_project_category}'")
        return non_addressable_result(
            98,
            "Physical construction / facilities delivery gate",
            "Physical construction, redevelopment, facilities or estate delivery is outside Accenture TAM. "
            + "; ".join(evidence_bits)
            + ". No explicit digital, data, systems, transformation or advisory subject was found.",
        )

    # ------------------------------------------------------------------
    # GATE 2B: Description reveals an obviously non-addressable business service.
    # Generic administration categories must not pull these into Strategy.
    # ------------------------------------------------------------------
    obvious_nonaddr_desc = any_pattern(
        description,
        [
            r"^environmental services?$",
            r"^(?:ict equipment|hardware refresh|hardware services|computer equipment and services|ict equipment and services|additional hardware and services|transformation hardware refresh - compute)$",
            r"\bsecurity vetting services?\b",
            r"\binsurance services?\b",
            r"\bsuperannuation(?: administration| services?)?\b",
            r"\binsurance and retirement services?\b",
            r"\baccommodation program management services?\b",
            r"^(?:travel|travel costs?|travel expenses?|contractor travel|travel services?)$",
            r"^(?:temporary )?accommodation(?: services?)?$",
            r"^(?:hotel|lodging|taxi|cab|airfare|flight booking|car hire|vehicle hire|rental car)(?: services?)?$",
        ],
    )
    if obvious_nonaddr_desc:
        return non_addressable_result(
            98,
            "Non-addressable Description gate",
            f"Description contains non-addressable business-service signal '{obvious_nonaddr_desc}'."
        )

    # ------------------------------------------------------------------
    # GATE 2B2: Training / academic delivery is outside the TAM by default.
    # Training is only addressable when it is clearly subordinate to an explicit
    # enterprise/digital implementation or transformation subject.
    # ------------------------------------------------------------------
    training_category = any_pattern(category, [
        r"\beducation and training services?\b", r"\btraining services?\b",
        r"\bacademic services?\b", r"\bvocational education\b",
    ])
    training_desc = any_pattern(description, [
        r"\btraining services?\b", r"\bacademic (?:studies|services?)\b",
        r"\bwar college\b", r"\bnaval shipbuilding college\b",
        r"\baircrew training\b", r"\bplatform and systems training\b",
        r"\btraining system\b", r"\bcourse delivery\b", r"\btaf[e]?\b",
    ])
    training_rescue = any_pattern(description, [
        r"\b(?:sap|s/4hana|oracle erp|servicenow|salesforce|workday)\b.*\b(?:implementation|transformation|deployment|migration)\b",
        r"\b(?:cloud|data platform|software|application|ict)\b.*\b(?:implementation|transformation|deployment|migration)\b",
    ])
    if (training_category or training_desc) and not training_rescue:
        return non_addressable_result(
            98, "Training / education exclusion gate",
            f"Training/education is outside the addressable TAM by default (signal: '{training_category or training_desc}'). No explicit broader enterprise/digital implementation subject was found."
        )

    # ------------------------------------------------------------------
    # GATE 2B2A: Narrow maritime physical-engineering exclusions.
    # Structural assessment/design assurance is treated as physical platform
    # engineering, not consulting/SI. Direct project-cost descriptions tied to
    # an explicitly named warship/platform are also outside TAM. Keep broader
    # maritime support, engineering and integration wording unchanged because
    # those contracts can be genuinely addressable or too ambiguous to exclude.
    # ------------------------------------------------------------------
    structural_platform_assurance = any_pattern(description, [
        r"\bstructural (?:assurance )?assessment services?\b",
        r"\bstructural design assurance\b",
        r"\bstructural assurance(?: assessment)?(?: services?)?\b",
    ])
    platform_direct_project_cost = any_pattern(description, [
        r"\b(?:air warfare destroyer|warship|naval ship|submarine|frigate|destroyer)\b.*\bdirect project costs?\b",
        r"\bdirect project costs?\b.*\b(?:air warfare destroyer|warship|naval ship|submarine|frigate|destroyer)\b",
    ])
    if structural_platform_assurance:
        return non_addressable_result(
            98,
            "Physical structural/platform assurance gate",
            f"Description identifies physical structural assessment/design assurance ('{structural_platform_assurance}'), which is outside Accenture TAM."
        )
    if platform_direct_project_cost:
        return non_addressable_result(
            98,
            "Physical maritime platform direct-cost gate",
            f"Description identifies direct project costs tied to a named maritime platform ('{platform_direct_project_cost}'), which is outside Accenture TAM."
        )

    # ------------------------------------------------------------------
    # GATE 2B3: Technical categories can describe embedded platform/mission work.
    # A physical Defence platform/mission subject blocks generic software,
    # engineering, assurance or simulation wording unless the description itself
    # identifies a distinct enterprise/digital service subject.
    # ------------------------------------------------------------------
    platform_subject = any_pattern(description, [
        r"\b(?:combat helicopter|helicopter|aircraft|aircrew|avionics|aeronautical)\b",
        r"\b(?:warship|naval ship|shipbuilding|submarine|combat vehicle|armoured vehicle)\b",
        r"\b(?:guided weapons?|missiles?|rockets?|torpedoes?|munitions?|ammunition)\b",
        r"\b(?:battle management|combat system|fire control|radar|sonar|electronic warfare|electro magnetic battle management|electromagnetic battle management)\b",
        r"\b(?:mission system|terminal control system|training simulator|simulation capability|amphibious capability)\b",
    ])
    explicit_enterprise_digital = any_pattern(description, [
        r"\b(?:enterprise|corporate) (?:software|application|platform|system|data)\b",
        r"\b(?:cloud migration|cloud services?|saas|data platform|data analytics|business intelligence|power bi)\b",
        r"\b(?:erp|sap|s/4hana|oracle erp|servicenow|salesforce|workday)\b",
        r"\b(?:enterprise application|business application) (?:development|implementation|integration|support)\b",
    ])
    if platform_subject and not explicit_enterprise_digital:
        return non_addressable_result(
            96, "Physical Defence platform / mission-system subject gate",
            f"Description identifies a Defence platform/mission subject '{platform_subject}'. Generic software, engineering, assurance, simulation or support terminology is insufficient without an explicit enterprise/digital service subject."
        )

    # ------------------------------------------------------------------
    # GATE 2C: Consulting/professional-services cohort defaults to addressable
    # for Defence service contracts unless a hard physical/non-addressable gate
    # above has already fired. This prevents generic procurement labels such as
    # Management support services from excluding obvious consulting work.
    # Description still determines the best Service Offering where possible.
    # ------------------------------------------------------------------
    consulting_default = (cat_type == "services" and cohort == "Consulting / professional services")

    # ------------------------------------------------------------------
    # GATE 3: SERVICES can still describe physical platform / weapons work.
    # Category remains primary, but unmistakably physical production/maintenance
    # wording prevents generic advisory/support terms from pulling the contract in.
    # ------------------------------------------------------------------
    if cat_type == "services":
        physical_service_subject = any_pattern(
            f"{category} {description}",
            [
                r"\b(?:guided weapons?|missiles?|torpedoes?|munitions?|ammunition)\b.*\b(?:production|manufactur|maintenance|repair|sustainment)\b",
                r"\b(?:aircraft|airframe|helicopter|ship|submarine|vehicle)\b.*\b(?:maintenance|repair|overhaul|production|manufactur)\b",
                r"\b(?:production|manufactur)\w*\b.*\b(?:guided weapons?|missiles?|torpedoes?|munitions?|aircraft|vehicles?)\b",
            ],
        )
        if physical_service_subject:
            return non_addressable_result(
                98,
                "Physical platform / weapons service gate",
                f"Service scope contains explicit physical platform, weapon production or maintenance evidence '{physical_service_subject}'."
            )

    # ------------------------------------------------------------------
    # GATE 3B: Generic management-support/professional-services descriptions.
    # When the procurement Category is neutral management support, generic
    # project/professional wording is only enough to establish addressability
    # when supplier or Defence organisation context is clearly technology-led.
    # Physical/platform/facilities gates above always take precedence.
    # ------------------------------------------------------------------
    technology_org_context = any_pattern(organisation_text, [
        r"\bddg\b", r"\bciog\b", r"\bictdd\b", r"\bict(?:rd|od)\b",
        r"\berp\b", r"\bc4 systems?\b", r"\bcyber warfare\b", r"\bsigint and cyber\b",
    ])
    generic_mgmt_support = bool(any_pattern(category, [r"^management support services?$"]))
    generic_project_professional_desc = any_pattern(description, [
        r"^project management services?$",
        r"^project management support(?: services?)?$",
        r"^project support(?: services?)?$",
        r"^professional services?$",
    ])
    if (
        cat_type == "services"
        and generic_mgmt_support
        and generic_project_professional_desc
        and (cohort in {"Technology / managed services", "Consulting / professional services"} or technology_org_context)
    ):
        return mapping_result(
            True, "Strategy, Transformation & Advisory", 84, "Medium",
            "Technology-context project/professional services", "", 13.0,
            f"Neutral Management support services Category is clarified by Description '{generic_project_professional_desc}' and technology supplier/organisation context.",
            {}, {}, {},
            "Technology context resolves otherwise-generic project/professional services as addressable Strategy/Advisory work."
        )

    # ------------------------------------------------------------------
    # GATE 3C: False-negative rescue for technology support/sustainment work.
    # Generic Management support / broad Defence categories are neutral, but
    # explicit digital subject matter or a technology supplier operating in a
    # clearly digital Defence organisation is enough to establish addressability.
    # The hard physical/platform/training/facilities gates above still win.
    # ------------------------------------------------------------------
    explicit_data_support = any_pattern(description, [
        r"\bdata support services?\b",
    ])
    explicit_cloud_support = any_pattern(description, [
        r"\bict cloud support services?\b",
        r"\bcloud support services?\b",
    ])
    explicit_security_support = any_pattern(description, [
        r"\bict security management support\b",
        r"\bcyber(?:security)? management support\b",
    ])
    explicit_application_sustainment = any_pattern(description, [
        r"\bapplication sustainment(?: and support)? services?\b",
        r"\bapplication sustainment support\b",
    ])
    generic_tech_support_desc = any_pattern(description, [
        r"^support services?$",
        r"^sustainment services?$",
        r"^sustainment and support services?$",
        r"^maintenance and sustainment services?$",
        r"^management services?$",
        r"^management support services?$",
    ])
    digital_project_delivery_desc = any_pattern(description, [
        r"^project delivery team$",
        r"^project management$",
    ])
    neutral_service_category = generic_mgmt_support or bool(any_pattern(category, [
        r"^military services and national defence$",
    ]))
    technology_supplier_context = cohort == "Technology / managed services"

    if cat_type == "services" and neutral_service_category:
        if explicit_data_support:
            return mapping_result(
                True, "Data, AI & Automation", 92, "High",
                "Explicit data support service", "", 18.0,
                "Description explicitly identifies Data Support Services; the neutral procurement Category does not override the service subject.",
                {}, {}, {},
                "Explicit data-service subject establishes addressability and Data, AI & Automation mapping."
            )
        if explicit_cloud_support:
            return mapping_result(
                True, "Cloud Infrastructure & Cyber", 92, "High",
                "Explicit cloud support service", "", 18.0,
                "Description explicitly identifies ICT/Cloud Support Services.",
                {}, {}, {},
                "Explicit cloud-service subject establishes addressability and Cloud Infrastructure & Cyber mapping."
            )
        if explicit_security_support:
            return mapping_result(
                True, "Cloud Infrastructure & Cyber", 92, "High",
                "Explicit ICT security support", "", 18.0,
                "Description explicitly identifies ICT Security Management Support.",
                {}, {}, {},
                "Explicit ICT/cyber-security subject establishes addressability and Cloud Infrastructure & Cyber mapping."
            )
        if explicit_application_sustainment:
            return mapping_result(
                True, "Managed Services & Operations", 92, "High",
                "Explicit application sustainment", "", 18.0,
                "Description explicitly identifies application sustainment/support work.",
                {}, {}, {},
                "Explicit application sustainment establishes addressability and Managed Services & Operations mapping."
            )
        if generic_tech_support_desc and technology_supplier_context and technology_org_context:
            return mapping_result(
                True, "Managed Services & Operations", 82, "Medium",
                "Technology supplier + digital organisation support", "", 12.0,
                "Generic support/sustainment wording is resolved by the combination of a recognised technology/managed-services supplier and a clearly digital Defence organisation.",
                {}, {}, {},
                "Supplier and organisation context jointly establish addressability for otherwise-generic technology support/sustainment work."
            )
        if digital_project_delivery_desc and technology_org_context:
            return mapping_result(
                True, "Strategy, Transformation & Advisory", 82, "Medium",
                "Digital-organisation project delivery", "", 12.0,
                "Generic project delivery/management wording occurs within a clearly digital Defence organisation, supporting addressable project/transformation work.",
                {}, {}, {},
                "Digital organisation context resolves otherwise-generic project delivery as addressable Strategy/Transformation work."
            )

    # ------------------------------------------------------------------
    # SCORING: Category first, Description second, Supplier weakly third.
    # ------------------------------------------------------------------
    category_scores, category_evidence = collect_scores(category, CATEGORY_RULES, 1.50)
    description_scores, description_evidence = collect_scores(description, DESCRIPTION_RULES, 1.00)
    supplier_scores = {o: 0.0 for o in FINAL_SERVICE_OFFERINGS}
    supplier_evidence = {o: [] for o in FINAL_SERVICE_OFFERINGS}
    for pattern, offering, score, label in SUPPLIER_HINTS:
        m = re.search(pattern, supplier, flags=re.IGNORECASE)
        if m:
            supplier_scores[offering] += score
            supplier_evidence[offering].append(f"{label}: {m.group(0)} (+{score:.1f})")

    total = merge_score_dicts(category_scores, description_scores, supplier_scores)

    # Specific Description evidence should beat generic procurement Categories.
    # This avoids generic labels such as Computer services / Management advisory /
    # Application implementation forcing everything into SI or Strategy when the
    # Description clearly says hosting, support, data, project support, etc.
    generic_category_for_description_precedence = any_pattern(category, [
        r"^computer services?$",
        r"^information technology consultation services?$",
        r"^application implementation services?$",
        r"^management advisory services?$",
        r"^management support services?$",
        r"^project management$",
        r"^project administration or planning$",
    ])
    desc_top_pre = max(description_scores, key=description_scores.get)
    desc_top_score_pre = description_scores[desc_top_pre]
    cat_top_score_pre = max(category_scores.values())
    description_precedence_applied = False
    if generic_category_for_description_precedence and desc_top_score_pre >= 10 and cat_top_score_pre > 0:
        # Add only enough weight to make the specific Description determinative.
        total[desc_top_pre] += cat_top_score_pre + 0.5
        description_precedence_applied = True

    ranked = sorted(total.items(), key=lambda x: x[1], reverse=True)
    winner, winner_score = ranked[0]
    second_choice, second_score = ranked[1]

    # Generic/no evidence normally remains outside TAM. Consulting firms are the
    # deliberate exception: for a Defence Services contract that survived all hard
    # exclusions, default to Strategy rather than silently dropping the contract.
    if winner_score <= 0:
        if consulting_default:
            return mapping_result(
                True, "Strategy, Transformation & Advisory", 88, "High",
                "Consulting supplier default", "", 12.0,
                "Recognised consulting/professional-services supplier on a Defence Services contract; no hard physical/non-addressable evidence was found.",
                {}, {}, {},
                "Consulting cohort defaults to addressable unless contradicted by a hard exclusion; generic or neutral procurement wording maps provisionally to Strategy, Transformation & Advisory."
            )
        return non_addressable_result(
            25,
            "No recognised addressable evidence",
            "Neither Category nor Description contains a recognised addressable Service Offering signal; supplier context was also insufficient."
        )

    # Generic engineering category alone is not sufficient to enter TAM.
    if (
        winner == "SI & Engineering"
        and category_scores[winner] > 0
        and description_scores[winner] == 0
        and "professional engineering" in category
    ):
        if consulting_default:
            return mapping_result(
                True, "Strategy, Transformation & Advisory", 78, "Medium",
                "Consulting supplier + generic professional engineering", "SI & Engineering", 6.0,
                "Generic professional engineering Category is ambiguous, but the recognised consulting supplier supports inclusion in the TAM.",
                category_scores, description_scores, supplier_scores,
                "Consulting cohort prevents false exclusion; mapped provisionally to Strategy pending more specific Description evidence."
            )
        return non_addressable_result(
            55,
            "Generic engineering Category only",
            "Professional engineering Category was not supported by explicit digital, systems, software, integration or architecture wording in Description."
        )

    sources = sum([
        category_scores[winner] > 0,
        description_scores[winner] > 0,
        supplier_scores[winner] > 0,
    ])
    confidence, band = confidence_from_scores(winner_score, second_score, sources)
    margin = winner_score - second_score

    cat_top = max(category_scores, key=category_scores.get)
    desc_top = max(description_scores, key=description_scores.get)
    category_signal = category_scores[cat_top]
    description_signal = description_scores[desc_top]

    if description_precedence_applied:
        driver = "Specific Description overrides generic Category"
        low_reason = f"Description provides a more specific Service Offering signal than generic Category '{category}'."
    elif category_signal > 0 and description_signal == 0:
        driver = "Category grouping"
        low_reason = "Classification is driven by Category evidence; Description contains no recognised supporting signal."
    elif category_signal == 0 and description_signal > 0:
        driver = "Description only"
        low_reason = "Classification is driven by Description because Category did not provide a recognised Service Offering signal."
    elif category_signal > 0 and description_signal > 0 and cat_top != desc_top:
        driver = "Category / Description disagreement"
        low_reason = f"Category points to {cat_top} while Description points to {desc_top}; competing evidence reduces confidence."
    elif margin < 5:
        driver = "Small first/second choice margin"
        low_reason = f"Selected Service Offering is only {margin:.1f} points ahead of {second_choice}."
    elif supplier_scores[winner] > 0 and category_scores[winner] == 0 and description_scores[winner] == 0:
        driver = "Supplier context only"
        low_reason = "Classification depends only on weak supplier context and should be manually reviewed."
    else:
        driver = "Category + Description agreement"
        low_reason = "Category and Description provide consistent evidence."

    reason_parts = []
    if category_evidence[winner]:
        reason_parts.append("Category: " + "; ".join(category_evidence[winner]))
    if description_evidence[winner]:
        reason_parts.append("Description: " + "; ".join(description_evidence[winner]))
    if supplier_evidence[winner]:
        reason_parts.append("Supplier: " + "; ".join(supplier_evidence[winner]))

    return mapping_result(
        True, winner, confidence, band, "Category -> Description -> Supplier scoring",
        second_choice, margin, " | ".join(reason_parts),
        category_scores, description_scores, supplier_scores,
        low_reason,
        confidence_driver=driver,
        category_evidence=category_evidence,
        description_evidence=description_evidence,
        supplier_evidence=supplier_evidence,
    )


def non_addressable_result(confidence: int, method: str, reason: str) -> dict[str, object]:
    return {
        "addressability": "Not Addressable",
        "is_addressable": False,
        "capability": "Non-addressable",
        "executive_capability": "Non-addressable",
        "detailed_capability": method,
        "service_offering": "Non-addressable",
        "service_offering_method": method,
        "service_offering_confidence": confidence,
        "service_offering_confidence_band": "Confirmed exclusion" if confidence >= 95 else "Review exclusion",
        "service_offering_second_choice": "",
        "service_offering_margin": 0.0,
        "service_offering_evidence": reason,
        "service_offering_score_breakdown": "{}",
        "category_score_breakdown": "{}",
        "description_score_breakdown": "{}",
        "supplier_score_breakdown": "{}",
        "category_evidence": "",
        "description_evidence": "",
        "supplier_evidence": "",
        "confidence_driver": method,
        "low_confidence_reason": reason,
        "confidence": confidence,
        "capability_confidence": confidence,
        "capability_confidence_band": "Confirmed exclusion" if confidence >= 95 else "Review exclusion",
        "capability_review_status": "Confirmed" if confidence >= 95 else "Review suggested",
        "classification_reason": reason,
    }


def mapping_result(
    addressable: bool,
    offering: str,
    confidence: int,
    band: str,
    method: str,
    second_choice: str,
    margin: float,
    evidence: str,
    category_scores: dict[str, float],
    description_scores: dict[str, float],
    supplier_scores: dict[str, float],
    low_reason: str,
    *,
    confidence_driver: str = "Authoritative",
    category_evidence: dict[str, list[str]] | None = None,
    description_evidence: dict[str, list[str]] | None = None,
    supplier_evidence: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    total = merge_score_dicts(category_scores, description_scores, supplier_scores) if category_scores else {offering: 100.0}
    cat_ev = "; ".join((category_evidence or {}).get(offering, []))
    desc_ev = "; ".join((description_evidence or {}).get(offering, []))
    supp_ev = "; ".join((supplier_evidence or {}).get(offering, []))
    return {
        "addressability": "Addressable" if addressable else "Not Addressable",
        "is_addressable": addressable,
        "capability": offering,
        "executive_capability": offering,
        "detailed_capability": offering,
        "service_offering": offering,
        "service_offering_method": method,
        "service_offering_confidence": confidence,
        "service_offering_confidence_band": band,
        "service_offering_second_choice": second_choice,
        "service_offering_margin": round(float(margin), 2),
        "service_offering_evidence": evidence,
        "service_offering_score_breakdown": json.dumps({k: round(v, 2) for k, v in total.items()}, sort_keys=True),
        "category_score_breakdown": json.dumps({k: round(v, 2) for k, v in category_scores.items()}, sort_keys=True),
        "description_score_breakdown": json.dumps({k: round(v, 2) for k, v in description_scores.items()}, sort_keys=True),
        "supplier_score_breakdown": json.dumps({k: round(v, 2) for k, v in supplier_scores.items()}, sort_keys=True),
        "category_evidence": cat_ev,
        "description_evidence": desc_ev,
        "supplier_evidence": supp_ev,
        "confidence_driver": confidence_driver,
        "low_confidence_reason": low_reason,
        "confidence": confidence,
        "capability_confidence": confidence,
        "capability_confidence_band": band,
        "capability_review_status": "High confidence" if confidence >= 85 else "Review suggested" if confidence >= 65 else "Manual review required",
        "classification_reason": evidence or low_reason,
    }


def apply_service_offering_mapping(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mapped = pd.DataFrame([classify_service_offering(row) for _, row in out.iterrows()], index=out.index)
    for col in mapped.columns:
        if col in out.columns:
            out = out.drop(columns=[col])
    combined = pd.concat([out, mapped], axis=1)
    combined["supplier_cohort"] = combined["supplier_group"].map(supplier_cohort)
    category_context_values = [classify_category_context(r.get(CATEGORY_TYPE, ""), r.get(CATEGORY, "")) for _, r in combined.iterrows()]
    combined["category_context"] = [x[0] for x in category_context_values]
    combined["category_context_signal"] = [x[1] for x in category_context_values]
    combined["category_context_reason"] = [x[2] for x in category_context_values]

    def cohort_effect(row: pd.Series) -> str:
        method = str(row.get("service_offering_method", ""))
        cohort = str(row.get("supplier_cohort", ""))
        cat_type = clean(row.get(CATEGORY_TYPE, ""))
        if "supplier/category conflict" in method.lower():
            return "Physical Category overrides, but professional/technology supplier conflict is flagged for manual review"
        if "defence prime" in method.lower():
            return "Defence-prime cohort increases scepticism for ambiguous Goods and keeps contract outside TAM pending review"
        if cat_type == "goods" and cohort in {"Consulting / professional services", "Technology / managed services"} and bool(row.get("is_addressable", False)):
            return "Consulting/technology-services cohort supports strong digital Description for an otherwise ambiguous Goods record"
        if cat_type == "goods" and cohort == "Defence prime / platform OEM":
            return "Defence-prime cohort supports a conservative interpretation of Goods as platform/materiel procurement"
        return "No cohort adjustment"

    combined["supplier_cohort_effect"] = combined.apply(cohort_effect, axis=1)
    return combined


def enforce_authoritative_contract_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Re-apply contract-level overrides after normal mapping as a final invariant.

    This protects known reviewed contracts from being changed by future scoring,
    supplier reconciliation, or default-retention logic.
    """
    out = df.copy()
    cn_series = out[CN_ID].fillna("").astype(str).str.strip().str.upper()
    for cn, offering in AUTHORITATIVE_SERVICE_OFFERING_OVERRIDES.items():
        mask = cn_series.eq(cn)
        if not mask.any():
            continue
        out.loc[mask, "is_addressable"] = True
        out.loc[mask, "addressability"] = "Addressable"
        out.loc[mask, "capability"] = offering
        out.loc[mask, "executive_capability"] = offering
        out.loc[mask, "detailed_capability"] = offering
        out.loc[mask, "service_offering"] = offering
        out.loc[mask, "service_offering_confidence"] = 100
        out.loc[mask, "service_offering_confidence_band"] = "Authoritative"
        out.loc[mask, "service_offering_method"] = "Authoritative CN override - final invariant"
        out.loc[mask, "confidence_driver"] = "Authoritative CN override"
        out.loc[mask, "low_confidence_reason"] = ""
        out.loc[mask, "service_offering_evidence"] = f"Confirmed project-level override for {cn}."
        out.loc[mask, "classification_reason"] = f"Confirmed project-level override for {cn}."
    return out


def retain_observed_accenture_wins(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee that observed Accenture awards cannot disappear from Accenture TAM.

    This restores a proven invariant from the original build. Supplier grouping is
    an identity operation; a recognised Accenture award must not lose addressability
    because a later generic classifier cannot interpret sparse public wording.
    """
    out = df.copy()
    out["accenture_win_override"] = False
    out["accenture_win_override_reason"] = ""

    is_acc = out.get("is_accenture", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    addr = bool_series(out["is_addressable"])
    restore = is_acc & ~addr
    if not restore.any():
        return out

    # Ask the normal classifier for the best available offering after temporarily
    # treating the supplier as consulting. Most rows already have usable evidence;
    # unresolved rows default to Strategy rather than vanishing from the TAM.
    for idx in out.index[restore]:
        row = out.loc[idx].copy()
        row["supplier_group"] = "Accenture"
        result = classify_service_offering(row)
        offering = str(result.get("service_offering", ""))
        if offering in {"", "Non-addressable"}:
            offering = "Strategy, Transformation & Advisory"
            confidence = 70
            reason = "Observed Accenture win retained; public Category/Description was insufficient for a precise Service Offering."
        else:
            confidence = max(70, int(result.get("service_offering_confidence", 70)))
            reason = "Observed Accenture win retained using the best available Category/Description Service Offering evidence."

        out.at[idx, "is_addressable"] = True
        out.at[idx, "addressability"] = "Addressable"
        out.at[idx, "capability"] = offering
        out.at[idx, "executive_capability"] = offering
        out.at[idx, "detailed_capability"] = offering
        out.at[idx, "service_offering"] = offering
        out.at[idx, "service_offering_confidence"] = confidence
        out.at[idx, "service_offering_confidence_band"] = "High" if confidence >= 85 else "Medium"
        out.at[idx, "service_offering_method"] = "Observed Accenture win guarantee"
        out.at[idx, "confidence_driver"] = "Observed Accenture win guarantee"
        out.at[idx, "low_confidence_reason"] = reason
        out.at[idx, "service_offering_evidence"] = reason
        out.at[idx, "classification_reason"] = reason
        out.at[idx, "accenture_win_override"] = True
        out.at[idx, "accenture_win_override_reason"] = reason

    # Final invariant: every source-recognised Accenture row is addressable.
    remaining = is_acc & ~bool_series(out["is_addressable"])
    if remaining.any():
        raise RuntimeError(f"Accenture retention invariant failed for {int(remaining.sum()):,} rows")
    return out


def export_supplier_audits(df: pd.DataFrame, audit_dir: Path) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    data = df.copy()
    data[VALUE] = pd.to_numeric(data[VALUE], errors="coerce").fillna(0)

    cols = [c for c in [
        CN_ID, SUPPLIER_NAME, SUPPLIER_ABN, SOURCE_SUPPLIER_DISPLAY,
        "supplier_group_pre_family", "supplier_group", "supplier_display",
        "supplier_id", "supplier_mapping_method", "supplier_workbook_reference",
        "is_accenture", "is_addressable", "service_offering", VALUE,
    ] if c in data.columns]
    data[cols].sort_values(VALUE, ascending=False).to_csv(
        audit_dir / "supplier_mapping_audit.csv", index=False
    )

    changed = data.get("supplier_group_pre_family", pd.Series("", index=data.index)).astype(str) != data["supplier_group"].astype(str)
    data.loc[changed, cols].sort_values(VALUE, ascending=False).to_csv(
        audit_dir / "supplier_mapping_changed.csv", index=False
    )

    summary = data.groupby("supplier_group", dropna=False).agg(
        contracts=(CN_ID, "size"),
        total_value=(VALUE, "sum"),
        addressable_value=(VALUE, lambda x: x[data.loc[x.index, "is_addressable"].astype(bool)].sum()),
    ).reset_index().sort_values("total_value", ascending=False)
    summary.to_csv(audit_dir / "supplier_group_summary.csv", index=False)

    acc = data[data["is_accenture"]].copy()
    acc[cols].sort_values(VALUE, ascending=False).to_csv(
        audit_dir / "accenture_supplier_reconciliation.csv", index=False
    )
# =============================================================================
# 7. RP / RE - INDEPENDENT LIGHTWEIGHT DERIVATION
# =============================================================================
def classify_reinvention(row: pd.Series) -> tuple[str, str, int, str]:
    if not bool(row.get("is_addressable", False)):
        return "Non-addressable", "Non-addressable", 100, "Outside addressable TAM"
    text = clean(" ".join([str(row.get(DESCRIPTION, "")), str(row.get(CATEGORY, ""))]))
    if re.search(r"\b(?:cyber|cybersecurity|siem|soc|zero trust|identity and access)\b", text):
        partner = "Cybersecurity"
    elif re.search(r"\b(?:cloud|software|application|ict|network|platform|erp|sap|oracle|servicenow)\b", text):
        partner = "Digital Core"
    elif re.search(r"\b(?:finance|financial|accounting|budget|payroll|treasury)\b", text):
        partner = "Finance"
    elif re.search(r"\b(?:supply chain|logistics|inventory|systems engineering|digital engineering|engineering)\b", text):
        partner = "Supply Chain and Engineering"
    elif re.search(r"\b(?:workforce|human resources|talent|learning|training|recruitment)\b", text):
        partner = "Talent"
    elif re.search(r"\b(?:customer experience|user experience|ux|marketing|brand|creative)\b", text):
        partner = "Song"
    else:
        partner = "Industry and Enterprise"

    if re.search(r"\b(?:artificial intelligence|machine learning|data|analytics|business intelligence|automation|power bi)\b", text):
        engine = "AI and Data"
    elif re.search(r"\b(?:cloud|software|application|ict|network|platform|integration|architecture|cyber|digital)\b", text):
        engine = "Technology"
    else:
        engine = "Industry and Process"
    return partner, engine, 80, "Independent Category/Description RP/RE mapping"


def apply_reinvention(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    vals = [classify_reinvention(r) for _, r in out.iterrows()]
    out["ReinventionPartner"] = [x[0] for x in vals]
    out["ReinventionEngine"] = [x[1] for x in vals]
    out["reinvention_mapping_confidence"] = [x[2] for x in vals]
    out["reinvention_mapping_reason"] = [x[3] for x in vals]
    return out


# =============================================================================
# 8. AUDITS
# =============================================================================
def export_audits(df: pd.DataFrame, audit_dir: Path) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    data = df.copy()
    data[VALUE] = pd.to_numeric(data[VALUE], errors="coerce").fillna(0)

    audit_cols = [c for c in [
        CN_ID, DESCRIPTION, CATEGORY, CATEGORY_TYPE, CATEGORY_CODE,
        SUPPLIER_NAME, "supplier_group", AGENCY, RAW_DIVISION, CANON_DIVISION,
        RAW_BRANCH, "defence_domain", VALUE, "is_addressable", "service_offering",
        "service_offering_confidence", "service_offering_confidence_band",
        "confidence_driver", "low_confidence_reason", "service_offering_method",
        "service_offering_second_choice", "service_offering_margin",
        "supplier_cohort", "supplier_cohort_effect",
        "category_context", "category_context_signal", "category_context_reason",
        "category_evidence", "description_evidence", "supplier_evidence",
        "category_score_breakdown", "description_score_breakdown",
        "supplier_score_breakdown", "service_offering_score_breakdown",
        "service_offering_evidence",
    ] if c in data.columns]

    # All Defence contracts - one row per source row / CN occurrence.
    data[audit_cols].sort_values(VALUE, ascending=False).to_csv(
        audit_dir / "service_offering_cn_validation_audit.csv", index=False
    )

    review = (
        pd.to_numeric(data["service_offering_confidence"], errors="coerce").fillna(0).lt(85)
        | data["capability_review_status"].isin(["Review suggested", "Manual review required"])
    )
    data.loc[review, audit_cols].sort_values(VALUE, ascending=False).to_csv(
        audit_dir / "service_offering_cn_validation_review_queue.csv", index=False
    )

    # Backward-compatible name, now with actual reasons.
    data.loc[review, audit_cols].sort_values(VALUE, ascending=False).to_csv(
        audit_dir / "capability_review.csv", index=False
    )

    addressable = data[bool_series(data["is_addressable"])].copy()
    addressable.groupby("service_offering", dropna=False).agg(
        contracts=(VALUE, "size"), value=(VALUE, "sum"),
        average_confidence=("service_offering_confidence", "mean")
    ).reset_index().sort_values("value", ascending=False).to_csv(
        audit_dir / "addressable_capability_summary.csv", index=False
    )

    data.groupby("defence_domain", dropna=False).agg(
        contracts=(VALUE, "size"), value=(VALUE, "sum")
    ).reset_index().sort_values("value", ascending=False).to_csv(
        audit_dir / "domain_summary.csv", index=False
    )


# =============================================================================
# 9. REGRESSION CHECKS
# =============================================================================
def run_regression_checks() -> None:
    cases = [
        ({CN_ID:"CN3472328", CATEGORY_TYPE:"Goods", CATEGORY:"Aerospace systems and components and equipment", DESCRIPTION:"Project Support Services", SUPPLIER_NAME:"Northrop Grumman"}, False, "Non-addressable"),
        ({CN_ID:"CN3296931", CATEGORY_TYPE:"Goods", CATEGORY:"Communications Devices and Accessories", DESCRIPTION:"Digitisation of Land Communications - Integrated Battlefield Telecommunications Network", SUPPLIER_NAME:"Boeing"}, False, "Non-addressable"),
        ({CN_ID:"CN3618564", CATEGORY_TYPE:"Services", CATEGORY:"Aircraft maintenance and repair services", DESCRIPTION:"Autonomic Logistics Information System Support Services", SUPPLIER_NAME:"Lockheed Martin"}, False, "Non-addressable"),
        ({CN_ID:"CN4176946", CATEGORY_TYPE:"Services", CATEGORY:"Manufacturing technologies", DESCRIPTION:"Capability Assurance Activity", SUPPLIER_NAME:"Lockheed Martin"}, False, "Non-addressable"),
        ({CN_ID:"TEST_SYSOPS", CATEGORY_TYPE:"Services", CATEGORY:"System administrators", DESCRIPTION:"Systems Operations and Sustainment", SUPPLIER_NAME:"Lockheed Martin"}, True, "Managed Services & Operations"),
        ({CN_ID:"TEST_SWMAINT", CATEGORY_TYPE:"Services", CATEGORY:"Software maintenance and support", DESCRIPTION:"Systems Sustainment Services", SUPPLIER_NAME:"Lockheed Martin"}, True, "Managed Services & Operations"),
        ({CN_ID:"TEST_SWENG", CATEGORY_TYPE:"Services", CATEGORY:"Software or hardware engineering", DESCRIPTION:"Software Developer Services", SUPPLIER_NAME:"Lockheed Martin"}, True, "SI & Engineering"),
        ({CN_ID:"TEST_LAUNCHER", CATEGORY_TYPE:"Goods", CATEGORY:"Launchers", DESCRIPTION:"Missile Launching System Maintenance", SUPPLIER_NAME:"Thales Australia"}, False, "Non-addressable"),
        ({CN_ID:"TEST_SERVER", CATEGORY_TYPE:"Goods", CATEGORY:"Computer servers", DESCRIPTION:"Software Support and Maintenance Services", SUPPLIER_NAME:"Thales Australia"}, False, "Non-addressable"),
        ({CN_ID:"CN3350297", CATEGORY_TYPE:"Goods", CATEGORY:"Personal safety and protection", DESCRIPTION:"Pilot Training System", SUPPLIER_NAME:"Lockheed Martin"}, False, "Non-addressable"),
        ({CN_ID:"CN3755438", CATEGORY_TYPE:"Goods", CATEGORY:"Rockets and subsystems", DESCRIPTION:"Project Support Services", SUPPLIER_NAME:"BAE Systems"}, False, "Non-addressable"),
        ({CN_ID:"CN1564141", CATEGORY_TYPE:"Goods", CATEGORY:"Measuring and observing and testing instruments", DESCRIPTION:"Air Traffic System Support", SUPPLIER_NAME:"Raytheon"}, False, "Non-addressable"),
        ({CN_ID:"TEST_ROCKET_SOFTWARE", CATEGORY_TYPE:"Goods", CATEGORY:"Rockets and subsystems", DESCRIPTION:"Software integration and development", SUPPLIER_NAME:"BAE Systems"}, False, "Non-addressable"),
        ({CN_ID:"TEST_AMBIGUOUS_GOODS_DIGITAL", CATEGORY_TYPE:"Goods", CATEGORY:"Miscellaneous goods", DESCRIPTION:"Software development and implementation", SUPPLIER_NAME:"Example Supplier"}, True, "SI & Engineering"),
        ({CN_ID:"TEST_AMBIGUOUS_GOODS_EY", CATEGORY_TYPE:"Goods", CATEGORY:"Miscellaneous goods", DESCRIPTION:"Software development and implementation", SUPPLIER_NAME:"EY", "supplier_group":"EY"}, True, "SI & Engineering"),
        ({CN_ID:"TEST_AMBIGUOUS_GOODS_BAE", CATEGORY_TYPE:"Goods", CATEGORY:"Miscellaneous goods", DESCRIPTION:"Software development and implementation", SUPPLIER_NAME:"BAE Systems", "supplier_group":"BAE Systems"}, False, "Non-addressable"),
        ({CN_ID:"TEST_ROCKET_EY_CONFLICT", CATEGORY_TYPE:"Goods", CATEGORY:"Rockets and subsystems", DESCRIPTION:"Software development and implementation", SUPPLIER_NAME:"EY", "supplier_group":"EY"}, False, "Non-addressable"),
        ({CN_ID:"CN3761567", CATEGORY_TYPE:"Services", CATEGORY:"Military services and national defence", DESCRIPTION:"ICT Services", SUPPLIER_NAME:"Fujitsu", "supplier_group":"Fujitsu"}, True, "SI & Engineering"),
        ({CN_ID:"TEST_CLOUD", CATEGORY_TYPE:"Services", CATEGORY:"Computer services", DESCRIPTION:"Cloud Services", SUPPLIER_NAME:"Amazon Web Services"}, True, "Cloud Infrastructure & Cyber"),
        ({CN_ID:"TEST_INTEGRATION", CATEGORY_TYPE:"Services", CATEGORY:"Information technology consultation services", DESCRIPTION:"System Integration Services", SUPPLIER_NAME:"IBM"}, True, "SI & Engineering"),
        ({CN_ID:"TEST_ENDUSER", CATEGORY_TYPE:"Services", CATEGORY:"Computer services", DESCRIPTION:"End User Support Services for ICT Service Desk", SUPPLIER_NAME:"Fujitsu"}, True, "Managed Services & Operations"),
        ({CN_ID:"TEST_VETTING", CATEGORY_TYPE:"Services", CATEGORY:"Business administration services", DESCRIPTION:"Security Vetting Services", SUPPLIER_NAME:"Example"}, False, "Non-addressable"),
        ({CN_ID:"TEST_HR_GENERIC", CATEGORY_TYPE:"Services", CATEGORY:"Human resources services", DESCRIPTION:"Management of Sale and Transfer of RAN Ships", SUPPLIER_NAME:"Thales Australia"}, False, "Non-addressable"),
        ({CN_ID:"TEST_FEAS", CATEGORY_TYPE:"Services", CATEGORY:"Feasibility studies or screening of project ideas", DESCRIPTION:"Project Definition Study", SUPPLIER_NAME:"Lockheed Martin"}, True, "Strategy, Transformation & Advisory"),
        ({CN_ID:"TEST-DATA", CATEGORY_TYPE:"Services", CATEGORY:"Data management services", DESCRIPTION:"Platform Integration", SUPPLIER_NAME:"Example"}, True, "Data, AI & Automation"),
        ({CN_ID:"TEST-CLOUD", CATEGORY_TYPE:"Services", CATEGORY:"Cloud computing services", DESCRIPTION:"Project Support Services", SUPPLIER_NAME:"Example"}, True, "Cloud Infrastructure & Cyber"),
        ({CN_ID:"TEST-PMO", CATEGORY_TYPE:"Services", CATEGORY:"Management advisory services", DESCRIPTION:"Project Support Services", SUPPLIER_NAME:"Example"}, True, "Strategy, Transformation & Advisory"),
        ({CN_ID:"CN4205371", CATEGORY_TYPE:"Services", CATEGORY:"Management support services", DESCRIPTION:"ICT Management and Support Services", SUPPLIER_NAME:"UNISYS AUSTRALIA LTD", "supplier_group":"Unisys"}, True, "Managed Services & Operations"),
        ({CN_ID:"CN3942327", CATEGORY_TYPE:"Services", CATEGORY:"Management support services", DESCRIPTION:"Management Support Services", SUPPLIER_NAME:"KPMG", "supplier_group":"KPMG"}, True, "Strategy, Transformation & Advisory"),
        ({CN_ID:"TEST_KPMG_GENERIC", CATEGORY_TYPE:"Services", CATEGORY:"Military services and national defence", DESCRIPTION:"Professional Services", SUPPLIER_NAME:"KPMG", "supplier_group":"KPMG"}, True, "Strategy, Transformation & Advisory"),
        ({CN_ID:"TEST_KPMG_WEAPONS", CATEGORY_TYPE:"Goods", CATEGORY:"Rockets and subsystems", DESCRIPTION:"Project Support Services", SUPPLIER_NAME:"KPMG", "supplier_group":"KPMG"}, False, "Non-addressable"),
        ({CN_ID:"CN4066214", CATEGORY_TYPE:"Services", CATEGORY:"Management support services", DESCRIPTION:"Capability Development Services", SUPPLIER_NAME:"Accenture", "supplier_group":"Accenture"}, True, "Data, AI & Automation"),
        ({CN_ID:"CN3546046", CATEGORY_TYPE:"Services", CATEGORY:"Software or hardware engineering", DESCRIPTION:"Maritime Combat Helicopter Capability Assurance Program", SUPPLIER_NAME:"DEPARTMENT OF DEFENCE"}, False, "Non-addressable"),
        ({CN_ID:"CN298411", CATEGORY_TYPE:"Services", CATEGORY:"Education and Training Services", DESCRIPTION:"PROVISION OF ACADEMIC STUDIES", SUPPLIER_NAME:"University of New South Wales"}, False, "Non-addressable"),
        ({CN_ID:"CN3362660", CATEGORY_TYPE:"Services", CATEGORY:"Education and Training Services", DESCRIPTION:"Provision of Training Services", SUPPLIER_NAME:"BOEING DEFENCE AUSTRALIA LTD", "supplier_group":"Boeing"}, False, "Non-addressable"),
        ({CN_ID:"CN3916100", CATEGORY_TYPE:"Services", CATEGORY:"Education and Training Services", DESCRIPTION:"Platform and Systems Training Services", SUPPLIER_NAME:"CAE AUSTRALIA PTY LTD", "supplier_group":"CAE"}, False, "Non-addressable"),
        ({CN_ID:"CN3801013", CATEGORY_TYPE:"Goods", CATEGORY:"Software", DESCRIPTION:"Electro Magnetic Battle Management Capability", SUPPLIER_NAME:"CONSUNET PTY LTD"}, False, "Non-addressable"),
        ({CN_ID:"TEST_CENTRAL_PROCESSING", CATEGORY_TYPE:"Services", CATEGORY:"Computer services", DESCRIPTION:"Centralised Processing Services", SUPPLIER_NAME:"Leidos", "supplier_group":"Leidos"}, True, "Managed Services & Operations"),
        ({CN_ID:"TEST_HOSTING_PRECEDENCE", CATEGORY_TYPE:"Services", CATEGORY:"Application implementation services", DESCRIPTION:"Infrastructure Hosting Services", SUPPLIER_NAME:"Leidos", "supplier_group":"Leidos"}, True, "Cloud Infrastructure & Cyber"),
        ({CN_ID:"TEST_SOFTWARE_SUPPORT_PRECEDENCE", CATEGORY_TYPE:"Services", CATEGORY:"Computer services", DESCRIPTION:"Software Support Services", SUPPLIER_NAME:"Leidos", "supplier_group":"Leidos"}, True, "Managed Services & Operations"),
        ({CN_ID:"TEST_ENVIRONMENTAL_SERVICES", CATEGORY_TYPE:"Services", CATEGORY:"Computer services", DESCRIPTION:"Environmental Services", SUPPLIER_NAME:"Leidos", "supplier_group":"Leidos"}, False, "Non-addressable"),
        ({CN_ID:"TEST_HARDWARE_REFRESH_SERVICE", CATEGORY_TYPE:"Services", CATEGORY:"Computer services", DESCRIPTION:"Hardware Refresh", SUPPLIER_NAME:"Leidos", "supplier_group":"Leidos"}, False, "Non-addressable"),
        ({CN_ID:"CN3461140", CATEGORY_TYPE:"Services", CATEGORY:"Professional engineering services", DESCRIPTION:"Command and Control Systems Integration Support to Amphibious Capability", SUPPLIER_NAME:"Leidos", "supplier_group":"Leidos"}, False, "Non-addressable"),
        ({CN_ID:"TEST_FN_PM_TECH", CATEGORY_TYPE:"Services", CATEGORY:"Management support services", DESCRIPTION:"Project Management Services", SUPPLIER_NAME:"FUJITSU AUSTRALIA LIMITED", "supplier_group":"Fujitsu", RAW_DIVISION:"DDG", RAW_BRANCH:"DDG - ICTDD"}, True, "Strategy, Transformation & Advisory"),
        ({CN_ID:"TEST_FN_APP_SUSTAIN", CATEGORY_TYPE:"Services", CATEGORY:"Management support services", DESCRIPTION:"Application Sustainment Support", SUPPLIER_NAME:"EVIDEN AUSTRALIA PTY LTD", "supplier_group":"Atos / Eviden"}, True, "Managed Services & Operations"),
        ({CN_ID:"TEST_FN_ICT_SECURITY", CATEGORY_TYPE:"Services", CATEGORY:"Management support services", DESCRIPTION:"ICT Security Uplift", SUPPLIER_NAME:"LEIDOS AUSTRALIA PTY LIMITED", "supplier_group":"Leidos"}, True, "Cloud Infrastructure & Cyber"),
        ({CN_ID:"TEST_FN_ICT_SUPPORT", CATEGORY_TYPE:"Services", CATEGORY:"Management support services", DESCRIPTION:"ICT Support", SUPPLIER_NAME:"UNISYS AUSTRALIA LTD", "supplier_group":"Unisys"}, True, "Managed Services & Operations"),
        ({CN_ID:"TEST_FN_DATA_SUPPORT", CATEGORY_TYPE:"Services", CATEGORY:"Military services and national defence", DESCRIPTION:"Data Support Services", SUPPLIER_NAME:"KNOW YOUR DATA PTY LTD", "supplier_group":"Know Your Data Pty Ltd"}, True, "Data, AI & Automation"),
        ({CN_ID:"TEST_FN_CLOUD_SUPPORT", CATEGORY_TYPE:"Services", CATEGORY:"Management support services", DESCRIPTION:"ICT Cloud Support Services", SUPPLIER_NAME:"SOL STAK PTY LTD", "supplier_group":"Sol Stak Pty Ltd Trust", RAW_DIVISION:"DDG", RAW_BRANCH:"DDG - ICTDD"}, True, "Cloud Infrastructure & Cyber"),
        ({CN_ID:"TEST_FN_GENERIC_DXC_SUPPORT", CATEGORY_TYPE:"Services", CATEGORY:"Management support services", DESCRIPTION:"Support Services", SUPPLIER_NAME:"DXC TECHNOLOGY AUSTRALIA PTY LIMITED", "supplier_group":"DXC Technology", RAW_DIVISION:"DDG", RAW_BRANCH:"DDG - ERP"}, True, "Managed Services & Operations"),
        ({CN_ID:"TEST_FN_PROJECT_DELIVERY_DIGITAL", CATEGORY_TYPE:"Services", CATEGORY:"Management support services", DESCRIPTION:"Project Delivery Team", SUPPLIER_NAME:"FUJITSU AUSTRALIA LIMITED", "supplier_group":"Fujitsu", RAW_DIVISION:"DDG", RAW_BRANCH:"DDG - ICTDD"}, True, "Strategy, Transformation & Advisory"),
        ({CN_ID:"TEST_FN_SOFTWARE_MAINT", CATEGORY_TYPE:"Services", CATEGORY:"Military services and national defence", DESCRIPTION:"Software Maintenance", SUPPLIER_NAME:"FUJITSU AUSTRALIA LIMITED", "supplier_group":"Fujitsu"}, True, "Managed Services & Operations"),
        ({CN_ID:"TEST_FN_HR_CONSULT", CATEGORY_TYPE:"Services", CATEGORY:"Human resources services", DESCRIPTION:"Professional Services", SUPPLIER_NAME:"Ernst & Young", "supplier_group":"EY"}, True, "Strategy, Transformation & Advisory"),
        ({CN_ID:"TEST_FN_DATA_REMEDIATION", CATEGORY_TYPE:"Services", CATEGORY:"Professional engineering services", DESCRIPTION:"Data remediation", SUPPLIER_NAME:"ATOS (AUSTRALIA) PTY LTD", "supplier_group":"Atos / Eviden"}, True, "Data, AI & Automation"),
        ({CN_ID:"TEST_DATA_CENTRE_FACILITY_OPERATOR", CATEGORY_TYPE:"Services", CATEGORY:"Professional engineering services", DESCRIPTION:"Data Centre Services", SUPPLIER_NAME:"Canberra Data Centres Pty Ltd", "supplier_group":"Canberra Data Centres Pty Ltd", RAW_DIVISION:"DDG", RAW_BRANCH:"DDG - ICTDD"}, False, "Non-addressable"),
        ({CN_ID:"TEST_DATA_CENTRE_HOSTING_REMAINS", CATEGORY_TYPE:"Services", CATEGORY:"Maintenance or support fees", DESCRIPTION:"Data Centre Hosting Services", SUPPLIER_NAME:"NTT COM ICT SOLUTIONS (AUSTRALIA)", "supplier_group":"NTT Com Ict Solutions (Australia)", RAW_DIVISION:"DDG", RAW_BRANCH:"DDG - ICTDD"}, True, "Cloud Infrastructure & Cyber"),
        ({CN_ID:"CN3652601", CATEGORY_TYPE:"Services", CATEGORY:"Computer services", DESCRIPTION:"Air Combat Capability Facilities Project", SUPPLIER_NAME:"Leidos", "supplier_group":"Leidos"}, False, "Non-addressable"),
    ]
    for payload, expected_addr, expected_offer in cases:
        row = pd.Series({"supplier_group": payload.get("supplier_group", payload.get(SUPPLIER_NAME,"")), **payload})
        result = classify_service_offering(row)
        if bool(result["is_addressable"]) != expected_addr or str(result["service_offering"]) != expected_offer:
            raise RuntimeError(
                f"Regression failed for {payload.get(CN_ID)}: expected ({expected_addr}, {expected_offer}), "
                f"got ({result['is_addressable']}, {result['service_offering']})"
            )


# =============================================================================
# MAIN
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fresh base-header-first Defence master build")
    parser.add_argument("--input", default="app_data.parquet", help="Fresh source parquet")
    parser.add_argument("--output-dir", default="master_output")
    parser.add_argument("--domain-lookup", default="defence_domain_lookup.csv")
    parser.add_argument("--supplier-mapping", default="supplier_mapping_master.xlsx")
    return parser.parse_args()


def main() -> None:
    run_regression_checks()
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    audit_dir = output_dir / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"1/8 Loading fresh source parquet: {input_path}")
    raw = pd.read_parquet(input_path)
    validate_source_headers(raw)
    print(f"Source headers preserved exactly: {len(raw.columns):,} columns")
    raw[VALUE] = pd.to_numeric(raw[VALUE], errors="coerce").fillna(0)

    print("2/8 Applying strict Defence procurer filter")
    master = strict_defence_filter(raw)
    print(f"Defence rows: {len(master):,}; total Defence value: ${master[VALUE].sum()/1e9:,.2f}B")

    print("3/8 Adding supplier identity without changing raw supplier headers")
    master = apply_supplier_mapping(master, Path(args.supplier_mapping))

    print("4/8 Adding canonical organisation fields without changing raw Division/Branch headers")
    master = add_canonical_org_fields(master)

    print("5/8 Mapping Defence domain from raw Division + Branch")
    master = add_defence_domain(master, Path(args.domain_lookup))

    print("6/8 Applying fresh Category -> Description -> Supplier Service Offering model")
    master = apply_service_offering_mapping(master)
    master["is_addressable"] = bool_series(master["is_addressable"])
    master = retain_observed_accenture_wins(master)
    master = enforce_authoritative_contract_overrides(master)
    master["is_addressable"] = bool_series(master["is_addressable"])

    print("7/8 Applying independent Reinvention Partner / Engine mapping")
    master = apply_reinvention(master)

    # Validate key reviewed contracts if they exist.
    for cn, expected in {
        "CN3472328": False,
        "CN3296931": False,
    }.items():
        rows = master[master[CN_ID].fillna("").astype(str).str.upper().eq(cn)]
        if not rows.empty and bool_series(rows["is_addressable"]).any() != expected:
            raise RuntimeError(f"{cn} regression: expected is_addressable={expected}")

    print("8/8 Writing canonical parquet and validation audits")
    master_path = output_dir / "master_defence_contracts.parquet"
    master.to_parquet(master_path, index=False)
    export_audits(master, audit_dir)
    export_supplier_audits(master, audit_dir)

    addressable = master[master["is_addressable"]]
    summary = {
        "rows": int(len(master)),
        "defence_value": float(master[VALUE].sum()),
        "addressable_value": float(addressable[VALUE].sum()),
        "accenture_addressable_value": float(addressable.loc[addressable["is_accenture"], VALUE].sum()),
        "service_offerings": addressable["service_offering"].value_counts().to_dict(),
        "domains": master["defence_domain"].value_counts().to_dict(),
        "source_headers_preserved": list(raw.columns),
    }
    (output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"Wrote: {master_path}")
    print(f"Addressable market: ${summary['addressable_value']/1e9:,.2f}B")
    print(f"Accenture addressable wins: ${summary['accenture_addressable_value']/1e9:,.3f}B")
    print(f"Supplier audit: {audit_dir / 'supplier_mapping_audit.csv'}")
    print(f"Review queue: {audit_dir / 'service_offering_cn_validation_review_queue.csv'}")
    print("Build complete.")


if __name__ == "__main__":
    main()
