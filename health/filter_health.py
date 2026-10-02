from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# =============================================================================
# HEALTH PORTFOLIO MASTER FILTER
# =============================================================================
# Pipeline:
#   whole-of-government AusTender
#       -> Health portfolio scope
#       -> addressability gate
#       -> Service Offering (addressable rows only)
#       -> classified Health parquet + audit / review queue
#
# IMPORTANT:
#   * Supplier identity is NOT used to decide Service Offering.
#   * Supplier identity is used only for canonical grouping and the explicit
#     business invariant that an observed Accenture win is addressable.
#   * Agency and supplier source names are preserved in raw columns, while
#     canonical group fields are added for longitudinal reporting.
#   * Non-addressable rows remain in the Health procurement population.
#   * Non-addressable rows receive NO Service Offering (blank / NA).
#   * Service Offering is determined from Description + Category only.
#   * Description is deliberately weighted more heavily than Category because
#     Health project descriptions are generally more informative than Defence.
# =============================================================================

CN_ID = "CN ID"
AGENCY = "Agency"
DESCRIPTION = "Description"
CATEGORY = "Category"
CATEGORY_TYPE = "Category Type"
VALUE = "Value"
SUPPLIER_NAME = "Supplier Name"
SUPPLIER_ABN = "Supplier ABN"
RAW_AGENCY_COL = "Agency Raw"
AGENCY_GROUP_COL = "agency_group"
SUPPLIER_GROUP_COL = "supplier_group"

FINAL_SERVICE_OFFERINGS = [
    "Strategy, Transformation & Advisory",
    "SI & Engineering",
    "Data, AI & Automation",
    "Cloud Infrastructure & Cyber",
    "Managed Services & Operations",
]

# Canonical Health agencies used by the dashboards. Historical source labels are
# mapped to these names so longitudinal spend is not split by machinery-of-
# government renames. The original source value is preserved in ``Agency Raw``.
HEALTH_AGENCY_ALIASES = {
    # Exact Health portfolio scope agreed for this dashboard. Keys are already
    # normalised with the same punctuation/spacing rules used by clean().
    # Historical aliases are mapped to a stable canonical reporting label where
    # appropriate, while distinct agencies remain distinct.
    "department of health and aged care": "Department of Health, Disability and Ageing",
    "department of health disability and ageing": "Department of Health, Disability and Ageing",
    "department of health and aged care therapeutic goods administration": "Department of Health and Aged Care - Therapeutic Goods Administration",
    "australian digital health agency": "Australian Digital Health Agency",
    "australian aged care quality agency": "Australian Aged Care Quality Agency",
    "australian institute of health and welfare": "Australian Institute of Health and Welfare",
    "department of social services": "Department of Social Services",
    "independent health and aged care pricing authority": "Independent Health and Aged Care Pricing Authority",
    "national health funding body": "National Health Funding Body",
    "national health and medical research council": "National Health and Medical Research Council",
    "organ and tissue authority": "Organ and Tissue Authority",
}


# Reviewed supplier families. These are label-only consolidations; they never add,
# remove or revalue contracts. The original Supplier Name remains untouched.
CANONICAL_SUPPLIER_FAMILIES = [
    ("Accenture", [r"\bACCENTURE\b"]),
    ("Deloitte", [r"\bDELOITTE\b"]),
    ("EY", [r"\bERNST\s*(?:&|AND)\s*YOUNG\b", r"^EY(?:\s|$)"]),
    ("KPMG", [r"\bKPMG\b"]),
    ("PwC", [r"\bPRICEWATERHOUSECOOPERS\b", r"\bPRICEWATERHOUSE\s+COOPERS\b", r"\bPWC\b"]),
    ("IBM", [r"\bIBM\b", r"\bINTERNATIONAL\s+BUSINESS\s+MACHINES\b"]),
    ("Fujitsu", [r"\bFUJITSU\b"]),
    ("DXC Technology", [r"\bDXC\b"]),
    ("Capgemini", [r"\bCAPGEMINI\b"]),
    ("Microsoft", [r"\bMICROSOFT\b"]),
    ("Amazon Web Services", [r"\bAMAZON\s+WEB\s+SERVICES\b", r"^AWS(?:\s|$)"]),
    ("Oracle", [r"\bORACLE\b"]),
    ("SAP", [r"^SAP(?:\s|$)", r"\bSAP AUSTRALIA\b"]),
    ("Data#3", [r"\bDATA#?3\b"]),
    ("Telstra", [r"\bTELSTRA\b"]),
    ("Optus", [r"\bOPTUS\b"]),
    ("Salesforce", [r"\bSALESFORCE\b"]),
    ("ServiceNow", [r"\bSERVICENOW\b"]),
]



def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9+#./ -]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def any_pattern(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return m.group(0)
    return None


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Input dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise SystemExit(f"Unsupported input type: {suffix}. Use CSV or parquet.")


def normalise_name(value: object) -> str:
    text = str(value or "").upper().replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_supplier_name(value: object) -> str:
    raw = str(value or "").strip()
    norm = normalise_name(raw)
    for canonical, patterns in CANONICAL_SUPPLIER_FAMILIES:
        if any(re.search(pattern, norm, flags=re.IGNORECASE) for pattern in patterns):
            return canonical
    return raw or "Unknown Supplier"


ACCENTURE_EXACT_ALIASES = {
    "ACCENTURE AUSTRALIA PTY LTD",
    "ACCENTURE AUSTRALIA HOLDINGS",
    "ACCENTURE AUSTRALIA HOLDINGS PTY LTD",
}


def is_accenture_identity(raw_name: object, source_display: object = "") -> bool:
    """Recognise Accenture from either raw supplier name or source display.

    Exact reviewed Australian aliases are listed explicitly, with a conservative
    ACCENTURE token fallback so spacing/case/punctuation variants are retained.
    """
    values = [normalise_name(raw_name), normalise_name(source_display)]
    for value in values:
        if not value:
            continue
        if value in ACCENTURE_EXACT_ALIASES:
            return True
        if re.search(r"\bACCENTURE\b", value):
            return True
    return False


def add_supplier_grouping(df: pd.DataFrame) -> pd.DataFrame:
    """Add one conservative canonical supplier label.

    Supplier Name is the source of truth because it is the field used for the
    manual reconciliation against fullDataset. supplier_display is deliberately
    not allowed to replace the raw legal-name evidence.
    """
    out = df.copy()
    raw_supplier = (
        out[SUPPLIER_NAME].fillna("").astype(str).str.strip()
        if SUPPLIER_NAME in out.columns
        else pd.Series("Unknown Supplier", index=out.index)
    )

    out[SUPPLIER_GROUP_COL] = raw_supplier.map(canonical_supplier_name)
    acc_mask = raw_supplier.map(lambda x: is_accenture_identity(x, ""))
    out.loc[acc_mask, SUPPLIER_GROUP_COL] = "Accenture"
    out["is_accenture"] = acc_mask.astype(bool)

    inconsistent = out["is_accenture"].ne(out[SUPPLIER_GROUP_COL].eq("Accenture"))
    if inconsistent.any():
        raise RuntimeError(
            f"Supplier grouping invariant failed for {int(inconsistent.sum()):,} rows: "
            "is_accenture must exactly match supplier_group == 'Accenture'."
        )
    return out


def canonical_agency_name(value: object) -> str | None:
    # clean() removes punctuation, so alias keys above are stored in that exact
    # normalised form (e.g. no comma after "Health").
    return HEALTH_AGENCY_ALIASES.get(clean(value))


# =============================================================================
# 1. HEALTH SCOPE
# =============================================================================

def apply_health_scope(df: pd.DataFrame) -> pd.DataFrame:
    if AGENCY not in df.columns:
        raise RuntimeError(f"Input dataset does not contain '{AGENCY}'.")

    canonical = df[AGENCY].map(canonical_agency_name)
    in_scope = canonical.notna()
    out = df.loc[in_scope].copy()
    if out.empty:
        raise RuntimeError("Health portfolio scope filter returned no rows.")

    # Preserve the original source label, then replace Agency with the canonical
    # reporting label so existing dashboards aggregate historical aliases together.
    out[RAW_AGENCY_COL] = out[AGENCY]
    out[AGENCY_GROUP_COL] = canonical.loc[in_scope].astype(str)
    out[AGENCY] = out[AGENCY_GROUP_COL]
    out["Health Portfolio Scope"] = True
    return out


# =============================================================================
# 2. ADDRESSABILITY
# =============================================================================
# Goods are conservative. Only software/licence/subscription/cloud-style goods are
# considered addressable. Medical products, devices, consumables and physical ICT
# hardware stay in total Health procurement but outside Accenture TAM.


# Health-specific hard exclusions. These are checked BEFORE generic addressable
# words such as management, support, program, data or implementation. This stops
# clinical/product procurement from entering TAM simply because the contract
# description also contains a generic professional-services term.
HARD_HEALTH_PRODUCT_CATEGORY_PATTERNS = [
    r"\bvaccines?\b",
    r"\bimmuni[sz]ation\b",
    r"\bpharmaceutical(?:s| products?)?\b",
    r"\bmedicines?\b",
    r"\bmedicinal products?\b",
    r"\btherapeutic goods?\b",
    r"\bdrugs?\b",
    r"\bblood products?\b",
    r"\bplasma\b",
    r"\bmedical equipment\b",
    r"\bmedical devices?\b",
    r"\bmedical supplies\b",
    r"\bmedical consumables?\b",
    r"\bsurgical (?:equipment|instruments?|supplies)\b",
    r"\blaboratory (?:equipment|supplies|consumables?)\b",
    r"\bdiagnostic (?:equipment|devices?|reagents?)\b",
    r"\breagents?\b",
    r"\bprosthe(?:sis|ses|tic)\b",
    r"\bimplants?\b",
    r"\bpersonal protective equipment\b",
    r"\bppe\b",
    r"\bpatient care and treatment products?\b",
    r"\bdisease prevention and control\b",
]

HARD_HEALTH_PRODUCT_DESCRIPTION_PATTERNS = [
    r"\bvaccine(?:s| administration| rollout| delivery| distribution| supply| program| programme)?\b",
    r"\bimmuni[sz]ation(?: services?| program| programme| delivery| administration)?\b",
    r"\b(?:supply|purchase|procurement|distribution|delivery) of (?:medicines?|pharmaceuticals?|drugs?|vaccines?)\b",
    r"\bpharmaceutical(?:s| products?)?\b",
    r"\bmedicines?\b",
    r"\bmedicinal products?\b",
    r"\bmedical (?:equipment|devices?|supplies|consumables?)\b",
    r"\bsurgical (?:equipment|instruments?|supplies)\b",
    r"\bdiagnostic (?:equipment|devices?|reagents?)\b",
    r"\bblood products?\b",
    r"\bplasma\b",
    r"\breagents?\b",
    r"\bprosthe(?:sis|ses|tic)\b",
    r"\bimplants?\b",
]

ADDRESSABLE_GOODS_PATTERNS = [
    r"\bsoftware\b",
    r"\blicen[cs](?:e|es|ing)\b",
    r"\bsubscription(?:s)?\b",
    r"\bsaas\b",
    r"\bsoftware as a service\b",
    r"\bcloud\b",
]

NON_ADDRESSABLE_GOODS_PATTERNS = [
    # Health / clinical products
    r"\bpharmaceutical(?:s| products?)?\b",
    r"\bdrugs?\b",
    r"\bvaccines?\b",
    r"\bblood products?\b",
    r"\bplasma\b",
    r"\bmedical (?:devices?|equipment|supplies|consumables?)\b",
    r"\bsurgical (?:equipment|instruments?|supplies)\b",
    r"\blaboratory (?:equipment|supplies|consumables?)\b",
    r"\bdiagnostic (?:equipment|devices?|reagents?)\b",
    r"\breagents?\b",
    r"\bprosthe(?:sis|ses|tic)\b",
    r"\bimplants?\b",
    r"\bpersonal protective equipment\b",
    r"\bppe\b",
    # General physical goods
    r"\bcomponents?\b",
    r"\bequipment\b",
    r"\baccessories\b",
    r"\bdevices?\b",
    r"\bparts?\b",
    r"\bhardware\b",
    r"\bmachinery\b",
    r"\bvehicles?\b",
    r"\bfurniture\b",
    r"\bclothing\b",
    r"\bfood\b",
    r"\bfuel\b",
    r"\bprinters?\b",
    r"\bcomputer servers?\b",
    r"\bcomputer equipment\b",
    r"\btelecommunications equipment\b",
]

# These service subjects are outside the transformation / technology TAM.
NON_ADDRESSABLE_SERVICE_CATEGORY_PATTERNS = [
    # Clinical / health delivery
    r"\bhealth services?\b",
    r"\bmedical services?\b",
    r"\bclinical services?\b",
    r"\bhospital services?\b",
    r"\bpathology services?\b",
    r"\blaboratory testing services?\b",
    r"\bdiagnostic services?\b",
    r"\bpharmacy services?\b",
    r"\bpatient care\b",
    r"\baged care services?\b",
    r"\bdisability support services?\b",
    r"\bmental health services?\b",
    r"\bmedical research\b",
    r"\bclinical research\b",
    r"\bclinical trials?\b",
    # Property / facilities / logistics
    r"\breal estate\b",
    r"\bproperty\b",
    r"\bconstruction\b",
    r"\bbuilding\b",
    r"\bfacilit(?:y|ies) management\b",
    r"\bfacility maintenance\b",
    r"\bcleaning\b",
    r"\bcatering\b",
    r"\bwaste\b",
    r"\btransportation\b",
    r"\bfreight\b",
    r"\bcourier\b",
    r"\bwarehousing\b",
    r"\btravel\b",
    r"\bhotel\b",
    r"\baccommodation\b",
    r"\butilities\b",
    r"\belectricity\b",
    r"\bsecurity guard\b",
    r"\bguarding\b",
    # Generic people sourcing / professional categories outside the target TAM
    r"\brecruitment\b",
    r"\bpersonnel recruitment\b",
    r"\blabour hire\b",
    r"\blegal services?\b",
]

NON_ADDRESSABLE_DESCRIPTION_PATTERNS = [
    r"\b(?:supply|purchase|procurement) of (?:vaccines?|pharmaceuticals?|drugs?|medical devices?|medical equipment|blood products?|plasma)\b",
    r"\b(?:clinical|medical|pathology|diagnostic) services?\b",
    r"\bclinical trials?\b",
    r"\bmedical research\b",
    r"\bpatient transport\b",
    r"\bcleaning services?\b",
    r"\bcatering services?\b",
    r"\bbuilding works?\b",
    r"\bconstruction works?\b",
    r"\bfacilit(?:y|ies) maintenance\b",
    r"\btravel services?\b",
    r"\baccommodation services?\b",
    r"\brecruitment services?\b",
]

# Recognised addressable subject-matter. At least one Category/Description signal is
# needed for Services unless a hard non-addressable gate fires.
ADDRESSABLE_SUBJECT_PATTERNS = [
    # Strategy / transformation
    r"\bstrategy\b", r"\bstrategic\b", r"\badvisory\b", r"\bconsult(?:ing|ancy)\b",
    r"\btransformation\b", r"\boperating model\b", r"\borganisational design\b",
    r"\bprocess improvement\b", r"\bchange management\b", r"\bbusiness case\b",
    r"\bprogram(?:me)? management\b", r"\bproject management\b", r"\bpmo\b",
    r"\bbusiness analysis\b", r"\bassurance\b", r"\bgovernance\b",
    # Data / AI
    r"\bdata\b", r"\banalytics\b", r"\bbusiness intelligence\b", r"\bpower bi\b",
    r"\bartificial intelligence\b", r"\bmachine learning\b", r"\bautomation\b", r"\brpa\b",
    # Cloud / cyber / infra
    r"\bcloud\b", r"\bhosting\b", r"\bcyber(?:security)?\b", r"\binformation security\b",
    r"\bnetwork(?:ing)?\b", r"\bidentity and access management\b", r"\bsiem\b", r"\bsoc\b",
    # SI / engineering
    r"\bsoftware\b", r"\bapplication\b", r"\bsystems? integration\b", r"\bimplementation\b",
    r"\bdigital\b", r"\bict\b", r"\binformation technology\b", r"\berp\b", r"\bsap\b",
    r"\bservicenow\b", r"\bsalesforce\b", r"\barchitecture\b", r"\bdevops\b",
    # Managed services / ops
    r"\bmanaged services?\b", r"\bservice desk\b", r"\bhelp desk\b", r"\bsupport services?\b",
    r"\bapplication support\b", r"\bsystem support\b", r"\bsoftware support\b",
    r"\bmaintenance and support\b", r"\boperations?\b", r"\bsustainment\b",
    r"\blicen[cs](?:e|es|ing)\b", r"\bsubscription(?:s)?\b",
]

# Broad categories that are too vague to establish addressability by themselves.
GENERIC_CATEGORY_PATTERNS = [
    r"^management support services?$",
    r"^management advisory services?$",
    r"^computer services?$",
    r"^information technology consultation services?$",
    r"^professional services?$",
    r"^project management$",
    r"^project administration or planning$",
]

# -----------------------------------------------------------------------------
# REVIEWED CATEGORY PRIORS
# -----------------------------------------------------------------------------
# Built from the supplied Health category population. These priors deliberately
# separate categories that are strong evidence for Accenture-style technology /
# transformation work from categories that are clearly outside the TAM. Broad
# categories such as Computer services, Management advisory services, Project
# management and Temporary personnel services remain contextual and still need
# Description evidence rather than being forced in or out solely by Category.

STRONG_ADDRESSABLE_CATEGORY_EXACT = {
    "software",
    "software maintenance and support",
    "software as a service saas - cloud",
    "platform as a service paas - cloud",
    "infrastructure as a service iaas - cloud",
    "platform software as a service",
    "application implementation services",
    "business intelligence consulting services",
    "data processing or preparation services",
    "software or hardware engineering",
    "computer programmers",
    "system administrators",
    "computer hardware maintenance and support",
    "computer hardware maintenance or support",
    "maintenance or support fees",
    "management information systems mis",
}

STRONG_NON_ADDRESSABLE_CATEGORY_EXACT = {
    # Clinical / health delivery and products
    "health programs",
    "comprehensive health services",
    "health administration services",
    "disease prevention and control",
    "individual health screening and assessment services",
    "psychologists services",
    "rehabilitation services",
    "healthcare provider support persons",
    "occupational health or safety services",
    "drugs and pharmaceutical products",
    "medical equipment and accessories and supplies",
    "patient care and treatment products and supplies",
    "medical training and education supplies",
    "laboratory and scientific equipment",
    "laboratory supplies and fixtures",
    "measuring and observing and testing instruments",
    "communication aids for the physically challenged",
    # Research
    "research programs",
    "medical science research and experimentation",
    "military science and research",
    # People sourcing / legal / employment
    "personnel recruitment",
    "employment services",
    "legal services",
    "business law services",
    # Property / facilities / logistics / physical goods
    "lease and rental of property or building",
    "property management services",
    "building construction and support and maintenance and repair services",
    "general building construction",
    "building support services",
    "nonresidential building construction services",
    "general building and office cleaning and maintenance services",
    "specialised warehousing and storage",
    "fleet management services",
    "vehicle leasing",
    "travel facilitation",
    "conference centres",
    "meeting facilities",
    "hotels and lodging and meeting facilities",
    "hotels and motels and inns",
    "restaurants and catering",
    "banquet and catering services",
    "office furniture",
    "furniture",
    "office supplies",
    "stationery",
    "food and beverage products",
    "computer equipment and accessories",
    "components for information technology or broadcasting or telecommunications",
    "electronic hardware and component parts and accessories",
    "communications devices and accessories",
    "notebook computers",
    "computer servers",
    "computer printers",
    "mobile phones",
}

# Categories from the supplied list that are genuinely mixed. They are retained
# as contextual rather than treated as automatic inclusions.
CONTEXTUAL_CATEGORY_EXACT = {
    "computer services",
    "information technology consultation services",
    "management advisory services",
    "management support services",
    "project management",
    "project administration or planning",
    "temporary personnel services",
    "data services",
    "information services",
    "statistics",
    "strategic planning consultation services",
    "corporate objectives or policy development",
    "business administration services",
    "professional procurement services",
    "risk management consultation services",
    "organisational structure consultation",
    "feasibility studies or screening of project ideas",
}


def category_prior(category: str) -> tuple[str, int, str] | None:
    """Return a reviewed prior from the supplied Health category universe."""
    if category in STRONG_NON_ADDRESSABLE_CATEGORY_EXACT:
        return ("non_addressable", 98, "Reviewed Health category is outside the target transformation / technology TAM")
    if category in STRONG_ADDRESSABLE_CATEGORY_EXACT:
        return ("addressable", 93, "Reviewed Health category is strong evidence of technology / digital delivery")
    if category in CONTEXTUAL_CATEGORY_EXACT:
        return ("contextual", 0, "Reviewed Health category is mixed and requires Description evidence")
    return None


def classify_addressability(row: pd.Series) -> dict[str, object]:
    cat_type = clean(row.get(CATEGORY_TYPE, ""))
    category = clean(row.get(CATEGORY, ""))
    description = clean(row.get(DESCRIPTION, ""))

    # RESEARCH GATE: research projects are outside the Accenture TAM.
    # Apply this before generic advisory/data/technology evidence so words such as
    # project, data, analysis or management cannot pull a research project into TAM.
    # The explicit Accenture-win guarantee is applied later as the only override.
    research_signal = any_pattern(f"{category} {description}", [
        r"\bresearch projects?\b",
        r"\bresearch studies?\b",
        r"\bresearch services?\b",
        r"\bmedical research\b",
        r"\bclinical research\b",
        r"\bhealth research\b",
        r"\bresearch grants?\b",
        r"\bresearch fellowships?\b",
        r"\bresearch program(?:me)?s?\b",
        r"\bresearch and development\b",
        r"\br&d\b",
    ])
    if research_signal:
        return {
            "is_addressable": False,
            "addressability": "Not Addressable",
            "addressability_confidence": 99,
            "addressability_reason": f"Research project/service is outside Accenture TAM: '{research_signal}'.",
            "addressability_method": "Research hard exclusion",
        }

    # HARD HEALTH SUBJECT GATE: clinical products, vaccines, medicines and medical
    # equipment are outside the transformation/technology TAM regardless of generic
    # words such as management, support, program or implementation elsewhere in the
    # record. The explicit Accenture-win guarantee is applied later as a separate
    # business invariant.
    hard_health_cat = any_pattern(category, HARD_HEALTH_PRODUCT_CATEGORY_PATTERNS)
    hard_health_desc = any_pattern(description, HARD_HEALTH_PRODUCT_DESCRIPTION_PATTERNS)
    if hard_health_cat or hard_health_desc:
        signal = hard_health_cat or hard_health_desc
        source = "Category" if hard_health_cat else "Description"
        return {
            "is_addressable": False,
            "addressability": "Not Addressable",
            "addressability_confidence": 99,
            "addressability_reason": f"Hard Health product/clinical exclusion in {source}: '{signal}'.",
            "addressability_method": "Health product / clinical hard exclusion",
        }

    # REVIEWED CATEGORY PRIOR: use the supplied category population as an
    # additional deterministic signal. Hard non-addressable categories win here;
    # strong digital/technology categories can establish addressability. Mixed
    # categories are intentionally left to the normal Description-led logic.
    prior = category_prior(category)
    if prior is not None:
        prior_class, prior_confidence, prior_reason = prior
        if prior_class == "non_addressable":
            return {
                "is_addressable": False,
                "addressability": "Not Addressable",
                "addressability_confidence": prior_confidence,
                "addressability_reason": f"{prior_reason}: '{row.get(CATEGORY, '')}'.",
                "addressability_method": "Reviewed Category hard exclusion",
            }
        if prior_class == "addressable":
            return {
                "is_addressable": True,
                "addressability": "Addressable",
                "addressability_confidence": prior_confidence,
                "addressability_reason": f"{prior_reason}: '{row.get(CATEGORY, '')}'.",
                "addressability_method": "Reviewed Category positive prior",
            }

    # FUNDING POOL GATE: generic funding/distribution arrangements are not TAM by
    # themselves. They are only addressable when the public Description explicitly
    # identifies transformation, digital, data, cloud, cyber, SI or managed-technology
    # work. This prevents pharmaceutical/wholesale funding pools from entering TAM
    # through a broad Category. The Accenture-win guarantee is still applied later.
    funding_pool = any_pattern(description, [
        r"^funding pool$", r"\bfunding pool\b", r"\bpharmaceutical benefits?\b",
        r"\bcommunity pharmacy\b", r"\bmedicine(?:s)? supply\b",
        r"\bwholesale(?:r| distribution)?\b", r"\bdrug distribution\b",
    ])
    strong_digital_desc = any_pattern(description, [
        r"\bdigital transformation\b", r"\bsoftware (?:development|implementation|support|maintenance)\b",
        r"\bsystems? integration\b", r"\bapplication (?:development|implementation|support)\b",
        r"\bdata (?:platform|engineering|migration|analytics|governance)\b",
        r"\bcloud (?:migration|services?|hosting|platform|infrastructure)\b",
        r"\bcyber(?:security| security)?\b", r"\bmanaged (?:ict |it |technology )?services?\b",
        r"\bservice desk\b", r"\btechnology transformation\b",
    ])
    if funding_pool and not strong_digital_desc:
        return {
            "is_addressable": False,
            "addressability": "Not Addressable",
            "addressability_confidence": 98,
            "addressability_reason": f"Generic Health funding/distribution arrangement ('{funding_pool}') with no explicit digital, transformation or technology delivery subject in Description.",
            "addressability_method": "Health funding pool hard exclusion",
        }

    # Goods: only explicitly digital/software goods enter the TAM.
    if cat_type == "goods":
        digital = any_pattern(f"{category} {description}", ADDRESSABLE_GOODS_PATTERNS)
        physical = any_pattern(category, NON_ADDRESSABLE_GOODS_PATTERNS)
        if physical and not digital:
            return {
                "is_addressable": False,
                "addressability": "Not Addressable",
                "addressability_confidence": 99,
                "addressability_reason": f"Physical/medical Goods signal in Category: '{physical}'.",
                "addressability_method": "Goods hard exclusion",
            }
        if digital:
            return {
                "is_addressable": True,
                "addressability": "Addressable",
                "addressability_confidence": 92,
                "addressability_reason": f"Goods record contains explicit software/licence/cloud signal: '{digital}'.",
                "addressability_method": "Digital Goods exception",
            }
        return {
            "is_addressable": False,
            "addressability": "Not Addressable",
            "addressability_confidence": 90,
            "addressability_reason": "Category Type is Goods with no explicit software/licence/cloud evidence.",
            "addressability_method": "Ambiguous Goods exclusion",
        }

    # Services: hard subject exclusions first.
    hard_cat = any_pattern(category, NON_ADDRESSABLE_SERVICE_CATEGORY_PATTERNS)
    hard_desc = any_pattern(description, NON_ADDRESSABLE_DESCRIPTION_PATTERNS)
    if hard_cat:
        return {
            "is_addressable": False,
            "addressability": "Not Addressable",
            "addressability_confidence": 98,
            "addressability_reason": f"Category identifies a non-addressable Health/business service subject: '{hard_cat}'.",
            "addressability_method": "Non-addressable Category gate",
        }
    if hard_desc:
        return {
            "is_addressable": False,
            "addressability": "Not Addressable",
            "addressability_confidence": 96,
            "addressability_reason": f"Description identifies a non-addressable Health/business service subject: '{hard_desc}'.",
            "addressability_method": "Non-addressable Description gate",
        }

    cat_signal = any_pattern(category, ADDRESSABLE_SUBJECT_PATTERNS)
    desc_signal = any_pattern(description, ADDRESSABLE_SUBJECT_PATTERNS)
    generic_cat = any_pattern(category, GENERIC_CATEGORY_PATTERNS)

    if desc_signal and cat_signal:
        return {
            "is_addressable": True,
            "addressability": "Addressable",
            "addressability_confidence": 94,
            "addressability_reason": f"Description ('{desc_signal}') and Category ('{cat_signal}') both contain addressable evidence.",
            "addressability_method": "Description + Category evidence",
        }
    if desc_signal:
        return {
            "is_addressable": True,
            "addressability": "Addressable",
            "addressability_confidence": 88 if not generic_cat else 84,
            "addressability_reason": f"Description contains addressable evidence: '{desc_signal}'.",
            "addressability_method": "Description evidence",
        }
    if cat_signal and not generic_cat:
        return {
            "is_addressable": True,
            "addressability": "Addressable",
            "addressability_confidence": 80,
            "addressability_reason": f"Category contains addressable evidence: '{cat_signal}', with no contradictory Description evidence.",
            "addressability_method": "Category evidence",
        }

    return {
        "is_addressable": False,
        "addressability": "Not Addressable",
        "addressability_confidence": 60 if generic_cat else 75,
        "addressability_reason": "No sufficiently specific addressable transformation, digital, data, cloud, cyber, SI or managed-services evidence was found in Description/Category.",
        "addressability_method": "No recognised addressable evidence",
    }


def enforce_accenture_addressability(df: pd.DataFrame) -> pd.DataFrame:
    """Observed Accenture wins are always inside Accenture's addressable market.

    This override changes addressability only. Service Offering is still determined
    solely from Description + Category by the normal classifier below.
    """
    out = df.copy()
    acc = out.get("is_accenture", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    if not acc.any():
        return out

    out.loc[acc, "is_addressable"] = True
    out.loc[acc, "addressability"] = "Addressable"
    out.loc[acc, "addressability_confidence"] = 100
    out.loc[acc, "addressability_method"] = "Observed Accenture win guarantee"
    out.loc[acc, "addressability_reason"] = (
        "Observed Accenture award: by business definition, work won by Accenture is inside the Accenture-addressable market. "
        "Service Offering remains determined from Description and Category only."
    )
    out["accenture_addressability_override"] = acc
    return out


# =============================================================================
# 3. SERVICE OFFERING - ADDRESSABLE ROWS ONLY
# =============================================================================
# Health descriptions are expected to be more informative, so Description receives
# more weight than Category. Supplier is intentionally excluded from scoring.

CATEGORY_RULES: dict[str, list[tuple[str, float, str]]] = {
    "Strategy, Transformation & Advisory": [
        (r"\bmanagement advisory\b", 14, "management advisory"),
        (r"\bstrategic planning\b", 16, "strategic planning"),
        (r"\bproject management\b", 12, "project management"),
        (r"\bproject administration or planning\b", 12, "project administration / planning"),
        (r"\bconsulting services?\b", 11, "consulting"),
        (r"\bbusiness administration services?\b", 6, "business administration - generic"),
    ],
    "SI & Engineering": [
        (r"\bsoftware development\b", 17, "software development"),
        (r"\bapplication implementation\b", 15, "application implementation"),
        (r"\bsystems? integration\b", 17, "systems integration"),
        (r"^software$", 14, "software"),
        (r"\binformation technology services?\b", 11, "IT services"),
        (r"\bcomputer services?\b", 7, "computer services - generic"),
    ],
    "Data, AI & Automation": [
        (r"\bdata management\b", 17, "data management"),
        (r"\bdatabase\b", 15, "database"),
        (r"\banalytics\b", 17, "analytics"),
        (r"\bbusiness intelligence\b", 17, "business intelligence"),
        (r"\bstatistical services?\b", 15, "statistical services"),
        (r"\bartificial intelligence\b", 18, "AI"),
        (r"\bautomation\b", 15, "automation"),
    ],
    "Cloud Infrastructure & Cyber": [
        (r"\bcloud computing\b", 18, "cloud computing"),
        (r"\bhosting\b", 16, "hosting"),
        (r"\bnetwork services?\b", 15, "network services"),
        (r"\bcyber security\b", 18, "cyber security"),
        (r"\bcybersecurity\b", 18, "cybersecurity"),
        (r"\binformation security\b", 17, "information security"),
        (r"\btelecommunications services?\b", 13, "telecommunications services"),
    ],
    "Managed Services & Operations": [
        (r"\bsoftware maintenance and support\b", 18, "software maintenance/support"),
        (r"\bsystem administrators?\b", 17, "system administration"),
        (r"\bservice management\b", 16, "service management"),
        (r"\bcustomer relationship management\b", 15, "CRM"),
        (r"\blearning management\b", 14, "learning management"),
        (r"\bmaintenance or support fees\b", 15, "maintenance/support fees"),
    ],
}

DESCRIPTION_RULES: dict[str, list[tuple[str, float, str]]] = {
    "Strategy, Transformation & Advisory": [
        (r"\b(?:strategy|strategic advisory|advisory services?|business advisory)\b", 16, "strategy / advisory"),
        (r"\b(?:business transformation|digital transformation|operating model|organisational design|process improvement|change management)\b", 16, "transformation / change"),
        (r"\b(?:business case|feasibility study|options analysis|independent review|assurance|governance review)\b", 15, "assessment / assurance"),
        (r"\b(?:program(?:me)? management|project management|pmo|project management office|portfolio management|project support services?)\b", 14, "program / project management"),
        (r"\b(?:business analysis|requirements analysis|service design|policy design)\b", 14, "business / service analysis"),
    ],
    "SI & Engineering": [
        (r"\b(?:systems? integration|system integration|application integration|integration services?)\b", 18, "systems/application integration"),
        (r"\b(?:software development|application development|software engineering|devops|devsecops)\b", 17, "software/application engineering"),
        (r"\b(?:implementation|deployment|configuration)\b.*\b(?:system|software|application|platform|erp|crm)\b", 15, "system/platform implementation"),
        (r"\b(?:system|software|application|platform|erp|crm)\b.*\b(?:implementation|deployment|configuration)\b", 15, "system/platform implementation"),
        (r"\b(?:solution architecture|enterprise architecture|technical architecture|ict architecture)\b", 14, "architecture"),
        (r"\b(?:sap|s/4hana|oracle erp|servicenow|salesforce)\b.*\b(?:implement|deploy|migrat|integrat|configur)\w*\b", 16, "enterprise platform implementation"),
    ],
    "Data, AI & Automation": [
        (r"\bdata (?:platform|integration|migration|engineering|management|governance|services?|warehouse|lake|quality|remediation|catalogue|catalog)\b", 17, "explicit data work"),
        (r"\b(?:analytics|business intelligence|power bi|tableau|databricks|snowflake)\b", 17, "analytics / BI"),
        (r"\b(?:artificial intelligence|machine learning|generative ai|genai|automation|rpa|intelligent automation)\b", 18, "AI / automation"),
        (r"\b(?:predictive modelling|forecasting model|data science)\b", 16, "data science / modelling"),
    ],
    "Cloud Infrastructure & Cyber": [
        (r"\b(?:cloud services?|cloud migration|cloud hosting|cloud infrastructure|aws|amazon web services|azure|gcp)\b", 18, "cloud"),
        (r"\b(?:cybersecurity|cyber security|security operations|siem|soc|zero trust|identity and access management|iam)\b", 18, "cybersecurity"),
        (r"\b(?:network architecture|network engineering|network implementation|network infrastructure|network services?)\b", 15, "network infrastructure"),
        (r"\b(?:hosting services?|infrastructure hosting|data centre hosting|data center hosting)\b", 16, "hosting infrastructure"),
        (r"\b(?:information security|security accreditation|penetration testing|vulnerability assessment)\b", 16, "information security"),
    ],
    "Managed Services & Operations": [
        (r"\b(?:managed services?|application managed services?)\b", 20, "managed services"),
        (r"\b(?:service desk|help desk|end user support|desktop support)\b", 18, "service desk / end-user support"),
        (r"\b(?:application support|system support|software support|application maintenance|system maintenance|software maintenance)\b", 17, "application/system support"),
        (r"\b(?:ict|information technology|it)\s+(?:management and )?support services?\b", 18, "ICT support"),
        (r"\b(?:operations?|operational support|sustainment|service management)\b.*\b(?:ict|system|application|software|platform|technology)\b", 15, "technology operations / sustainment"),
        (r"\b(?:renewal|subscription|licen[cs]e renewal|maintenance renewal)\b", 15, "licence/subscription renewal"),
        (r"\b(?:servicenow|salesforce|workday|sap|oracle)\b.*\b(?:support|maintenance|operations?|managed|renewal)\b", 17, "enterprise platform run/support"),
    ],
}


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


def confidence_from_scores(winner_score: float, second_score: float, category_support: bool, description_support: bool) -> tuple[int, str]:
    margin = winner_score - second_score
    if winner_score <= 0:
        return 35, "Low"
    if description_support and category_support and winner_score >= 28 and margin >= 10:
        return 97, "Very high"
    if description_support and winner_score >= 20 and margin >= 8:
        return 92, "High"
    if winner_score >= 15 and margin >= 5:
        return 84, "Medium"
    if winner_score >= 10 and margin >= 3:
        return 72, "Medium"
    return 50, "Low"


def classify_service_offering(row: pd.Series) -> dict[str, object]:
    if not bool(row.get("is_addressable", False)):
        return {
            "service_offering": pd.NA,
            "Service Offering": pd.NA,
            "capability": pd.NA,
            "service_offering_confidence": 100,
            "service_offering_confidence_band": "Not applicable",
            "service_offering_second_choice": pd.NA,
            "service_offering_margin": pd.NA,
            "service_offering_method": "Not classified - outside addressable TAM",
            "service_offering_evidence": "",
            "category_score_breakdown": "{}",
            "description_score_breakdown": "{}",
            "service_offering_score_breakdown": "{}",
        }

    category = clean(row.get(CATEGORY, ""))
    description = clean(row.get(DESCRIPTION, ""))

    # Description is primary for Health; Category is supporting evidence.
    category_scores, category_evidence = collect_scores(category, CATEGORY_RULES, 1.00)
    description_scores, description_evidence = collect_scores(description, DESCRIPTION_RULES, 1.35)
    total = {o: category_scores[o] + description_scores[o] for o in FINAL_SERVICE_OFFERINGS}

    ranked = sorted(total.items(), key=lambda x: x[1], reverse=True)
    winner, winner_score = ranked[0]
    second, second_score = ranked[1]

    # Addressability can be clear even when SO evidence is sparse. Do not invent an
    # offering: leave it unresolved and push it to review.
    if winner_score <= 0:
        return {
            "service_offering": pd.NA,
            "Service Offering": pd.NA,
            "capability": pd.NA,
            "service_offering_confidence": 35,
            "service_offering_confidence_band": "Low",
            "service_offering_second_choice": pd.NA,
            "service_offering_margin": 0.0,
            "service_offering_method": "Addressable - SO unresolved",
            "service_offering_evidence": "Addressable evidence was found, but Description/Category did not provide enough detail to assign one of the five Service Offerings.",
            "category_score_breakdown": json.dumps(category_scores, sort_keys=True),
            "description_score_breakdown": json.dumps(description_scores, sort_keys=True),
            "service_offering_score_breakdown": json.dumps(total, sort_keys=True),
        }

    category_support = category_scores[winner] > 0
    description_support = description_scores[winner] > 0
    confidence, band = confidence_from_scores(winner_score, second_score, category_support, description_support)
    margin = winner_score - second_score

    evidence_parts = []
    if description_evidence[winner]:
        evidence_parts.append("Description: " + "; ".join(description_evidence[winner]))
    if category_evidence[winner]:
        evidence_parts.append("Category: " + "; ".join(category_evidence[winner]))

    if description_support and category_support:
        method = "Description + Category scoring"
    elif description_support:
        method = "Description-led scoring"
    else:
        method = "Category-led scoring"

    return {
        "service_offering": winner,
        "Service Offering": winner,
        "capability": winner,
        "service_offering_confidence": confidence,
        "service_offering_confidence_band": band,
        "service_offering_second_choice": second,
        "service_offering_margin": round(float(margin), 2),
        "service_offering_method": method,
        "service_offering_evidence": " | ".join(evidence_parts),
        "category_score_breakdown": json.dumps({k: round(v, 2) for k, v in category_scores.items()}, sort_keys=True),
        "description_score_breakdown": json.dumps({k: round(v, 2) for k, v in description_scores.items()}, sort_keys=True),
        "service_offering_score_breakdown": json.dumps({k: round(v, 2) for k, v in total.items()}, sort_keys=True),
    }



# =============================================================================
# SECONDARY SERVICE OFFERING CLASSIFIER
# =============================================================================
# This pass is intentionally conservative and ONLY runs for contracts that are:
#   1) already classified as addressable; and
#   2) unresolved by the primary Description + Category scorer.
#
# It does not use supplier identity and it never changes addressability.  Its job
# is to recover useful Service Offering coverage from common AusTender wording
# that is too broad for the primary regex set (for example "ICT Labour Hire",
# "Data services", "Software as a Service" and generic ICT support wording).
# Existing primary classifications are never overwritten.

SECONDARY_CATEGORY_RULES: list[tuple[str, str, int, str]] = [
    (r"\bdata services?\b|\bstatistics\b|\bdata processing or preparation services?\b|\binformation services?\b",
     "Data, AI & Automation", 80, "secondary category family: data / statistics / information services"),
    (r"\bsoftware or hardware engineering\b|\bcomputer programmers?\b|\bsoftware engineering\b",
     "SI & Engineering", 80, "secondary category family: software / engineering delivery"),
    (r"\bmanagement support services?\b|\bcorporate objectives or policy development\b|\bfeasibility studies?\b|\bnational planning services?\b",
     "Strategy, Transformation & Advisory", 78, "secondary category family: management / policy / planning"),
    (r"\binfrastructure as a service\b|\bplatform as a service\b|\bsoftware as a service\b|\bsaas\b|\biaas\b|\bpaas\b|\binternet services?\b|\bnetwork security\b",
     "Cloud Infrastructure & Cyber", 80, "secondary category family: cloud / internet / network security"),
]

SECONDARY_DESCRIPTION_RULES: list[tuple[str, str, int, str]] = [
    # Managed operations / run-state signals should be checked before generic ICT.
    (r"\b(?:maintenance|support|sustainment|operations?|service management|service desk|help desk|managed services?)\b.*\b(?:ict|it|system|application|software|platform|infrastructure|technology)\b|"
     r"\b(?:ict|it|system|application|software|platform|infrastructure|technology)\b.*\b(?:maintenance|support|sustainment|operations?|service management|service desk|help desk|managed services?)\b",
     "Managed Services & Operations", 84, "secondary description: technology support / operations"),
    (r"\b(?:licen[cs]e|subscription|renewal|reseller licence|reseller license)\b",
     "Managed Services & Operations", 80, "secondary description: licence / subscription / renewal"),
    (r"\binfrastructure services?\b.*\b(?:system|platform|record|environment)\b|\b(?:system|platform|record|environment)\b.*\binfrastructure services?\b",
     "Managed Services & Operations", 82, "secondary description: ongoing infrastructure services for a system / platform"),

    # Explicit cloud / cyber / infrastructure signals.
    (r"\b(?:cloud|azure|aws|amazon web services|google cloud|gcp|cyber|cybersecurity|security operations centre|security operations center|soc|identity and access management|iam|telecommunications?|network infrastructure|network security)\b",
     "Cloud Infrastructure & Cyber", 84, "secondary description: cloud / cyber / network"),

    # Explicit data / analytics signals.
    (r"\b(?:data|analytics?|analysis|reporting|business intelligence|statistics?|information management)\b.*\b(?:develop|development|collect|collection|manage|management|analyse|analyze|analysis|report|reporting|platform|capability|services?)\b|"
     r"\b(?:develop|development|collect|collection|manage|management|analyse|analyze|analysis|report|reporting)\b.*\b(?:data|analytics?|statistics?|information)\b",
     "Data, AI & Automation", 84, "secondary description: data / analytics delivery"),

    # Explicit system / software delivery signals.
    (r"\b(?:system integrator|systems integrator|software services?|digital suite|digital services?|digital solution|application services?|ict solution|technology solution|system solution)\b",
     "SI & Engineering", 80, "secondary description: system / software / digital delivery"),
    (r"\b(?:build|develop|development|design|deliver|delivery|implement|implementation|integrate|integration|migrate|migration)\b.*\b(?:system|software|application|platform|digital|ict|technology)\b|"
     r"\b(?:system|software|application|platform|digital|ict|technology)\b.*\b(?:build|develop|development|design|deliver|delivery|implement|implementation|integrate|integration|migrate|migration)\b",
     "SI & Engineering", 82, "secondary description: technology build / implementation"),

    # Advisory / project delivery wording.
    (r"\b(?:strategy|advisory|consulting|consultancy|business design|business analysis|project management|program management|programme management|delivery management|change management|transformation|pmo|assurance|review)\b",
     "Strategy, Transformation & Advisory", 78, "secondary description: advisory / transformation / project delivery"),
]


def classify_service_offering_secondary(row: pd.Series) -> dict[str, object] | None:
    """Return a fallback SO classification for an addressable unresolved row.

    Description is evaluated first. Category is a fallback. Supplier identity is
    deliberately excluded so this remains reproducible and portfolio-agnostic.
    """
    if not bool(row.get("is_addressable", False)):
        return None

    current = row.get("service_offering", pd.NA)
    if pd.notna(current) and str(current).strip():
        return None

    description = clean(row.get(DESCRIPTION, ""))
    category = clean(row.get(CATEGORY, ""))

    # Special handling for labour-hire categories: only classify when the public
    # description itself says what kind of work the labour supports.  Generic
    # "Labour Hire" remains unresolved rather than being forced into a segment.
    is_temp_personnel = bool(re.search(r"\btemporary personnel services?\b|\blabour hire\b", category, flags=re.I))
    if is_temp_personnel:
        labour_rules = [
            (r"\b(?:cyber|security|cloud|network|iam|identity and access)\b", "Cloud Infrastructure & Cyber", 76, "secondary labour-hire context: cloud / cyber / network"),
            (r"\b(?:data|analytics?|business intelligence|reporting|database)\b", "Data, AI & Automation", 76, "secondary labour-hire context: data / analytics"),
            (r"\b(?:project|program|programme|pmo|business analyst|change|transformation|advisory)\b", "Strategy, Transformation & Advisory", 74, "secondary labour-hire context: project / advisory"),
            (r"\b(?:support|service desk|help desk|operations?|sustainment|maintenance)\b", "Managed Services & Operations", 74, "secondary labour-hire context: support / operations"),
            (r"\b(?:ict|it|developer|programmer|software|application|system|technical|technology)\b", "SI & Engineering", 70, "secondary labour-hire context: generic ICT delivery"),
        ]
        for pattern, offering, confidence, reason in labour_rules:
            m = re.search(pattern, description, flags=re.I)
            if m:
                return {
                    "service_offering": offering,
                    "confidence": confidence,
                    "reason": f"{reason}; matched '{m.group(0)}' in Description.",
                    "evidence_source": "Description",
                }
        return None

    # Description wins in the secondary pass, mirroring the primary Health logic.
    for pattern, offering, confidence, reason in SECONDARY_DESCRIPTION_RULES:
        m = re.search(pattern, description, flags=re.I)
        if m:
            return {
                "service_offering": offering,
                "confidence": confidence,
                "reason": f"{reason}; matched '{m.group(0)}' in Description.",
                "evidence_source": "Description",
            }

    for pattern, offering, confidence, reason in SECONDARY_CATEGORY_RULES:
        m = re.search(pattern, category, flags=re.I)
        if m:
            return {
                "service_offering": offering,
                "confidence": confidence,
                "reason": f"{reason}; matched '{m.group(0)}' in Category.",
                "evidence_source": "Category",
            }

    # Generic ICT consultation is deliberately last.  If there is no more
    # specific technical/run/data signal, consultation maps to Strategy.
    if re.search(r"\binformation technology consultation services?\b|\bict consultation\b|\bit consultation\b", category, flags=re.I):
        return {
            "service_offering": "Strategy, Transformation & Advisory",
            "confidence": 72,
            "reason": "secondary category fallback: generic ICT consultation with no more specific Description signal.",
            "evidence_source": "Category",
        }

    return None


def apply_secondary_service_offering(df: pd.DataFrame) -> pd.DataFrame:
    """Fill only unresolved addressable SOs using the deterministic fallback pass."""
    out = df.copy()
    out["secondary_so_applied"] = False
    out["secondary_so_reason"] = ""
    out["secondary_so_evidence_source"] = ""

    unresolved_mask = (
        out["is_addressable"].fillna(False).astype(bool)
        & out["service_offering"].isna()
    )

    for idx in out.index[unresolved_mask]:
        result = classify_service_offering_secondary(out.loc[idx])
        if not result:
            continue
        offering = str(result["service_offering"])
        confidence = int(result["confidence"])
        reason = str(result["reason"])
        source = str(result["evidence_source"])

        out.at[idx, "service_offering"] = offering
        out.at[idx, "Service Offering"] = offering
        out.at[idx, "capability"] = offering
        out.at[idx, "service_offering_confidence"] = confidence
        out.at[idx, "service_offering_confidence_band"] = "Medium" if confidence >= 70 else "Low"
        out.at[idx, "service_offering_second_choice"] = pd.NA
        out.at[idx, "service_offering_margin"] = pd.NA
        out.at[idx, "service_offering_method"] = "Secondary Description/Category classifier"
        out.at[idx, "service_offering_evidence"] = reason
        out.at[idx, "secondary_so_applied"] = True
        out.at[idx, "secondary_so_reason"] = reason
        out.at[idx, "secondary_so_evidence_source"] = source

    return out

def apply_classifier(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    addr = pd.DataFrame([classify_addressability(row) for _, row in out.iterrows()], index=out.index)
    out = pd.concat([out, addr], axis=1)

    # Business invariant: every observed Accenture win is addressable. This happens
    # before SO classification so Accenture rows still receive SO from Description +
    # Category only, never from supplier identity.
    out = enforce_accenture_addressability(out)

    so = pd.DataFrame([classify_service_offering(row) for _, row in out.iterrows()], index=out.index)
    for col in so.columns:
        if col in out.columns:
            out = out.drop(columns=[col])
    out = pd.concat([out, so], axis=1)

    # Second-stage mapping for addressable rows the primary scorer could not map.
    # This improves coverage without manually assigning individual contracts.
    out = apply_secondary_service_offering(out)

    out["classification_review_required"] = (
        pd.to_numeric(out["addressability_confidence"], errors="coerce").fillna(0).lt(85)
        | (
            out["is_addressable"].fillna(False).astype(bool)
            & (
                pd.to_numeric(out["service_offering_confidence"], errors="coerce").fillna(0).lt(85)
                | out["service_offering"].isna()
            )
        )
    )
    return out


# =============================================================================
# 4. REGRESSION CHECKS
# =============================================================================

def run_regression_checks() -> None:
    cases = [
        ({CATEGORY_TYPE: "Services", CATEGORY: "Health research", DESCRIPTION: "Research project into health outcomes and data analysis"}, False, None),
        ({CATEGORY_TYPE: "Goods", CATEGORY: "Medical equipment and accessories", DESCRIPTION: "Supply of patient monitoring devices"}, False, None),
        ({CATEGORY_TYPE: "Goods", CATEGORY: "Software", DESCRIPTION: "Annual software licence subscription"}, True, "Managed Services & Operations"),
        ({CATEGORY_TYPE: "Services", CATEGORY: "Health services", DESCRIPTION: "Clinical health services"}, False, None),
        ({CATEGORY_TYPE: "Services", CATEGORY: "Management advisory services", DESCRIPTION: "Digital transformation strategy and operating model"}, True, "Strategy, Transformation & Advisory"),
        ({CATEGORY_TYPE: "Services", CATEGORY: "Computer services", DESCRIPTION: "Application development and systems integration"}, True, "SI & Engineering"),
        ({CATEGORY_TYPE: "Services", CATEGORY: "Data management services", DESCRIPTION: "Data platform and analytics services"}, True, "Data, AI & Automation"),
        ({CATEGORY_TYPE: "Services", CATEGORY: "Computer services", DESCRIPTION: "Cloud migration and Azure hosting services"}, True, "Cloud Infrastructure & Cyber"),
        ({CATEGORY_TYPE: "Services", CATEGORY: "Software maintenance and support", DESCRIPTION: "Application support and maintenance services"}, True, "Managed Services & Operations"),
        ({CATEGORY_TYPE: "Services", CATEGORY: "Professional engineering services", DESCRIPTION: "Biomedical equipment maintenance"}, False, None),
        ({CATEGORY_TYPE: "Services", CATEGORY: "Management advisory services", DESCRIPTION: "Funding Pool"}, False, None),
        ({CATEGORY_TYPE: "Services", CATEGORY: "Medical Equipment and Accessories and Supplies", DESCRIPTION: "Supply of Vaccines and Essential Medical Products"}, False, None),
        ({CATEGORY_TYPE: "Services", CATEGORY: "Management advisory services", DESCRIPTION: "COVID-19 Vaccine Administration Program Management"}, False, None),
        ({CATEGORY_TYPE: "Services", CATEGORY: "Disease prevention and control", DESCRIPTION: "Vaccine rollout support services"}, False, None),
        ({CATEGORY_TYPE: "Goods", CATEGORY: "Medical Equipment and Accessories and Supplies", DESCRIPTION: "Connected patient monitoring equipment with software"}, False, None),
        ({CATEGORY_TYPE: "Goods", CATEGORY: "Pharmaceutical products", DESCRIPTION: "Medicine supply and distribution"}, False, None),
        ({CATEGORY_TYPE: "Services", CATEGORY: "Business intelligence consulting services", DESCRIPTION: "BI delivery services"}, True, "Data, AI & Automation"),
        ({CATEGORY_TYPE: "Services", CATEGORY: "Medical science research and experimentation", DESCRIPTION: "Data analysis for clinical study"}, False, None),
        ({CATEGORY_TYPE: "Services", CATEGORY: "Computer services", DESCRIPTION: "General support arrangement"}, False, None),
    ]

    for payload, expected_addr, expected_so in cases:
        row = pd.Series(payload)
        addr = classify_addressability(row)
        row2 = row.copy()
        row2["is_addressable"] = addr["is_addressable"]
        so = classify_service_offering(row2)
        actual_so = so["service_offering"]
        if pd.isna(actual_so):
            actual_so = None
        if bool(addr["is_addressable"]) != expected_addr or actual_so != expected_so:
            raise RuntimeError(
                f"Regression check failed for {payload}: expected ({expected_addr}, {expected_so}), "
                f"got ({addr['is_addressable']}, {actual_so})"
            )


# =============================================================================
# 5. OUTPUTS / AUDIT
# =============================================================================

def export_audits(df: pd.DataFrame, audit_dir: Path) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    value = pd.to_numeric(df.get(VALUE, 0), errors="coerce").fillna(0)
    data = df.copy()
    data[VALUE] = value

    audit_cols = [c for c in [
        CN_ID, RAW_AGENCY_COL, AGENCY, AGENCY_GROUP_COL, CATEGORY_TYPE, CATEGORY, DESCRIPTION,
        SUPPLIER_NAME, SUPPLIER_GROUP_COL, "is_accenture", VALUE,
        "is_addressable", "addressability", "addressability_confidence",
        "addressability_method", "addressability_reason",
        "service_offering", "service_offering_confidence",
        "service_offering_confidence_band", "service_offering_method",
        "service_offering_second_choice", "service_offering_margin",
        "service_offering_evidence", "category_score_breakdown",
        "description_score_breakdown", "service_offering_score_breakdown",
        "secondary_so_applied", "secondary_so_reason", "secondary_so_evidence_source",
        "accenture_addressability_override", "classification_review_required",
    ] if c in data.columns]

    data[audit_cols].sort_values(VALUE, ascending=False).to_csv(
        audit_dir / "health_classification_audit.csv", index=False
    )
    data.loc[data["classification_review_required"], audit_cols].sort_values(VALUE, ascending=False).to_csv(
        audit_dir / "health_classification_review_queue.csv", index=False
    )

    # Separate automated-SO diagnostics. These are not manual override files; they
    # simply make the second-stage rules transparent and make residual gaps easy
    # to quantify after each rebuild.
    if "secondary_so_applied" in data.columns:
        data.loc[data["secondary_so_applied"].fillna(False).astype(bool), audit_cols]\
            .sort_values(VALUE, ascending=False)\
            .to_csv(audit_dir / "health_secondary_so_mappings.csv", index=False)

    unresolved_so = (
        data["is_addressable"].fillna(False).astype(bool)
        & data["service_offering"].isna()
    )
    data.loc[unresolved_so, audit_cols].sort_values(VALUE, ascending=False).to_csv(
        audit_dir / "health_unresolved_so_queue.csv", index=False
    )

    # Accenture reconciliation by raw source supplier name. This makes it obvious
    # whether historical legal-name variants have all been consolidated.
    if "is_accenture" in data.columns:
        acc_rows = data[data["is_accenture"].fillna(False).astype(bool)].copy()
        acc_name_audit = (
            acc_rows.groupby(SUPPLIER_NAME, dropna=False)
            .agg(contracts=(CN_ID, "count") if CN_ID in acc_rows.columns else (VALUE, "size"),
                 total_value=(VALUE, "sum"))
            .reset_index()
            .sort_values("total_value", ascending=False)
        )
        acc_name_audit.to_csv(audit_dir / "accenture_supplier_name_audit.csv", index=False)

    summary = data.groupby(["addressability", "service_offering"], dropna=False).agg(
        contracts=(CN_ID, "count") if CN_ID in data.columns else (VALUE, "size"),
        total_value=(VALUE, "sum"),
    ).reset_index().sort_values("total_value", ascending=False)
    summary.to_csv(audit_dir / "health_addressability_so_summary.csv", index=False)


# =============================================================================
# 6. MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Health portfolio scope + addressability + Service Offering dataset.")
    p.add_argument("--input", default="fullDataset.parquet", help="Whole-of-government AusTender CSV or parquet")
    p.add_argument("--output", default="master_output/health_portfolio_contracts.parquet")
    p.add_argument("--audit-dir", default="master_output/audit")
    return p.parse_args()


def main() -> None:
    run_regression_checks()
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    audit_dir = Path(args.audit_dir)

    # Convenience fallback for environments that still keep the same extract as CSV.
    if not input_path.exists() and input_path.name == "fullDataset.parquet":
        csv_fallback = input_path.with_suffix(".csv")
        if csv_fallback.exists():
            input_path = csv_fallback

    print(f"1/5 Loading source: {input_path}")
    raw = load_data(input_path)
    required = {AGENCY, DESCRIPTION, CATEGORY, CATEGORY_TYPE, VALUE}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError("Missing required source columns: " + ", ".join(missing))

    raw[VALUE] = pd.to_numeric(raw[VALUE], errors="coerce").fillna(0)

    print("2/5 Applying exact Health agency scope + agency normalisation")
    health = apply_health_scope(raw)
    print(f"Health rows: {len(health):,}; Health procurement: A${health[VALUE].sum()/1e9:,.2f}B")

    print("3/5 Normalising supplier names")
    health = add_supplier_grouping(health)

    # Reconcile the filtered population BEFORE addressability. These are the
    # headline source-data checks and should match a manual Excel filter.
    scoped_total = float(health[VALUE].sum())
    scoped_acc = float(health.loc[health["is_accenture"], VALUE].sum())
    print(f"Source-filter reconciliation - Health procurement: A${scoped_total/1e9:,.3f}B")
    print(f"Source-filter reconciliation - Accenture wins: A${scoped_acc/1e9:,.3f}B")

    print("4/5 Applying addressability then Service Offering")
    health = apply_classifier(health)

    # Enforce the central invariant requested for Health.
    invalid = (~health["is_addressable"].fillna(False).astype(bool)) & health["service_offering"].notna()
    if invalid.any():
        raise RuntimeError(f"Invariant failed: {int(invalid.sum())} non-addressable rows received a Service Offering.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep the dashboard master lean: original source columns plus only the
    # derived fields the dashboards actually need. Detailed classifier workings
    # remain in the audit CSV instead of bloating every contract row.
    source_cols = [c for c in raw.columns if c in health.columns]
    master_derived = [
        RAW_AGENCY_COL, SUPPLIER_GROUP_COL, "is_accenture",
        "is_addressable", "addressability", "addressability_confidence",
        "addressability_reason", "Service Offering",
        "service_offering_confidence", "secondary_so_applied",
        "classification_review_required",
    ]
    master_cols = source_cols + [c for c in master_derived if c in health.columns and c not in source_cols]
    master = health[master_cols].copy()
    master.to_parquet(output_path, index=False)
    export_audits(health, audit_dir)

    addr = health[health["is_addressable"].fillna(False).astype(bool)]
    tam = addr[VALUE].sum()
    unresolved_mask = addr["service_offering"].isna()
    unresolved = int(unresolved_mask.sum())
    unresolved_value = float(addr.loc[unresolved_mask, VALUE].sum())
    mapped_rows = int((~unresolved_mask).sum())
    mapped_value = float(addr.loc[~unresolved_mask, VALUE].sum())
    secondary_rows = int(addr.get("secondary_so_applied", False).fillna(False).astype(bool).sum())
    secondary_value = float(addr.loc[addr.get("secondary_so_applied", False).fillna(False).astype(bool), VALUE].sum())
    acc = health[health.get("is_accenture", False).fillna(False).astype(bool)]
    acc_value = acc[VALUE].sum()
    acc_addr_value = acc.loc[acc["is_addressable"].fillna(False).astype(bool), VALUE].sum()

    # Hard reconciliation invariant: addressability must never remove an Accenture win.
    if abs(float(acc_value) - float(acc_addr_value)) > 0.01:
        raise RuntimeError(
            f"Accenture addressability invariant failed: observed A${acc_value/1e9:,.3f}B "
            f"vs addressable A${acc_addr_value/1e9:,.3f}B."
        )

    print("5/5 Complete")
    print(f"Output: {output_path}")
    print(f"Total Health procurement: A${health[VALUE].sum()/1e9:,.2f}B")
    print(f"Addressable TAM: A${tam/1e9:,.2f}B ({tam/health[VALUE].sum()*100:.1f}% of Health procurement)")
    print(f"Addressable rows: {len(addr):,}")
    print(f"Addressable rows mapped to SO: {mapped_rows:,}/{len(addr):,} ({mapped_rows/len(addr)*100 if len(addr) else 0:.1f}%)")
    print(f"Addressable value mapped to SO: A${mapped_value/1e9:,.2f}B/A${tam/1e9:,.2f}B ({mapped_value/tam*100 if tam else 0:.1f}%)")
    print(f"Secondary classifier mapped: {secondary_rows:,} rows / A${secondary_value/1e9:,.2f}B")
    print(f"Addressable rows with unresolved SO: {unresolved:,} / A${unresolved_value/1e9:,.2f}B")
    print(f"Accenture observed wins: A${acc_value/1e9:,.3f}B")
    print(f"Accenture wins retained as addressable: A${acc_addr_value/1e9:,.3f}B")
    if not acc.empty and SUPPLIER_NAME in acc.columns:
        print("Accenture raw supplier-name breakdown:")
        acc_breakdown = (
            acc.groupby(SUPPLIER_NAME, dropna=False)[VALUE]
            .sum()
            .sort_values(ascending=False)
        )
        for supplier_name, supplier_value in acc_breakdown.items():
            print(f"  {supplier_name}: A${supplier_value/1e9:,.3f}B")
    print(f"Accenture supplier audit: {audit_dir / 'accenture_supplier_name_audit.csv'}")
    print(f"Audit: {audit_dir / 'health_classification_audit.csv'}")
    print(f"Review queue: {audit_dir / 'health_classification_review_queue.csv'}")
    print(f"Secondary SO mappings: {audit_dir / 'health_secondary_so_mappings.csv'}")
    print(f"Unresolved SO queue: {audit_dir / 'health_unresolved_so_queue.csv'}")


if __name__ == "__main__":
    main()
