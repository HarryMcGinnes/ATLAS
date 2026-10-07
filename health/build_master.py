from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


# ============================================================
# CANONICAL COLUMNS
# ============================================================

CN_ID = "CN ID"
AGENCY = "Agency"
RAW_AGENCY_COL = "Agency Raw"

DESCRIPTION = "Description"
CATEGORY = "Category"
CATEGORY_TYPE = "Category Type"

VALUE = "Value"
SUPPLIER_NAME = "Supplier Name"

SUPPLIER_GROUP_COL = "supplier_group"

FINAL_SERVICE_OFFERINGS = [
    "Strategy, Transformation & Advisory",
    "SI & Engineering",
    "Data, AI & Automation",
    "Cloud Infrastructure & Cyber",
    "Managed Services & Operations",
]


# ============================================================
# SUPPLIER CONSOLIDATION
# ============================================================

CANONICAL_SUPPLIER_FAMILIES = [
    (
        "Accenture",
        [
            r"\bACCENTURE\b",
        ],
    ),
    (
        "Deloitte",
        [
            r"\bDELOITTE\b",
        ],
    ),
    (
        "EY",
        [
            r"\bERNST\s*(?:&|AND)\s*YOUNG\b",
            r"^EY(?:\s|$)",
        ],
    ),
    (
        "KPMG",
        [
            r"\bKPMG\b",
        ],
    ),
    (
        "PwC",
        [
            r"\bPRICEWATERHOUSECOOPERS\b",
            r"\bPRICEWATERHOUSE\s+COOPERS\b",
            r"\bPWC\b",
        ],
    ),
    (
        "IBM",
        [
            r"\bIBM\b",
            r"\bINTERNATIONAL\s+BUSINESS\s+MACHINES\b",
        ],
    ),
    (
        "Fujitsu",
        [
            r"\bFUJITSU\b",
        ],
    ),
    (
        "DXC Technology",
        [
            r"\bDXC\b",
        ],
    ),
    (
        "Capgemini",
        [
            r"\bCAPGEMINI\b",
        ],
    ),
    (
        "Microsoft",
        [
            r"\bMICROSOFT\b",
        ],
    ),
    (
        "Amazon Web Services",
        [
            r"\bAMAZON\s+WEB\s+SERVICES\b",
            r"^AWS(?:\s|$)",
        ],
    ),
    (
        "Oracle",
        [
            r"\bORACLE\b",
        ],
    ),
    (
        "SAP",
        [
            r"^SAP(?:\s|$)",
            r"\bSAP AUSTRALIA\b",
        ],
    ),
    (
        "Data#3",
        [
            r"\bDATA#?3\b",
        ],
    ),
    (
        "Telstra",
        [
            r"\bTELSTRA\b",
        ],
    ),
    (
        "Optus",
        [
            r"\bOPTUS\b",
        ],
    ),
    (
        "Salesforce",
        [
            r"\bSALESFORCE\b",
        ],
    ),
    (
        "ServiceNow",
        [
            r"\bSERVICENOW\b",
        ],
    ),
]


# ============================================================
# ADDRESSABILITY
# ============================================================

HARD_HEALTH_PRODUCT_PATTERNS = [
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
    r"\bpatient care and treatment products?\b",
    r"\bdisease prevention and control\b",
]


RESEARCH_PATTERNS = [
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
]


NON_ADDRESSABLE_SERVICE_PATTERNS = [
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
    r"\baccommodation\b",
    r"\butilities\b",
    r"\belectricity\b",
    r"\brecruitment\b",
    r"\bpersonnel recruitment\b",
    r"\blabour hire\b",
    r"\blegal services?\b",
]


ADDRESSABLE_SUBJECT_PATTERNS = [
    r"\bstrategy\b",
    r"\bstrategic\b",
    r"\badvisory\b",
    r"\bconsult(?:ing|ancy)\b",
    r"\btransformation\b",
    r"\boperating model\b",
    r"\borganisational design\b",
    r"\bprocess improvement\b",
    r"\bchange management\b",
    r"\bbusiness case\b",
    r"\bprogram(?:me)? management\b",
    r"\bproject management\b",
    r"\bpmo\b",
    r"\bbusiness analysis\b",
    r"\bassurance\b",
    r"\bgovernance\b",
    r"\bdata\b",
    r"\banalytics\b",
    r"\bbusiness intelligence\b",
    r"\bpower bi\b",
    r"\bartificial intelligence\b",
    r"\bmachine learning\b",
    r"\bautomation\b",
    r"\brpa\b",
    r"\bcloud\b",
    r"\bhosting\b",
    r"\bcyber(?:security)?\b",
    r"\binformation security\b",
    r"\bnetwork(?:ing)?\b",
    r"\bidentity and access management\b",
    r"\bsiem\b",
    r"\bsoc\b",
    r"\bsoftware\b",
    r"\bapplication\b",
    r"\bsystems? integration\b",
    r"\bimplementation\b",
    r"\bdigital\b",
    r"\bict\b",
    r"\binformation technology\b",
    r"\berp\b",
    r"\bsap\b",
    r"\bservicenow\b",
    r"\bsalesforce\b",
    r"\barchitecture\b",
    r"\bdevops\b",
    r"\bmanaged services?\b",
    r"\bservice desk\b",
    r"\bhelp desk\b",
    r"\bsupport services?\b",
    r"\bapplication support\b",
    r"\bsystem support\b",
    r"\bsoftware support\b",
    r"\bmaintenance and support\b",
    r"\boperations?\b",
    r"\bsustainment\b",
    r"\blicen[cs](?:e|es|ing)\b",
    r"\bsubscription(?:s)?\b",
]


ADDRESSABLE_GOODS_PATTERNS = [
    r"\bsoftware\b",
    r"\blicen[cs](?:e|es|ing)\b",
    r"\bsubscription(?:s)?\b",
    r"\bsaas\b",
    r"\bsoftware as a service\b",
    r"\bcloud\b",
]


# ============================================================
# SERVICE OFFERING RULES
# ============================================================

SERVICE_OFFERING_RULES = {
    "Managed Services & Operations": [
        r"\bmanaged services?\b",
        r"\bservice desk\b",
        r"\bhelp desk\b",
        r"\bapplication support\b",
        r"\bsystem support\b",
        r"\bsoftware support\b",
        r"\bapplication maintenance\b",
        r"\bsystem maintenance\b",
        r"\bsoftware maintenance\b",
        r"\bmaintenance and support\b",
        r"\bsustainment\b",
        r"\bservice management\b",
        r"\blicen[cs](?:e|es|ing)\b",
        r"\bsubscription(?:s)?\b",
        r"\brenewal\b",
    ],

    "Cloud Infrastructure & Cyber": [
        r"\bcloud\b",
        r"\bazure\b",
        r"\baws\b",
        r"\bamazon web services\b",
        r"\bgcp\b",
        r"\bgoogle cloud\b",
        r"\bcyber(?:security)?\b",
        r"\bsecurity operations\b",
        r"\bidentity and access management\b",
        r"\biam\b",
        r"\bnetwork security\b",
        r"\bnetwork infrastructure\b",
        r"\bhosting\b",
    ],

    "Data, AI & Automation": [
        r"\bdata platform\b",
        r"\bdata engineering\b",
        r"\bdata migration\b",
        r"\bdata analytics\b",
        r"\bdata governance\b",
        r"\banalytics\b",
        r"\bbusiness intelligence\b",
        r"\bpower bi\b",
        r"\btableau\b",
        r"\bartificial intelligence\b",
        r"\bmachine learning\b",
        r"\bgenerative ai\b",
        r"\bgenai\b",
        r"\bautomation\b",
        r"\brpa\b",
        r"\bdata science\b",
    ],

    "SI & Engineering": [
        r"\bsystems? integration\b",
        r"\bapplication integration\b",
        r"\bsoftware development\b",
        r"\bapplication development\b",
        r"\bsoftware engineering\b",
        r"\bdevops\b",
        r"\bdevsecops\b",
        r"\bsystem implementation\b",
        r"\bsoftware implementation\b",
        r"\bapplication implementation\b",
        r"\bplatform implementation\b",
        r"\bsolution architecture\b",
        r"\benterprise architecture\b",
        r"\btechnical architecture\b",
    ],

    "Strategy, Transformation & Advisory": [
        r"\bstrategy\b",
        r"\bstrategic advisory\b",
        r"\badvisory services?\b",
        r"\bbusiness advisory\b",
        r"\bbusiness transformation\b",
        r"\bdigital transformation\b",
        r"\boperating model\b",
        r"\borganisational design\b",
        r"\bprocess improvement\b",
        r"\bchange management\b",
        r"\bbusiness case\b",
        r"\bfeasibility study\b",
        r"\boptions analysis\b",
        r"\bindependent review\b",
        r"\bassurance\b",
        r"\bgovernance review\b",
        r"\bprogram(?:me)? management\b",
        r"\bproject management\b",
        r"\bpmo\b",
        r"\bportfolio management\b",
        r"\bbusiness analysis\b",
        r"\brequirements analysis\b",
        r"\bservice design\b",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the canonical ATLAS Health master dataset."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "health/data/"
            "health_contracts_raw.parquet"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "health/data/processed"
        ),
    )

    parser.add_argument(
        "--audit-dir",
        default=(
            "audits/austender/"
            "health_master"
        ),
    )

    return parser.parse_args()


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    text = (
        str(value)
        .lower()
        .replace("&", " and ")
    )

    text = re.sub(
        r"[^a-z0-9+#./ -]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def normalise_supplier(
    value: object,
) -> str:
    text = (
        str(value or "")
        .upper()
        .replace("&", " AND ")
    )

    text = re.sub(
        r"[^A-Z0-9#]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def canonical_supplier_name(
    value: object,
) -> str:
    raw = str(
        value or ""
    ).strip()

    normalised = normalise_supplier(
        raw
    )

    for (
        canonical,
        patterns,
    ) in CANONICAL_SUPPLIER_FAMILIES:

        for pattern in patterns:

            if re.search(
                pattern,
                normalised,
                flags=re.IGNORECASE,
            ):
                return canonical

    return (
        raw
        if raw
        else "Unknown Supplier"
    )


def any_pattern(
    text: str,
    patterns: list[str],
) -> str | None:

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(0)

    return None


# ============================================================
# SUPPLIER GROUPING
# ============================================================

def add_supplier_grouping(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    if SUPPLIER_NAME not in out.columns:
        raise RuntimeError(
            f"Health raw dataset is missing "
            f"{SUPPLIER_NAME}"
        )

    raw_supplier = (
        out[
            SUPPLIER_NAME
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    out[
        SUPPLIER_GROUP_COL
    ] = raw_supplier.map(
        canonical_supplier_name
    )

    out[
        "is_accenture"
    ] = out[
        SUPPLIER_GROUP_COL
    ].eq(
        "Accenture"
    )

    return out


# ============================================================
# ADDRESSABILITY
# ============================================================

def classify_addressability(
    row: pd.Series,
) -> dict[str, object]:

    category = clean(
        row.get(
            CATEGORY,
            "",
        )
    )

    description = clean(
        row.get(
            DESCRIPTION,
            "",
        )
    )

    category_type = clean(
        row.get(
            CATEGORY_TYPE,
            "",
        )
    )

    combined = (
        f"{category} {description}"
    )

    research = any_pattern(
        combined,
        RESEARCH_PATTERNS,
    )

    if research:
        return {
            "is_addressable": False,
            "addressability": (
                "Not Addressable"
            ),
            "addressability_confidence": 99,
            "addressability_reason": (
                "Research project/service is outside "
                f"Accenture TAM: '{research}'."
            ),
            "addressability_method": (
                "Research hard exclusion"
            ),
        }

    health_product = any_pattern(
        combined,
        HARD_HEALTH_PRODUCT_PATTERNS,
    )

    if health_product:
        return {
            "is_addressable": False,
            "addressability": (
                "Not Addressable"
            ),
            "addressability_confidence": 99,
            "addressability_reason": (
                "Health product/clinical exclusion: "
                f"'{health_product}'."
            ),
            "addressability_method": (
                "Health product / clinical "
                "hard exclusion"
            ),
        }

    if category_type == "goods":

        digital = any_pattern(
            combined,
            ADDRESSABLE_GOODS_PATTERNS,
        )

        if digital:
            return {
                "is_addressable": True,
                "addressability": (
                    "Addressable"
                ),
                "addressability_confidence": 92,
                "addressability_reason": (
                    "Goods record contains explicit "
                    "software/licence/cloud evidence: "
                    f"'{digital}'."
                ),
                "addressability_method": (
                    "Digital Goods exception"
                ),
            }

        return {
            "is_addressable": False,
            "addressability": (
                "Not Addressable"
            ),
            "addressability_confidence": 90,
            "addressability_reason": (
                "Category Type is Goods with no "
                "explicit software/licence/cloud "
                "evidence."
            ),
            "addressability_method": (
                "Goods exclusion"
            ),
        }

    excluded_service = any_pattern(
        combined,
        NON_ADDRESSABLE_SERVICE_PATTERNS,
    )

    if excluded_service:
        return {
            "is_addressable": False,
            "addressability": (
                "Not Addressable"
            ),
            "addressability_confidence": 96,
            "addressability_reason": (
                "Non-addressable Health/business "
                f"service subject: '{excluded_service}'."
            ),
            "addressability_method": (
                "Non-addressable service gate"
            ),
        }

    evidence = any_pattern(
        combined,
        ADDRESSABLE_SUBJECT_PATTERNS,
    )

    if evidence:
        return {
            "is_addressable": True,
            "addressability": (
                "Addressable"
            ),
            "addressability_confidence": 88,
            "addressability_reason": (
                "Description/Category contains "
                "addressable transformation or "
                "technology evidence: "
                f"'{evidence}'."
            ),
            "addressability_method": (
                "Description / Category evidence"
            ),
        }

    return {
        "is_addressable": False,
        "addressability": (
            "Not Addressable"
        ),
        "addressability_confidence": 70,
        "addressability_reason": (
            "No sufficiently specific "
            "transformation, digital, data, cloud, "
            "cyber, SI or managed-services evidence "
            "was found."
        ),
        "addressability_method": (
            "No recognised addressable evidence"
        ),
    }


def enforce_accenture_addressability(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    acc = (
        out[
            "is_accenture"
        ]
        .fillna(False)
        .astype(bool)
    )

    if not acc.any():
        return out

    out.loc[
        acc,
        "is_addressable",
    ] = True

    out.loc[
        acc,
        "addressability",
    ] = "Addressable"

    out.loc[
        acc,
        "addressability_confidence",
    ] = 100

    out.loc[
        acc,
        "addressability_method",
    ] = (
        "Observed Accenture win guarantee"
    )

    out.loc[
        acc,
        "addressability_reason",
    ] = (
        "Observed Accenture award: work won by "
        "Accenture is inside the "
        "Accenture-addressable market."
    )

    out[
        "accenture_addressability_override"
    ] = acc

    return out


# ============================================================
# SERVICE OFFERING
# ============================================================

def classify_service_offering(
    row: pd.Series,
) -> dict[str, object]:

    if not bool(
        row.get(
            "is_addressable",
            False,
        )
    ):
        return {
            "service_offering": pd.NA,
            "Service Offering": pd.NA,
            "capability": pd.NA,
            "service_offering_confidence": 100,
            "service_offering_confidence_band": (
                "Not applicable"
            ),
            "service_offering_method": (
                "Outside addressable TAM"
            ),
            "service_offering_evidence": "",
        }

    description = clean(
        row.get(
            DESCRIPTION,
            "",
        )
    )

    category = clean(
        row.get(
            CATEGORY,
            "",
        )
    )

    combined = (
        f"{description} {category}"
    )

    scores: dict[str, int] = {
        offering: 0
        for offering in FINAL_SERVICE_OFFERINGS
    }

    evidence: dict[
        str,
        list[str],
    ] = {
        offering: []
        for offering in FINAL_SERVICE_OFFERINGS
    }

    for (
        offering,
        patterns,
    ) in SERVICE_OFFERING_RULES.items():

        for pattern in patterns:

            match = re.search(
                pattern,
                description,
                flags=re.IGNORECASE,
            )

            if match:
                scores[
                    offering
                ] += 3

                evidence[
                    offering
                ].append(
                    "Description: "
                    + match.group(0)
                )

            match = re.search(
                pattern,
                category,
                flags=re.IGNORECASE,
            )

            if match:
                scores[
                    offering
                ] += 1

                evidence[
                    offering
                ].append(
                    "Category: "
                    + match.group(0)
                )

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    winner = ranked[0][0]
    winner_score = ranked[0][1]

    second_score = (
        ranked[1][1]
        if len(ranked) > 1
        else 0
    )

    if winner_score <= 0:
        return {
            "service_offering": pd.NA,
            "Service Offering": pd.NA,
            "capability": pd.NA,
            "service_offering_confidence": 35,
            "service_offering_confidence_band": (
                "Low"
            ),
            "service_offering_method": (
                "Addressable - SO unresolved"
            ),
            "service_offering_evidence": (
                "Addressable evidence was found, "
                "but Description/Category did not "
                "provide enough information to "
                "assign a Service Offering."
            ),
        }

    margin = (
        winner_score
        - second_score
    )

    if (
        winner_score >= 6
        and margin >= 3
    ):
        confidence = 94
        band = "High"

    elif winner_score >= 3:
        confidence = 84
        band = "Medium"

    else:
        confidence = 70
        band = "Medium"

    return {
        "service_offering": winner,
        "Service Offering": winner,
        "capability": winner,
        "service_offering_confidence": (
            confidence
        ),
        "service_offering_confidence_band": (
            band
        ),
        "service_offering_method": (
            "Description-led classification"
        ),
        "service_offering_evidence": (
            " | ".join(
                evidence[
                    winner
                ]
            )
        ),
    }


# ============================================================
# REGRESSION CHECKS
# ============================================================

def run_regression_checks() -> None:

    cases = [
        (
            {
                CATEGORY_TYPE: "Services",
                CATEGORY: "Health research",
                DESCRIPTION: (
                    "Research project into "
                    "health outcomes"
                ),
            },
            False,
        ),
        (
            {
                CATEGORY_TYPE: "Goods",
                CATEGORY: "Medical equipment",
                DESCRIPTION: (
                    "Supply patient monitors"
                ),
            },
            False,
        ),
        (
            {
                CATEGORY_TYPE: "Goods",
                CATEGORY: "Software",
                DESCRIPTION: (
                    "Annual software licence "
                    "subscription"
                ),
            },
            True,
        ),
        (
            {
                CATEGORY_TYPE: "Services",
                CATEGORY: (
                    "Management advisory services"
                ),
                DESCRIPTION: (
                    "Digital transformation "
                    "strategy"
                ),
            },
            True,
        ),
        (
            {
                CATEGORY_TYPE: "Services",
                CATEGORY: "Computer services",
                DESCRIPTION: (
                    "Application development and "
                    "systems integration"
                ),
            },
            True,
        ),
    ]

    for (
        payload,
        expected,
    ) in cases:

        result = (
            classify_addressability(
                pd.Series(
                    payload
                )
            )
        )

        if bool(
            result[
                "is_addressable"
            ]
        ) != expected:

            raise RuntimeError(
                "Health addressability regression "
                f"check failed for: {payload}"
            )


# ============================================================
# AUDITS
# ============================================================

def export_audits(
    df: pd.DataFrame,
    audit_dir: Path,
) -> None:

    audit_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_cols = [
        column
        for column in [
            CN_ID,
            RAW_AGENCY_COL,
            AGENCY,
            CATEGORY_TYPE,
            CATEGORY,
            DESCRIPTION,
            SUPPLIER_NAME,
            SUPPLIER_GROUP_COL,
            "is_accenture",
            VALUE,
            "is_addressable",
            "addressability",
            "addressability_confidence",
            "addressability_method",
            "addressability_reason",
            "Service Offering",
            "service_offering_confidence",
            "service_offering_confidence_band",
            "service_offering_method",
            "service_offering_evidence",
            "accenture_addressability_override",
            "classification_review_required",
        ]
        if column in df.columns
    ]

    df[
        audit_cols
    ].sort_values(
        VALUE,
        ascending=False,
    ).to_csv(
        audit_dir
        / "health_classification_audit.csv",
        index=False,
    )

    review = df[
        df[
            "classification_review_required"
        ]
    ]

    review[
        audit_cols
    ].sort_values(
        VALUE,
        ascending=False,
    ).to_csv(
        audit_dir
        / "health_classification_review_queue.csv",
        index=False,
    )

    unresolved = df[
        df[
            "is_addressable"
        ].fillna(
            False
        ).astype(bool)
        & df[
            "Service Offering"
        ].isna()
    ]

    unresolved[
        audit_cols
    ].sort_values(
        VALUE,
        ascending=False,
    ).to_csv(
        audit_dir
        / "health_unresolved_so_queue.csv",
        index=False,
    )

    acc = df[
        df[
            "is_accenture"
        ].fillna(
            False
        ).astype(bool)
    ]

    if not acc.empty:

        acc_summary = (
            acc
            .groupby(
                SUPPLIER_NAME,
                dropna=False,
            )
            .agg(
                contracts=(
                    CN_ID,
                    "nunique",
                )
                if CN_ID in acc.columns
                else (
                    VALUE,
                    "size",
                ),
                total_value=(
                    VALUE,
                    "sum",
                ),
            )
            .reset_index()
            .sort_values(
                "total_value",
                ascending=False,
            )
        )

        acc_summary.to_csv(
            audit_dir
            / "accenture_supplier_name_audit.csv",
            index=False,
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    run_regression_checks()
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    audit_dir = Path(args.audit_dir)
    output_path = output_dir / "master_health_contracts.parquet"
    if not input_path.exists():
        raise SystemExit(f"Health Silver dataset not found: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    print(f"1/5 Loading Health Silver dataset: {input_path}")
    health = pd.read_parquet(input_path)
    required = {CN_ID, AGENCY, DESCRIPTION, CATEGORY, CATEGORY_TYPE,
                VALUE, SUPPLIER_NAME, SUPPLIER_GROUP_COL, "is_accenture"}
    missing = sorted(required - set(health.columns))
    if missing:
        raise RuntimeError("Health Silver dataset missing: " + ", ".join(missing))
    if health.empty:
        raise RuntimeError("Health Silver dataset is empty")
    if not health.index.is_unique:
        health = health.reset_index(drop=True)
    original_columns = list(health.columns)
    original_rows = len(health)
    original_value = pd.to_numeric(health[VALUE], errors="coerce").fillna(0).sum()
    health[VALUE] = pd.to_numeric(health[VALUE], errors="coerce").fillna(0)
    incoming_groups = health[SUPPLIER_GROUP_COL].copy()
    incoming_accenture = health["is_accenture"].copy()
    if incoming_groups.isna().any() or incoming_groups.astype(str).str.strip().eq("").any():
        raise RuntimeError("Shared Silver contains blank supplier_group")
    if not pd.api.types.is_bool_dtype(incoming_accenture):
        normalized = incoming_accenture.astype(str).str.strip().str.lower()
        if not normalized.isin(["true", "false"]).all():
            raise RuntimeError("is_accenture must be a boolean from shared Silver")
        health["is_accenture"] = normalized.eq("true")
    if not health["is_accenture"].eq(health[SUPPLIER_GROUP_COL].eq("Accenture")).all():
        raise RuntimeError("Shared Silver supplier_group / is_accenture disagreement")
    print(f"Health Silver rows: {original_rows:,}; columns: {len(original_columns):,}")
    print("2/5 Preserving shared supplier identity (no regrouping)")

    print("3/5 Applying Health POC addressability rules")
    addr = pd.DataFrame(
        [classify_addressability(row) for _, row in health.iterrows()],
        index=health.index,
    )
    for column in addr.columns:
        if column in health.columns:
            health = health.drop(columns=[column])
    health = pd.concat([health, addr], axis=1)
    health = enforce_accenture_addressability(health)

    print("4/5 Applying Health POC Service Offering rules")
    offerings = pd.DataFrame(
        [classify_service_offering(row) for _, row in health.iterrows()],
        index=health.index,
    )
    for column in offerings.columns:
        if column in health.columns:
            health = health.drop(columns=[column])
    health = pd.concat([health, offerings], axis=1)
    health["classification_review_required"] = (
        pd.to_numeric(health["addressability_confidence"], errors="coerce").fillna(0).lt(85)
        | (health["is_addressable"].astype(bool) & (
            pd.to_numeric(health["service_offering_confidence"], errors="coerce").fillna(0).lt(85)
            | health["Service Offering"].isna()
        ))
    )

    if len(health) != original_rows:
        raise RuntimeError("Health enrichment changed the Silver row count")
    if abs(float(health[VALUE].sum() - original_value)) > 0.01:
        raise RuntimeError("Health enrichment changed total procurement value")
    for column in original_columns:
        if column not in health.columns:
            raise RuntimeError(f"Silver column lost during enrichment: {column}")
    if not health[SUPPLIER_GROUP_COL].equals(incoming_groups):
        raise RuntimeError("Health enrichment changed shared supplier_group")
    if not health["is_accenture"].equals(incoming_accenture.astype(bool)):
        raise RuntimeError("Health enrichment changed shared is_accenture")
    if (health["is_accenture"] & ~health["is_addressable"]).any():
        raise RuntimeError("Observed Accenture win is not addressable")
    if (~health["is_addressable"] & health["Service Offering"].notna()).any():
        raise RuntimeError("Non-addressable Health contract received Service Offering")

    print("5/5 Writing wide Health Gold master and audits")
    health.to_parquet(output_path, index=False)
    export_audits(health, audit_dir)
    addressable = health.loc[health["is_addressable"]]
    acc = health.loc[health["is_accenture"]]
    unresolved = addressable.loc[addressable["Service Offering"].isna()]
    summary = {
        "health_rows": int(len(health)),
        "silver_columns_preserved": len(original_columns),
        "gold_columns": len(health.columns),
        "total_health_procurement": float(health[VALUE].sum()),
        "addressable_tam": float(addressable[VALUE].sum()),
        "accenture_wins": float(acc[VALUE].sum()),
        "addressable_rows": int(len(addressable)),
        "unresolved_service_offering_rows": int(len(unresolved)),
        "supplier_groups": int(health[SUPPLIER_GROUP_COL].nunique()),
        "output": str(output_path),
    }
    (audit_dir / "health_master_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("=" * 64)
    print("ATLAS HEALTH GOLD COMPLETE")
    print(f"Health rows: {len(health):,}")
    print(f"Columns: {len(original_columns):,} Silver -> {len(health.columns):,} Gold")
    print(f"Total Health procurement: A${summary['total_health_procurement']/1e9:,.3f}B")
    print(f"Addressable TAM: A${summary['addressable_tam']/1e9:,.3f}B")
    print(f"Accenture addressable wins: A${summary['accenture_wins']/1e9:,.3f}B")
    print(f"Unresolved Service Offering: {len(unresolved):,}")
    print(f"Wrote: {output_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
