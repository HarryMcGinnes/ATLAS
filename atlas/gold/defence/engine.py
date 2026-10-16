"""ATLAS configurable multi-signal Service Offering engine."""

import re
from pathlib import Path

import yaml


CONFIG_PATH = Path(
    "config/defence/service_offerings.yaml"
)

SOURCE_CAPS = {
    "Description": 40,
    "Category": 30,
    "Organisation": 15,
    "Supplier": 10,
    "Domain": 5,
}

CLASSIFICATION_THRESHOLD = 50
REVIEW_THRESHOLD = 65
HIGH_CONFIDENCE_THRESHOLD = 80
MINIMUM_CLEAR_MARGIN = 10


def load_config(path=CONFIG_PATH):
    """Load the Defence Service Offering configuration."""

    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not config:
        raise ValueError(
            "Service Offering configuration is empty."
        )

    if not config.get("service_offerings"):
        raise ValueError(
            "No Service Offerings found in configuration."
        )

    return config


def clean_text(value):
    """Normalise a source value for regex matching."""

    if value is None:
        return ""

    text = str(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def first_available(contract, fields):
    """Return the first populated field from a list."""

    for field in fields:
        value = contract.get(field)

        if value is not None:
            text = str(value).strip()

            if text:
                return text

    return ""


def empty_result(reason, score=0, runner_up=None):
    """Return a standard unclassified result."""

    return {
        "service_offering": None,
        "service_offering_score": int(score),
        "service_offering_grade": "D",
        "service_offering_runner_up": runner_up,
        "service_offering_runner_up_score": 0,
        "service_offering_score_margin": 0,
        "service_offering_confidence": int(score),
        "service_offering_reason": reason,
        "service_offering_description_score": 0,
        "service_offering_category_score": 0,
        "service_offering_organisation_score": 0,
        "service_offering_supplier_score": 0,
        "service_offering_domain_score": 0,
        "service_offering_evidence_sources": 0,
        "classification_review_required": True,
    }


def source_score(
    text,
    rules,
    source_name,
    cap,
):
    """
    Score one evidence source.

    Only the strongest rule for each source is retained. This prevents
    duplicate or overlapping regex rules from artificially increasing
    the classification score.
    """

    best_score = 0
    best_reason = None
    best_match = None

    for rule in rules:
        pattern = rule["pattern"]
        weight = int(rule.get("weight", 0))

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match and weight > best_score:
            best_score = min(weight, cap)
            best_reason = rule.get(
                "reason",
                pattern,
            )
            best_match = match.group(0)

    if best_score == 0:
        return 0, None

    evidence = (
        f"{source_name}: "
        f"{best_reason} "
        f"[{best_match}] "
        f"(+{best_score})"
    )

    return best_score, evidence


def matches_any(text, patterns):
    """Return the first matching pattern, if any."""

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(0)

    return None


def get_contract_context(contract):
    """Build the source fields used by the classifier."""

    description = clean_text(
        contract.get("Description", "")
    )

    category = clean_text(
        contract.get("Category", "")
    )

    category_type = clean_text(
        contract.get("Category Type", "")
    )

    supplier = clean_text(
        first_available(
            contract,
            [
                "supplier_group",
                "Supplier Name",
            ],
        )
    )

    division = clean_text(
        first_available(
            contract,
            [
                "Agency Division",
                "Agency Divison",
                "div_clean",
            ],
        )
    )

    branch = clean_text(
        first_available(
            contract,
            [
                "Agency Branch",
                "branch_clean",
            ],
        )
    )

    organisation = (
        f"{division} {branch}"
    ).strip()

    domain = clean_text(
        contract.get(
            "defence_domain",
            "",
        )
    )

    return {
        "Description": description,
        "Category": category,
        "Category Type": category_type,
        "Supplier": supplier,
        "Organisation": organisation,
        "Domain": domain,
    }


def goods_guardrail(context, config):
    """
    Apply a Service Offering classification guardrail for Goods.

    This does not determine addressability. It only prevents physical
    Defence goods from being assigned to a Service Offering without
    strong explicit digital evidence.
    """

    if context["Category Type"] != "goods":
        return True, None

    settings = config.get(
        "goods_guardrail",
        {},
    )

    physical_patterns = settings.get(
        "physical_patterns",
        [],
    )

    digital_patterns = settings.get(
        "digital_patterns",
        [],
    )

    combined_text = (
        f"{context['Category']} "
        f"{context['Description']}"
    )

    physical_match = matches_any(
        context["Category"],
        physical_patterns,
    )

    digital_match = matches_any(
        combined_text,
        digital_patterns,
    )

    if physical_match and not digital_match:
        return (
            False,
            (
                "Goods classification guardrail: "
                f"physical Category evidence "
                f"[{physical_match}] and no strong "
                "digital Service Offering evidence."
            ),
        )

    if not digital_match:
        return (
            False,
            (
                "Goods classification guardrail: "
                "no strong digital Service Offering evidence."
            ),
        )

    return True, None


def confidence_grade(score):
    """Convert the 100-point evidence score into a grade."""

    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return "A"

    if score >= REVIEW_THRESHOLD:
        return "B"

    if score >= CLASSIFICATION_THRESHOLD:
        return "C"

    return "D"


def classify_contract(contract, config):
    """Classify one Defence contract using multiple evidence sources."""

    context = get_contract_context(
        contract
    )

    goods_allowed, goods_reason = goods_guardrail(
        context,
        config,
    )

    if not goods_allowed:
        return empty_result(
            goods_reason
        )

    results = []

    for position, offering in enumerate(
        config["service_offerings"]
    ):
        name = offering["name"]
        rules = offering.get("rules", {})

        description_score, description_evidence = (
            source_score(
                context["Description"],
                rules.get("Description", []),
                "Description",
                SOURCE_CAPS["Description"],
            )
        )

        category_score, category_evidence = (
            source_score(
                context["Category"],
                rules.get("Category", []),
                "Category",
                SOURCE_CAPS["Category"],
            )
        )

        organisation_score, organisation_evidence = (
            source_score(
                context["Organisation"],
                rules.get("Organisation", []),
                "Organisation",
                SOURCE_CAPS["Organisation"],
            )
        )

        supplier_score, supplier_evidence = (
            source_score(
                context["Supplier"],
                rules.get("Supplier", []),
                "Supplier",
                SOURCE_CAPS["Supplier"],
            )
        )

        domain_score, domain_evidence = (
            source_score(
                context["Domain"],
                rules.get("Domain", []),
                "Domain",
                SOURCE_CAPS["Domain"],
            )
        )

        scores = {
            "description": description_score,
            "category": category_score,
            "organisation": organisation_score,
            "supplier": supplier_score,
            "domain": domain_score,
        }

        evidence = [
            value
            for value in [
                description_evidence,
                category_evidence,
                organisation_evidence,
                supplier_evidence,
                domain_evidence,
            ]
            if value
        ]

        total_score = min(
            sum(scores.values()),
            100,
        )

        results.append(
            {
                "name": name,
                "position": position,
                "score": total_score,
                "scores": scores,
                "evidence": evidence,
                "source_count": sum(
                    score > 0
                    for score in scores.values()
                ),
            }
        )

    ranked = sorted(
        results,
        key=lambda item: (
            -item["score"],
            item["position"],
        ),
    )

    winner = ranked[0]
    runner_up = ranked[1]

    winner_score = winner["score"]
    runner_up_score = runner_up["score"]
    margin = winner_score - runner_up_score

    if winner_score == 0:
        return empty_result(
            "No matching Service Offering evidence."
        )

    if winner_score == runner_up_score:
        return {
            "service_offering": None,
            "service_offering_score": winner_score,
            "service_offering_grade": "D",
            "service_offering_runner_up": runner_up["name"],
            "service_offering_runner_up_score": runner_up_score,
            "service_offering_score_margin": 0,
            "service_offering_confidence": winner_score,
            "service_offering_reason": (
                "Ambiguous classification: "
                f"{winner['name']} and "
                f"{runner_up['name']} both scored "
                f"{winner_score}/100."
            ),
            "service_offering_description_score": (
                winner["scores"]["description"]
            ),
            "service_offering_category_score": (
                winner["scores"]["category"]
            ),
            "service_offering_organisation_score": (
                winner["scores"]["organisation"]
            ),
            "service_offering_supplier_score": (
                winner["scores"]["supplier"]
            ),
            "service_offering_domain_score": (
                winner["scores"]["domain"]
            ),
            "service_offering_evidence_sources": (
                winner["source_count"]
            ),
            "classification_review_required": True,
        }

    grade = confidence_grade(
        winner_score
    )

    if winner_score < CLASSIFICATION_THRESHOLD:
        service_offering = None
        review_required = True
        reason = (
            f"Weak Service Offering evidence for "
            f"{winner['name']}: "
            f"{winner_score}/100. "
            + "; ".join(winner["evidence"])
        )
    else:
        service_offering = winner["name"]

        review_required = (
            winner_score < REVIEW_THRESHOLD
            or margin < MINIMUM_CLEAR_MARGIN
        )

        reason = "; ".join(
            winner["evidence"]
        )

    return {
        "service_offering": service_offering,
        "service_offering_score": winner_score,
        "service_offering_grade": grade,
        "service_offering_runner_up": runner_up["name"],
        "service_offering_runner_up_score": runner_up_score,
        "service_offering_score_margin": margin,
        "service_offering_confidence": winner_score,
        "service_offering_reason": reason,
        "service_offering_description_score": (
            winner["scores"]["description"]
        ),
        "service_offering_category_score": (
            winner["scores"]["category"]
        ),
        "service_offering_organisation_score": (
            winner["scores"]["organisation"]
        ),
        "service_offering_supplier_score": (
            winner["scores"]["supplier"]
        ),
        "service_offering_domain_score": (
            winner["scores"]["domain"]
        ),
        "service_offering_evidence_sources": (
            winner["source_count"]
        ),
        "classification_review_required": review_required,
    }
