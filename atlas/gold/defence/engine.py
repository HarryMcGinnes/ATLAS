
"""ATLAS Defence multi-signal Service Offering classification engine.

Service Offering classification is independent of addressability.
Scores represent evidence strength, not statistical probabilities.
"""

from pathlib import Path
import re

import yaml


CONFIG_PATH = Path("config/defence/service_offerings.yaml")

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

DECISIVE_DESCRIPTION_THRESHOLD = 35
DECISIVE_CATEGORY_THRESHOLD = 25
SUPPORTING_DESCRIPTION_THRESHOLD = 15


def load_config(path=CONFIG_PATH):
    """Load and validate classification configuration."""
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid Service Offering configuration")

    offerings = config.get("service_offerings")

    if not isinstance(offerings, list) or len(offerings) < 2:
        raise ValueError("At least two Service Offerings are required")

    names = [item["name"] for item in offerings]

    if len(names) != len(set(names)):
        raise ValueError("Duplicate Service Offering names")

    for offering in offerings:
        rules = offering.get("rules", {})
        if not isinstance(rules, dict):
            raise ValueError(f"Invalid rules for {offering['name']}")

        for source, source_rules in rules.items():
            if source not in SOURCE_CAPS:
                raise ValueError(f"Unknown evidence source: {source}")

            for rule in source_rules:
                re.compile(rule["pattern"])
                weight = int(rule.get("weight", 0))
                if weight < 0 or weight > SOURCE_CAPS[source]:
                    raise ValueError(
                        f"Invalid {source} weight in {offering['name']}"
                    )

    return config


def clean_text(value):
    """Safely normalise source text."""
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {"nan", "none", "<na>", "nat"}:
        return ""

    text = text.lower().replace("&", " and ")
    return re.sub(r"\s+", " ", text).strip()


def first_available(contract, fields):
    """Return the first non-empty source value."""
    for field in fields:
        value = clean_text(contract.get(field))
        if value:
            return value
    return ""


def get_contract_context(contract):
    """Extract the five evidence sources."""
    division = first_available(
        contract,
        ["Agency Division", "Agency Divison", "div_clean"],
    )
    branch = first_available(
        contract,
        ["Agency Branch", "branch_clean"],
    )

    return {
        "Description": clean_text(contract.get("Description")),
        "Category": clean_text(contract.get("Category")),
        "Category Type": clean_text(contract.get("Category Type")),
        "Organisation": f"{division} {branch}".strip(),
        "Supplier": first_available(
            contract, ["supplier_group", "Supplier Name"]
        ),
        "Domain": clean_text(contract.get("defence_domain")),
    }


def matches_any(text, patterns):
    """Return the first matched text or None."""
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def goods_guardrail(context, config):
    """Restrict Service Offering assignment for physical Goods.

    This is a classification guardrail, not an addressability
    decision. Physical/digital conflicts remain unclassified.
    """
    if context["Category Type"] != "goods":
        return True, None

    settings = config.get("goods_guardrail", {})
    physical = matches_any(
        context["Category"],
        settings.get("physical_patterns", []),
    )
    digital = matches_any(
        context["Description"],
        settings.get("digital_patterns", []),
    )
    category_digital = matches_any(
        context["Category"],
        settings.get("digital_patterns", []),
    )

    if physical:
        return False, (
            "Physical Goods category requires review"
            + (f"; digital Description [{digital}]" if digital else "")
        )

    if not digital and not category_digital:
        return False, "Goods without explicit digital evidence"

    return True, None


def source_score(text, rules, source_name, cap):
    """Take the strongest matching rule per evidence source."""
    best_score = 0
    best_evidence = None

    for rule in rules:
        match = re.search(
            rule["pattern"], text, flags=re.IGNORECASE
        )

        if not match:
            continue

        score = min(int(rule.get("weight", 0)), cap)

        if score > best_score:
            best_score = score
            reason = rule.get("reason", rule["pattern"])
            best_evidence = (
                f"{source_name}: {reason} "
                f"[{match.group(0)}] (+{score})"
            )

    return best_score, best_evidence


def confidence_grade(score, classified=True):
    """Grade evidence strength, not statistical confidence."""
    if not classified:
        return "D"
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return "A"
    if score >= REVIEW_THRESHOLD:
        return "B"
    return "C"


def make_result(
    offering=None,
    score=0,
    runner_up=None,
    runner_up_score=0,
    margin=0,
    scores=None,
    source_count=0,
    reason="",
    review=True,
):
    """Return a consistent classification record."""
    scores = scores or {}

    return {
        "service_offering": offering,
        "service_offering_score": int(score),
        "service_offering_grade": confidence_grade(
            score, offering is not None
        ),
        "service_offering_runner_up": runner_up,
        "service_offering_runner_up_score": int(runner_up_score),
        "service_offering_score_margin": int(margin),
        # Compatibility with existing audit scripts.
        "service_offering_confidence": int(score),
        "service_offering_reason": reason,
        "service_offering_description_score": int(
            scores.get("Description", 0)
        ),
        "service_offering_category_score": int(
            scores.get("Category", 0)
        ),
        "service_offering_organisation_score": int(
            scores.get("Organisation", 0)
        ),
        "service_offering_supplier_score": int(
            scores.get("Supplier", 0)
        ),
        "service_offering_domain_score": int(
            scores.get("Domain", 0)
        ),
        "service_offering_evidence_sources": int(source_count),
        "classification_review_required": bool(review),
    }


def score_offering(context, offering):
    """Score one Service Offering against all evidence sources."""
    rules = offering.get("rules", {})
    scores = {}
    evidence = []

    for source, cap in SOURCE_CAPS.items():
        score, reason = source_score(
            context[source],
            rules.get(source, []),
            source,
            cap,
        )

        scores[source] = score
        if reason:
            evidence.append(reason)

    return {
        "name": offering["name"],
        "score": min(sum(scores.values()), 100),
        "scores": scores,
        "evidence": evidence,
        "source_count": sum(v > 0 for v in scores.values()),
    }


def classify_contract(contract, config):
    """Classify a Defence contract independently of addressability."""
    context = get_contract_context(contract)

    allowed, guardrail_reason = goods_guardrail(context, config)

    if not allowed:
        return make_result(reason=guardrail_reason)

    results = [
        score_offering(context, offering)
        for offering in config["service_offerings"]
    ]

    # Stable ordering makes equal-score behaviour deterministic.
    ranked = sorted(
        results,
        key=lambda item: item["score"],
        reverse=True,
    )

    winner = ranked[0]
    runner_up = ranked[1]

    score = winner["score"]
    second_score = runner_up["score"]
    margin = score - second_score
    scores = winner["scores"]

    common = {
        "score": score,
        "runner_up": runner_up["name"],
        "runner_up_score": second_score,
        "margin": margin,
        "scores": scores,
        "source_count": winner["source_count"],
    }

    if score == 0:
        return make_result(
            reason="No matching Service Offering evidence",
            **common,
        )

    if margin == 0:
        return make_result(
            reason=(
                f"Ambiguous: {winner['name']} and "
                f"{runner_up['name']} both scored {score}/100"
            ),
            **common,
        )

    decisive_description = (
        scores["Description"] >= DECISIVE_DESCRIPTION_THRESHOLD
    )

    supported_category = (
        scores["Category"] >= DECISIVE_CATEGORY_THRESHOLD
        and scores["Description"] >= SUPPORTING_DESCRIPTION_THRESHOLD
    )

    sufficient_evidence = (
        score >= CLASSIFICATION_THRESHOLD
        or decisive_description
        or supported_category
    )

    # Context alone cannot establish a Service Offering.
    # At least Category or Description must support the winner.
    subject_evidence = (
        scores["Description"] > 0
        or scores["Category"] > 0
    )

    if not sufficient_evidence or not subject_evidence:
        return make_result(
            reason=(
                f"Insufficient subject evidence for "
                f"{winner['name']}: {score}/100. "
                + "; ".join(winner["evidence"])
            ),
            **common,
        )

    review_required = (
        score < REVIEW_THRESHOLD
        or margin < MINIMUM_CLEAR_MARGIN
    )

    return make_result(
        offering=winner["name"],
        reason="; ".join(winner["evidence"]),
        review=review_required,
        **common,
    )
