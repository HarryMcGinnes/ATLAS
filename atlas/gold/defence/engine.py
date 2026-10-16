
"""ATLAS Defence Service Offering classification engine.

Uses Description, Category, Organisation, Supplier and Domain.
Addressability is deliberately handled separately.
Scores represent evidence strength, not calibrated probabilities.
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
DECISIVE_CATEGORY_THRESHOLD = 24
SUPPORTING_DESCRIPTION_THRESHOLD = 15


def clean_text(value):
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {"nan", "none", "<na>", "nat"}:
        return ""

    text = text.lower().replace("&", " and ")
    return re.sub(r"\s+", " ", text).strip()


def first_available(contract, fields):
    for field in fields:
        value = clean_text(contract.get(field))
        if value:
            return value
    return ""


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid classification configuration")

    offerings = config.get("service_offerings")

    if not isinstance(offerings, list) or len(offerings) < 2:
        raise ValueError("At least two Service Offerings required")

    names = [item["name"] for item in offerings]

    if len(names) != len(set(names)):
        raise ValueError("Duplicate Service Offering names")

    for offering in offerings:
        rules = offering.get("rules", {})

        if not isinstance(rules, dict):
            raise ValueError("Invalid Service Offering rules")

        for source, entries in rules.items():
            if source not in SOURCE_CAPS:
                raise ValueError(f"Unknown evidence source: {source}")

            if not isinstance(entries, list):
                raise ValueError(f"Invalid rules for {source}")

            for entry in entries:
                re.compile(entry["pattern"])
                weight = int(entry.get("weight", 0))

                if not 0 <= weight <= SOURCE_CAPS[source]:
                    raise ValueError(
                        f"Invalid weight in {offering['name']}: {source}"
                    )

    return config


def get_contract_context(contract):
    division = first_available(
        contract,
        [
            "Agency Division",
            "35. Agency Division",
            "Agency Divison",
            "div_clean",
        ],
    )

    branch = first_available(
        contract,
        [
            "Agency Branch",
            "36. Agency Branch",
            "branch_clean",
        ],
    )

    return {
        "Description": clean_text(contract.get("Description")),
        "Category": clean_text(contract.get("Category")),
        "Category Type": clean_text(contract.get("Category Type")),
        "Organisation": f"{division} {branch}".strip(),
        "Supplier": first_available(
            contract,
            ["supplier_group", "Supplier Name"],
        ),
        "Domain": clean_text(contract.get("defence_domain")),
    }


def matches_any(text, patterns):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def goods_guardrail(context, config):
    """Prevent physical Goods from receiving misleading SO labels."""

    if context["Category Type"] != "goods":
        return True, None

    settings = config.get("goods_guardrail", {})

    physical = matches_any(
        context["Category"],
        settings.get("physical_patterns", []),
    )

    digital = matches_any(
        context["Description"] + " " + context["Category"],
        settings.get("digital_patterns", []),
    )

    if physical:
        return False, (
            "Physical Goods category; digital component requires review"
            if digital else "Physical Goods category"
        )

    if not digital:
        return False, "Goods without explicit digital evidence"

    return True, None


def source_score(text, rules, source, cap):
    """Retain the strongest matching rule per evidence source."""

    best_score = 0
    best_reason = None

    for rule in rules:
        match = re.search(
            rule["pattern"],
            text,
            flags=re.IGNORECASE,
        )

        weight = min(int(rule.get("weight", 0)), cap)

        if match and weight > best_score:
            best_score = weight
            best_reason = (
                f"{source}: "
                f"{rule.get('reason', rule['pattern'])} "
                f"[{match.group(0)}] (+{weight})"
            )

    return best_score, best_reason


def score_offering(context, offering):
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


def confidence_grade(score, classified):
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


def classify_contract(contract, config):
    """Classify a contract without deciding addressability."""

    context = get_contract_context(contract)

    allowed, guardrail_reason = goods_guardrail(context, config)

    if not allowed:
        return make_result(reason=guardrail_reason)

    ranked = sorted(
        [
            score_offering(context, offering)
            for offering in config["service_offerings"]
        ],
        key=lambda item: item["score"],
        reverse=True,
    )

    winner, runner_up = ranked[:2]

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

    description = scores["Description"]
    category = scores["Category"]
    organisation = scores["Organisation"]
    supplier = scores["Supplier"]

    # Strong explicit descriptions are sufficient independently.
    decisive_description = (
        description >= DECISIVE_DESCRIPTION_THRESHOLD
    )

    # Category reinforced by a relevant description.
    category_with_description = (
        category >= DECISIVE_CATEGORY_THRESHOLD
        and description >= SUPPORTING_DESCRIPTION_THRESHOLD
    )

    # Generic professional-service categories can be reinforced
    # by supplier or procuring organisation context.
    category_with_context = (
        category >= DECISIVE_CATEGORY_THRESHOLD
        and (organisation > 0 or supplier > 0)
    )

    # Supplier, Organisation and Domain cannot classify alone.
    subject_evidence = description > 0 or category > 0

    sufficient_evidence = (
        subject_evidence
        and (
            score >= CLASSIFICATION_THRESHOLD
            or decisive_description
            or category_with_description
            or category_with_context
        )
    )

    if margin == 0:
        return make_result(
            reason=(
                f"Ambiguous: {winner['name']} and "
                f"{runner_up['name']} both scored {score}/100"
            ),
            **common,
        )

    if not sufficient_evidence:
        return make_result(
            reason=(
                f"Insufficient subject evidence for "
                f"{winner['name']}: {score}/100. "
                + "; ".join(winner["evidence"])
            ),
            **common,
        )

    # Preserve provisional classifications while flagging
    # weaker or closely competing interpretations for review.
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
