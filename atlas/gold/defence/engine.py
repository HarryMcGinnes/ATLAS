"""ATLAS configurable classification engine."""

import re
from pathlib import Path

import yaml


CONFIG_PATH = Path("config/defence/service_offerings.yaml")


def load_config(path=CONFIG_PATH):
    """Load the Defence Service Offering configuration."""

    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def classify_contract(contract, config):
    """Score a contract against configured classification rules."""

    scores = {}
    evidence = {}

    for offering in config.get("service_offerings", []):
        name = offering["name"]

        scores[name] = 0
        evidence[name] = []

        for rule in offering.get("rules", []):
            field = rule["field"]
            pattern = rule["pattern"]
            weight = rule.get("weight", 1)

            value = str(contract.get(field, "") or "")

            if re.search(
                pattern,
                value,
                flags=re.IGNORECASE,
            ):
                scores[name] += weight
                evidence[name].append(
                    rule.get("reason", pattern)
                )

    # Safety check in case no Service Offerings exist in config.
    if not scores:
        return {
            "service_offering": None,
            "service_offering_score": 0,
            "service_offering_runner_up": None,
            "service_offering_runner_up_score": 0,
            "service_offering_score_margin": 0,
            "service_offering_confidence": 0,
            "service_offering_reason": "No Service Offerings configured",
            "classification_review_required": True,
        }

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    winner, winner_score = ranked[0]

    if len(ranked) > 1:
        runner_up, runner_up_score = ranked[1]
    else:
        runner_up = None
        runner_up_score = 0

    # Nothing matched.
    if winner_score == 0:
        return {
            "service_offering": None,
            "service_offering_score": 0,
            "service_offering_runner_up": None,
            "service_offering_runner_up_score": 0,
            "service_offering_score_margin": 0,
            "service_offering_confidence": 0,
            "service_offering_reason": "No matching rules",
            "classification_review
