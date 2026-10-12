
"""ATLAS configurable classification engine."""

import re
from pathlib import Path

import yaml


CONFIG_PATH = Path("config/defence/service_offerings.yaml")


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def classify_contract(contract, config):
    """Score a contract against configured classification rules."""

    scores = {}
    evidence = {}

    for offering in config["service_offerings"]:
        name = offering["name"]
        scores[name] = 0
        evidence[name] = []

        for rule in offering.get("rules", []):
            field = rule["field"]
            pattern = rule["pattern"]
            weight = rule.get("weight", 1)

            value = str(contract.get(field, "") or "")

            if re.search(pattern, value, flags=re.IGNORECASE):
                scores[name] += weight
                evidence[name].append(rule.get("reason", pattern))

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    winner, score = ranked[0]

    if score == 0:
        return {
            "service_offering": None,
            "service_offering_confidence": 0,
            "service_offering_reason": "No matching rules",
        }

    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    margin = score - runner_up

    return {
        "service_offering": winner,
        "service_offering_confidence": round(
            100 * margin / max(score, 1)
        ),
        "service_offering_reason": "; ".join(evidence[winner]),
    }
