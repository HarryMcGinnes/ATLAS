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
                evidence[name].append(
                    rule.get("reason", pattern)
                )

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    winner, score = ranked[0]
    runner_up, runner_up_score = ranked[1]

    if score == 0:
        return {
            "service_offering": None,
            "service_offering_score": 0,
            "service_offering_runner_up": None,
            "service_offering_runner_up_score": 0,
            "service_offering_score_margin": 0,
            "service_offering_confidence": 0,
            "service_offering_reason": "No matching rules",
            "classification_review_required": True,
        }

    margin = score - runner_up_score

    relative_margin = round(
        100 * margin / max(score, 1)
    )

    tie = score == runner_up_score

    if tie:
        winner_value = None
        confidence = 0
        review_required = True
        reason = (
            f"Ambiguous classification: "
            f"{winner} and {runner_up} both scored {score}"
        )
    else:
        winner_value = winner
        confidence = relative_margin

        # Initial review rule.
        # We will tune this after analysing real results.
        review_required = (
            score < 12
            or margin < 4
        )

        reason = "; ".join(evidence[winner])

    return {
        "service_offering": winner_value,
        "service_offering_score": score,
        "service_offering_runner_up": runner_up,
        "service_offering_runner_up_score": runner_up_score,
        "service_offering_score_margin": margin,
        "service_offering_confidence": confidence,
        "service_offering_reason": reason,
        "classification_review_required": review_required,
    }
