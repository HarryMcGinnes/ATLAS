
"""ATLAS Defence Service Offering classification."""

SERVICE_OFFERINGS = (
    "Strategy, Transformation & Advisory",
    "SI & Engineering",
    "Data, AI & Automation",
    "Cloud Infrastructure & Cyber",
    "Managed Services & Operations",
)


def classify_service_offering(contract):
    """
    Classify a contract independently of addressability.

    Classification rules will be added in the next step.
    """
    return {
        "service_offering": None,
        "service_offering_confidence": 0,
        "service_offering_reason": "Awaiting classification",
    }

