import math

def evaluate_shadow(production_log, shadow_log, criteria):
    """
    Evaluate whether a shadow model is ready for promotion.
    """

    n = len(production_log)

    # Production accuracy
    production_correct = sum(
        1
        for entry in production_log
        if entry["prediction"] == entry["actual"]
    )

    production_accuracy = production_correct / n

    # Shadow accuracy
    shadow_correct = sum(
        1
        for entry in shadow_log
        if entry["prediction"] == entry["actual"]
    )

    shadow_accuracy = shadow_correct / n

    # Accuracy gain
    accuracy_gain = shadow_accuracy - production_accuracy

    # Agreement rate
    agreements = sum(
        1
        for prod, shadow in zip(production_log, shadow_log)
        if prod["prediction"] == shadow["prediction"]
    )

    agreement_rate = agreements / n

    # P95 latency
    latencies = sorted(
        entry["latency_ms"] for entry in shadow_log
    )

    index = math.ceil(0.95 * n) - 1
    shadow_latency_p95 = latencies[index]

    # Promotion decision
    promote = (
        accuracy_gain >= criteria["min_accuracy_gain"]
        and shadow_latency_p95 <= criteria["max_latency_p95"]
        and agreement_rate >= criteria["min_agreement_rate"]
    )

    return {
        "promote": promote,
        "metrics": {
            "shadow_accuracy": shadow_accuracy,
            "production_accuracy": production_accuracy,
            "accuracy_gain": accuracy_gain,
            "shadow_latency_p95": shadow_latency_p95,
            "agreement_rate": agreement_rate
        }
    }