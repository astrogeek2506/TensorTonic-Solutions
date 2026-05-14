def detect_drift(reference_counts, production_counts, threshold):
    # Normalize reference histogram
    ref_total = sum(reference_counts)
    p = [x / ref_total for x in reference_counts]

    # Normalize production histogram
    prod_total = sum(production_counts)
    q = [x / prod_total for x in production_counts]

    # Compute TVD
    tvd = 0.5 * sum(abs(pi - qi) for pi, qi in zip(p, q))

    # Detect drift
    drift_detected = tvd > threshold

    return {
        "score": tvd,
        "drift_detected": drift_detected
    }