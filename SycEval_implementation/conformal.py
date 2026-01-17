
def filter_claims(claims: list[str], scores: list[float], threshold: float) -> tuple[list[str], list[str]]:
    """
    Filters claims based on score threshold.
    Returns (kept_claims, dropped_claims)
    """
    kept = []
    dropped = []
    
    for c, s in zip(claims, scores):
        if s >= threshold:
            kept.append(c)
        else:
            dropped.append(c)
            
    return kept, dropped
