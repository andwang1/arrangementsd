def produce_name(member, variant):
    name = []
    if "l2ae" in member:
        return "L2-AE"
    if "AURORA" in member:
        return "AURORA"
    if "best" in member:
        return "BEST"
    if "sample" in member:
        return "BD-SAMPLING"
    if "beta0" in member:
        return "NO-GLOBAL-PULL"
    if "manualBD" in member:
        return "PREDEFINED-BD"
    return "-".join(name)