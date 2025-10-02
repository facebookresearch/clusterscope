import re


def extract_gpus_from_gres(gres_string: str) -> int:
    """Extract the number of gpus from the GRES resources string"""
    gpus = 0
    gres_items = gres_string.split(",")
    for gres in gres_items:
        # If a gpu resource has been found.
        if gres.startswith("gpu:"):
            gpus = parse_gres(gres)

    return gpus


def parse_gres(gres_str: str) -> int:
    """Parse GPU count from GRES string.

    Handles formats like:
    - 'gpu:4'
    - 'gpu:a100:2'
    - 'gpu:volta:8(S:0-1)'
    - 'gpu:pascal:2'
    - '(null)'
    """
    if not gres_str or gres_str == "(null)":
        return 0

    match = re.search(r"gpu(?::\w+)?:(\d+)", gres_str)
    if match:
        return int(match.group(1))

    return 0
