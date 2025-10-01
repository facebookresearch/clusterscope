import re


def parse_gpu_memory(gres_str: str) -> int:
    """Parse GPU memory from GRES string if available (returns MB)."""
    if not gres_str:
        return 0

    # Look for patterns like gpu_mem:16000M or gpu:a100:2(S:0-1)(mem:40960M)
    match = re.search(r"mem:(\d+)([MGT])?", gres_str, re.IGNORECASE)
    if match:
        value = int(match.group(1))
        unit = match.group(2).upper() if match.group(2) else "M"

        if unit == "G":
            return value * 1024
        elif unit == "T":
            return value * 1024 * 1024
        else:  # M or no unit
            return value

    return 0


def parse_gpu_gres(gres_str: str) -> int:
    """Parse GPU count from GRES string (e.g., 'gpu:4' or 'gpu:a100:2')."""
    if not gres_str or gres_str == "(null)":
        return 0

    # Match patterns like gpu:4, gpu:a100:2, gpu:tesla:4, etc.
    match = re.search(r"gpu(?::\w+)?:(\d+)", gres_str)
    if match:
        return int(match.group(1))

    # If just 'gpu' with no count, assume 0
    return 0
