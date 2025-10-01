def parse_memory_to_gb(memory) -> int:
    """Parse memory string and convert to GB.

    Returns:
        int: Memory in GB
    """
    mem_value = memory.rstrip("GT")
    if memory.endswith("T"):
        return int(mem_value) * 1024
    elif memory.endswith("G"):
        return int(mem_value)
    else:
        raise RuntimeError(f"Invalid memory format: {memory}")
