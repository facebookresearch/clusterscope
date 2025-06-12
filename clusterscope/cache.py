# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import ast
import functools
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, Union

CACHE_PATH = "/tmp/clusterscopewhoami"


def save(
    values: Dict[Hashable, Union[str, float, bool, int]], filepath: str = CACHE_PATH
) -> None:
    loaded = load()

    with open(filepath, "a") as f:
        for key, val in values.items():
            # Values are assumed to be static, so not updating existing keys
            if key not in loaded:
                f.write(f"{key}={repr(val)}\n")


def load(filepath: str = CACHE_PATH) -> Dict[Hashable, Union[str, float, bool, int]]:
    loaded: Dict[Hashable, Union[str, float, bool, int]] = {}
    path = Path(filepath)
    if path.exists():
        with path.open("r") as f:
            for line in f:
                if "=" in line:
                    key, val_str = line.strip().split("=", 1)
                    try:
                        loaded[key] = ast.literal_eval(val_str)
                    except (ValueError, SyntaxError):
                        loaded[key] = val_str
    return loaded


def fs_cache(var_name: str, filepath: str = CACHE_PATH):
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache = load(filepath=filepath)

            if var_name in cache:
                return cache[var_name]

            result = fn(*args, **kwargs)
            save(filepath=filepath, values={var_name: result})
            return result

        return wrapper

    return decorator
