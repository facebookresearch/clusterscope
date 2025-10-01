# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import ast
import fcntl
import functools
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, Union

CACHE_PATH = "/tmp/clusterscopewhoami"


def save(
    values: Dict[Hashable, Union[str, float, bool, int]], filepath: str = CACHE_PATH
) -> None:
    path = Path(filepath)

    if not path.exists():
        old_umask = os.umask(0)
        try:
            fd = os.open(filepath, os.O_CREAT | os.O_WRONLY, 0o666)
            os.close(fd)
        finally:
            os.umask(old_umask)

    # Ensure file has permissions for all users to read/write
    try:
        os.chmod(filepath, 0o666)
    except PermissionError:
        # Try to write anyway and let it fail with a better error
        pass

    loaded = load(filepath)

    try:
        fd = os.open(filepath, os.O_WRONLY | os.O_APPEND)
    except PermissionError:
        # Can't open the file due to permission error, ignore caching instead of failing
        logging.debug(
            f"PermissionError when trying to open {filepath=}, won't cache results"
        )
        return
    f = os.fdopen(fd, "a")

    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        for key, val in values.items():
            if key not in loaded:
                f.write(f"{key}={repr(val)}\n")
        f.flush()
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


def load(filepath: str = CACHE_PATH) -> Dict[Hashable, Union[str, float, bool, int]]:
    loaded: Dict[Hashable, Union[str, float, bool, int]] = {}
    path = Path(filepath)

    if not path.exists():
        return loaded

    fd = os.open(filepath, os.O_RDONLY)
    f = os.fdopen(fd, "r")

    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        for line in f:
            if "=" in line:
                key, val_str = line.strip().split("=", 1)
                try:
                    loaded[key] = ast.literal_eval(val_str)
                except (ValueError, SyntaxError):
                    loaded[key] = val_str
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()

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
