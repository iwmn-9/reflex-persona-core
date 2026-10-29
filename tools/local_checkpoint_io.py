"""Publish operational status despite brief Windows reader/rename conflicts.

Frozen calculation checkpoints are not modified by this helper. Observers use
post-checkpoint log notices while a calculation is active, then audit its saved
file after the worker releases ownership. Only operational writers retry here.
"""
import json
import os
from pathlib import Path
import time


def read_json(path):
    # Close the handle before parsing; do not hold a rename-conflicting reader
    # while processing a potentially large operational snapshot.
    raw = Path(path).read_bytes()
    return json.loads(raw)


def atomic_json(path, value, *, sharing_timeout=2.0):
    path = Path(path)
    temporary = path.with_suffix('.new')
    temporary.write_bytes(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf8'))
    deadline = time.monotonic() + sharing_timeout
    while True:
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            if (os.name != 'nt' or getattr(error, 'winerror', None) not in (5, 32)
                    or time.monotonic() >= deadline):
                raise
            time.sleep(min(.02, max(0.0, deadline-time.monotonic())))
