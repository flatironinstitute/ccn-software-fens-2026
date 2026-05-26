#!/usr/bin/env python3

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from .fetch import DOWNLOADABLE_FILES, fetch_data, fetch_all
from .plotting import *

import pathlib

repo_dir = pathlib.Path(__file__).parent.parent.parent
os.environ["NEMOS_DATA_DIR"] = os.environ.get("NEMOS_DATA_DIR", str(repo_dir / "data"))
