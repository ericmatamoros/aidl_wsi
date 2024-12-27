"""
Top-level package global definitions
"""
import os

from pathlib import Path


BASE_PATH = Path(os.path.dirname(__file__))
CONFIG_PATH = BASE_PATH / "config"
DATA_PATH = BASE_PATH / "data"
