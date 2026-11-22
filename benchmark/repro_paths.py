"""Paths used by the benchmark scripts inside the reproducibility package.

Added for the package: the original scripts used absolute paths of the authors' working folders. Every path is now
relative to the package root, and re-runs write below output/ so that the stored results in results/ are never
overwritten. Set CATFORGE_BENCH_OUT to send benchmark re-runs elsewhere (reproduce.py uses output/benchmarks for
--benchmarks and output/quick for --quick).
"""
import os
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]           # package root
DATA = PKG / "data"
CATFORGE = PKG / "catforge"                          # verbatim copies of the platform modules
RESULTS = PKG / "results"                            # stored results that the manuscript reports
OUTPUT = PKG / "output"
BENCH_OUT = Path(os.environ.get("CATFORGE_BENCH_OUT", OUTPUT / "benchmarks")).resolve()

# datasets (neutral names; see data/README.md for sources)
DRM_FILE = DATA / "drm_literature.csv"
WGS_FILE = DATA / "wgs_literature.csv"
OCM_FILE = DATA / "ocm_literature.csv"
