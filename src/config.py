"""Project-wide configuration.

This file defines constants and paths used throughout all weekly labs and source
modules. Having all configuration in one place makes it easy to change settings
for the entire project.

Students: you usually don't need to edit this file.
Instructors: feel free to tune defaults for your class.

For beginners: Think of this as the "settings file" for the entire project.
Instead of hardcoding values like file paths or dataset names everywhere, we
define them once here and import them where needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ============================================================
# Valid Labels for FEVER Fact-Checking
# ============================================================
# These are the only three possible labels (categories) in the FEVER dataset.
# For beginners: A label is like a "tag" that tells us whether a claim is:
# - "SUPPORTS": The evidence proves the claim is true
# - "REFUTES": The evidence proves the claim is false
# - "NOT ENOUGH INFO": We don't have enough evidence to decide
LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

# ============================================================
# Dataset Configuration
# ============================================================
# DEFAULT_DATASET: The name of the HuggingFace dataset we use in labs.
# For beginners: HuggingFace is a platform for sharing datasets and models.
# This dataset contains claims + gold evidence from the FEVER task.
# We use a curated version (copenlu/fever_gold_evidence) that includes
# evidence text directly, so we don't need to download all of Wikipedia.
DEFAULT_DATASET = "copenlu/fever_gold_evidence"

# DEFAULT_DATASET_CONFIG: The specific subset/configuration of the dataset.
# For beginners: Some HuggingFace datasets have multiple versions or subsets.
# "default" means we're using the standard version (most common choice).
DEFAULT_DATASET_CONFIG = "default"   # Hugging Face "subset" name

# DEFAULT_SEED: The random seed used for reproducibility.
# For beginners: 42 is a popular choice (from "The Hitchhiker's Guide to the Galaxy"),
# but any integer works. Using the same seed ensures experiments are reproducible.
DEFAULT_SEED = 42

# ============================================================
# Project Paths
# ============================================================
# For beginners: The @dataclass decorator is a Python feature that automatically
# creates a class with __init__, __repr__, and other useful methods. This saves
# us from writing boilerplate code. Think of it as a convenient way to group
# related data together.
#
# frozen=True means this dataclass is "immutable" (cannot be changed after creation).
# This prevents accidental modifications to paths, which could cause confusing bugs.
@dataclass(frozen=True)
class Paths:
    """Central location for all project file paths.

    For beginners: Instead of writing paths like "../outputs/" or "../../data/"
    everywhere (which breaks if you move files), we define all paths here once.
    Then we can use PATHS.data_dir, PATHS.outputs_dir, etc. throughout the code.

    Attributes
    ----------
    project_root : Path
        The root directory of the project (where this README.md is located)
    data_dir : Path
        Where downloaded data files are stored (project_root/data)
    outputs_dir : Path
        Where weekly labs save outputs like plots and reports (project_root/outputs)
    cache_dir : Path
        Where cached data is stored for faster re-runs (project_root/.cache)
    """
    # project_root: Find the root directory of the project by going up one level from src/
    # __file__ is the path to this file (config.py)
    # .resolve() makes it an absolute path (e.g., /home/user/si376-w2026-dev/src/config.py)
    # .parents[1] goes up one level: src/ -> project root
    # For beginners: This ensures paths work correctly whether you run code from
    # the project root, from scripts/, or from anywhere else.
    project_root: Path = Path(__file__).resolve().parents[1]

    # data_dir: Where we store downloaded datasets
    # The / operator for Path objects joins paths (like os.path.join)
    # For beginners: This is equivalent to "project_root/data" but works on all operating systems
    data_dir: Path = project_root / "data"

    # outputs_dir: Where weekly labs save their results (plots, reports, etc.)
    outputs_dir: Path = project_root / "outputs"

    # cache_dir: Where data_loading.py stores cached data for faster re-runs
    # For beginners: The . prefix makes this a "hidden" directory (won't clutter file listings)
    cache_dir: Path = project_root / ".cache"

# Create a single instance of Paths that everyone imports
# For beginners: Instead of creating Paths() over and over, we create it once here
# and import PATHS everywhere. For example: from src.config import PATHS
PATHS = Paths()
