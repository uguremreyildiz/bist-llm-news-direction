"""
config.py -- Paths and protocol constants used by the revision analyses.
========================================================================
Public inputs live inside this repository. Private inputs (raw article texts and per-model LLM
outputs) stay on the authors' machine and are never copied into the repository; scripts that
need them check `available()` and skip those parts when they are absent.
"""
from pathlib import Path                                        # path handling

REPO = Path(__file__).resolve().parents[1]                      # repository root
HOME = Path.home()                                              # authors' home directory

# -- Public: in this repository --------------------------------------------
PANEL = REPO / "data" / "panel_bge_m3.parquet"                  # shared analysis panel
PRED_CSV = REPO / "outputs" / "oos_predictions.csv"             # written by reproduce.py
OUT_TABLES = REPO / "revision" / "outputs" / "tables"           # CSV / JSON outputs
OUT_FIGURES = REPO / "revision" / "outputs" / "figures"         # PNG / PDF outputs

# -- Private: authors' machine only ----------------------------------------
DESKTOP = HOME / "Desktop"                                      # private data root
RAW_DIR = DESKTOP / "RawDataset"                                # monthly GDELT JSONL per ticker
ENRICHED_DIRS = {                                               # per-model enrichment output
    "gpt": DESKTOP / "GPT_4_1_NanoDataset",                     # openai/gpt-4.1-nano
    "llama": DESKTOP / "Llama3_370BinstructDataset",            # meta-llama/llama-3.3-70b-instruct
    "gemma": DESKTOP / "Gemma4_26B_Dataset",                    # google/gemma-4-26b-a4b-it
}
LLM_FEATURES = DESKTOP / "DATASET" / "LLMFeatures"              # merged three-model article records

# -- Protocol constants (identical to reproduce.py) -------------------------
SEED = 42                                                       # global seed
BOOT_SEED = 20260101                                            # bootstrap seed
Q_PCA = 32                                                      # PCA components per embedding stream
TAU0, H, STEP = 156, 12, 12                                     # expanding window parameters
DEV_FRACTION = 0.60                                             # tuning / out-of-sample boundary
COST_BPS = 10                                                   # one-way transaction cost
PERIODS = 52                                                    # weekly annualisation
BLOCK_LEN = 8                                                   # moving-block bootstrap block length (weeks)
N_BOOT = 5000                                                   # bootstrap draws
SELECTED = dict(C=0.02005544632966435, penalty="l1")            # the published model


def available(path):
    """True when a private input is present on this machine."""
    return path is not None and Path(path).exists()


def ensure_dirs():
    """Create the output directories if they do not exist."""
    for d in (OUT_TABLES, OUT_FIGURES):
        Path(d).mkdir(parents=True, exist_ok=True)
