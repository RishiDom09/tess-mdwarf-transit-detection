"""Read the checked-in numerical recipe, independently of the working directory."""
import json
from pathlib import Path
REPOSITORY = Path(__file__).resolve().parents[2]
CONFIG = json.loads((REPOSITORY/'configs/experiment.json').read_text())
BIWEIGHT_SETTINGS = CONFIG['biweight']
CLIPPING_SETTINGS = CONFIG['upper_clipping']
def bls_settings():
    return dict(CONFIG['bls'])
