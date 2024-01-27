from src.utils import get_project_root

DATA_FOLDER = get_project_root() / "data"
RAW_DATA_FOLDER = DATA_FOLDER / "01_raw"
PROCESSED_DATA_FOLDER = DATA_FOLDER / "02_processed"
REFINED_DATA_FOLDER = DATA_FOLDER / "03_refined"

MODELS_FOLDER = get_project_root() / "models"
