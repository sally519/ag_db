from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(slots=True)
class Settings:
    env: str = os.getenv("RAG_DB_ENV", "dev")
    data_dir: Path = Path(os.getenv("RAG_DB_DATA_DIR", "./data"))
    log_level: str = os.getenv("RAG_DB_LOG_LEVEL", "INFO")

