"""Safe JSON persistence helpers using SQLite/SQLAlchemy to avoid race conditions."""

import json
import os
from pathlib import Path
from typing import Any
from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker

import config

Base = declarative_base()

class StoreItem(Base):
    __tablename__ = 'store'
    key = Column(String, primary_key=True)
    value = Column(Text)

DB_PATH = os.path.join(config.PROJECT_DIR, "trading_agent.db")
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def _keys_for_path(path: str | Path) -> tuple[str, str]:
    path_obj = Path(path)
    resolved = str(path_obj.resolve())
    legacy = path_obj.name
    return resolved, legacy

def read_json(path: str | Path, default: Any = None) -> Any:
    key, legacy_key = _keys_for_path(path)
    with SessionLocal() as db:
        for candidate in (key, legacy_key):
            item = db.query(StoreItem).filter(StoreItem.key == candidate).first()
            if item and item.value:
                try:
                    return json.loads(item.value)
                except Exception:
                    pass
    return default

def write_json_atomic(path: str | Path, data: Any) -> None:
    key, legacy_key = _keys_for_path(path)
    value = json.dumps(data, indent=2)
    with SessionLocal() as db:
        item = db.query(StoreItem).filter(StoreItem.key == key).first()
        if item:
            item.value = value
        else:
            item = StoreItem(key=key, value=value)
            db.add(item)
        # Migrate any legacy basename-only key to avoid stale collisions.
        legacy_item = db.query(StoreItem).filter(StoreItem.key == legacy_key).first()
        if legacy_item and legacy_key != key:
            db.delete(legacy_item)
        db.commit()
