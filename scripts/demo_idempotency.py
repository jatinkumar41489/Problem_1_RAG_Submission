"""Evidence script: prints the vector count before/after re-ingesting the SAME corpus (screenshot this)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import Settings
from app.services import Services

s = Settings.from_env()
svc = Services(s)
print(f"collection={s.chroma_collection_name} model={svc.encoder.name}")
print(f"[start]            vectors = {svc.store.count()}")
for n in (1, 2):
    r = svc.pipeline.ingest_path(s.sample_dir)
    print(f"[ingest run #{n}]    vectors = {r.count_after} | new={r.files_new} updated={r.files_updated} unchanged={r.files_unchanged} upserted={r.chunks_upserted}")
print("idempotent:", "YES - count unchanged after re-ingest" if r.files_unchanged == r.files_discovered and r.chunks_upserted == 0 else "NO")
svc.close()
