import json
from pathlib import Path

domain = "employee"
data_dir = Path("data") / domain
with open(data_dir / "clean_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")
with open(data_dir / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)
    print(f"Metadata entries: {len(metadata.get('id_to_metadata', {}))}")