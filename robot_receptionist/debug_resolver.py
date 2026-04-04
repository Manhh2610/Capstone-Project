import json
from bim.resolver import RoomResolver
from pathlib import Path

# Load data
DATA_PATH = Path("data/rooms.json")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    rooms_data = json.load(f)

# Init resolver
resolver = RoomResolver(rooms_data)

# Test queries
queries = ["phòng 101", "phòng sếp", "không gian xanh xyz"]
print("--- Resolver Debug ---")
for q in queries:
    results = resolver.collection.query(
        query_texts=[q],
        n_results=3
    )
    print(f"\nQuery: '{q}'")
    for i in range(len(results['ids'][0])):
        room_id = results['ids'][0][i]
        dist = results['distances'][0][i]
        doc = results['documents'][0][i]
        print(f"  [{i+1}] {room_id}: Distance={dist:.4f}, Doc='{doc}'")
