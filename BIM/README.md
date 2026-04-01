# BIM Navigation Service — Bước 2

Pathfinding trong tòa nhà từ `rooms.json` → NetworkX → Dijkstra → steps list tiếng Việt.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy server

```bash
python bim_service.py
# hoặc
uvicorn bim_service:app --host 0.0.0.0 --port 8001 --reload
```

API docs: http://localhost:8001/docs

## Chạy tests

```bash
pytest tests/ -v
```

## Endpoints

| Method | URL | Mô tả |
|--------|-----|--------|
| GET | `/health` | Health check |
| GET | `/rooms` | Danh sách phòng |
| GET | `/graph/info` | Thông tin đồ thị |
| POST | `/navigate` | Tìm đường A→B |
| GET | `/navigate?from_id=X&to_id=Y` | Tìm đường (GET) |

## Ví dụ

```bash
# Tìm đường từ lối vào đến phòng 201 (tầng 2)
curl -X POST http://localhost:8001/navigate \
  -H "Content-Type: application/json" \
  -d '{"from_id": "entrance", "to_id": "room_201", "preference": "elevator"}'

# Tìm theo alias tiếng Việt
curl -X POST http://localhost:8001/navigate \
  -H "Content-Type: application/json" \
  -d '{"from_id": "cửa chính", "to_id": "phòng giám đốc"}'
```

## Cấu trúc `rooms.json`

```json
{
  "nodes": [{ "id": "room_101", "name": "Phòng 101", "type": "room", "floor": 0, ... }],
  "edges": [{ "from": "entrance", "to": "lobby_f0", "distance": 5, "bidirectional": true }]
}
```

**Node types:** `entrance`, `lobby`, `corridor`, `room`, `meeting_room`, `toilet`, `staircase`, `elevator`
