"""
Resolve room queries to room_id using Exact Match and Semantic Search (ChromaDB + sentence-transformers).
"""
import chromadb
from sentence_transformers import SentenceTransformer

class RoomResolver:
    def __init__(self, rooms_data: list[dict]):
        """
        Initialize RoomResolver with a list of room dictionaries.
        Sets up an in-memory ChromaDB for semantic search.
        """
        self.rooms_data = rooms_data
        self.exact_map = {}
        
        # Build exact match map: lowercase names and aliases -> room_id
        for room in self.rooms_data:
            room_id = room.get("id")
            if not room_id:
                continue
            
            name = room.get("name", "").lower()
            if name:
                self.exact_map[name] = room_id
                
            for alias in room.get("aliases", []):
                self.exact_map[alias.lower()] = room_id
                
        # Initialize Embedding Model using sentence-transformers
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        # Initialize in-memory ChromaDB
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            name="rooms",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Index data into ChromaDB
        self._index_rooms()

    def _index_rooms(self):
        ids = []
        documents = []
        embeddings = []
        
        for room in self.rooms_data:
            room_id = room.get("id")
            if not room_id:
                continue
                
            room_name = room.get("name", "")
            aliases = room.get("aliases", [])
            
            # Combine name and aliases
            search_text = f"{room_name} {' '.join(aliases)}"
            
            ids.append(room_id)
            documents.append(search_text)
            
        if ids:
            embeddings = self.embedding_model.encode(documents).tolist()
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings
            )

    def resolve(self, query: str) -> str | None:
        """
        Resolve natural language text to a room_id.
        1. Exact Match (lowercase)
        2. Semantic Search (cosine similarity >= 0.5)
        """
        if not query:
            return None
            
        # 1. Exact Match Fallback
        q_lower = query.strip().lower()
        if q_lower in self.exact_map:
            return self.exact_map[q_lower]
            
        # 2. Semantic Search using ChromaDB
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=1
        )
        
        if not results or not results['ids'] or not results['ids'][0]:
            return None
            
        # For cosine space in ChromaDB, the distance is Cosine Distance.
        # Cosine Similarity = 1 - Cosine Distance
        distance = results['distances'][0][0]
        similarity = 1.0 - distance
        
        if similarity >= 0.5:
            return results['ids'][0][0]
            
        return None
