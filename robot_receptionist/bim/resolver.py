import os
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

class RoomResolver:
    def __init__(self, rooms_data: dict, persist_directory: str = "data/chroma_db"):
        """
        Initialize RoomResolver with rooms data and set up ChromaDB.
        Uses ONNX-based MiniLM-L6-V2 for efficient execution on Jetson without Torch.
        """
        self.rooms_data = rooms_data
        self.persist_directory = persist_directory
        
        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize Embedding Model (ONNX variant)
        # This will download the model file (~80MB) on the first run.
        # It's much lighter and more compatible with Jetson than sentence-transformers + torch.
        self.ef = embedding_functions.ONNXMiniLM_L6_V2()
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="rooms",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Index data if empty
        if self.collection.count() == 0:
            self._index_rooms()

    def _index_rooms(self):
        """Index all rooms from the provided dictionary into ChromaDB."""
        nodes = self.rooms_data.get("nodes", [])
        ids = []
        documents = []
        metadatas = []
        
        for node in nodes:
            room_id = node.get("id")
            room_name = node.get("name")
            aliases = node.get("aliases", [])
            
            # Combine name and aliases for embedding
            # Example: "Phòng 101, 101"
            search_text = f"{room_name}, {' '.join(aliases)}"
            
            ids.append(room_id)
            documents.append(search_text)
            metadatas.append({"id": room_id, "name": room_name})
            
        if ids:
            # We add data directly to collection; Chroma will use EF to embed it
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

    def resolve(self, query: str) -> str | None:
        """
        Resolve a natural language query to a room_id.
        Returns None if no match is found above the 0.4 threshold.
        """
        if not query:
            return None
        
        results = self.collection.query(
            query_texts=[query],
            n_results=1
        )
        
        if not results or not results['ids'] or not results['ids'][0]:
            return None
            
        # ChromaDB returns distances (lower is better for cosine distance)
        # Cosine Similarity = 1 - Cosine Distance
        # We need Cosine Similarity > 0.6, so Cosine Distance < 0.4
        distance = results['distances'][0][0]
        if distance > 0.4:  # Similarity < 0.6
            return None
            
        return results['ids'][0][0]
