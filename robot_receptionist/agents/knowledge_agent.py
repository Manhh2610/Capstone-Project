import os
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class KnowledgeAgent:
    """
    Agent that manages the Knowledge Base (RAG) using ChromaDB and ONNX embeddings.
    """
    def __init__(self, kn_path: str = "data/knowledge.md", persist_directory: str = "data/chroma_db"):
        self.kn_path = kn_path
        self.persist_directory = persist_directory
        
        # Ensure directories exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Use ONNXMiniLM_L6_V2 (same as RoomResolver for performance/compatibility)
        self.ef = embedding_functions.ONNXMiniLM_L6_V2()
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="knowledge",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Index data if empty
        if self.collection.count() == 0:
            self._index_knowledge()

    def _index_knowledge(self):
        """Read knowledge.md, chunk it, and index into ChromaDB."""
        if not os.path.exists(self.kn_path):
            logger.warning(f"Knowledge file not found at {self.kn_path}")
            return

        with open(self.kn_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Simple Chunking by Sections (Headers)
        sections = content.split("\n## ")
        ids = []
        documents = []
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            
            # Clean section text
            clean_text = section if i == 0 else "## " + section
            
            ids.append(f"chunk_{i}")
            documents.append(clean_text.strip())
            
        if ids:
            logger.info(f"Indexing {len(ids)} knowledge chunks...")
            self.collection.add(
                ids=ids,
                documents=documents
            )

    def query_knowledge(self, query: str, top_n: int = 2) -> str:
        """
        Query the knowledge base for relevant context.
        Returns a combined string of the most relevant chunks.
        """
        if not query:
            return ""
            
        results = self.collection.query(
            query_texts=[query],
            n_results=top_n
        )
        
        if not results or not results['documents'] or not results['documents'][0]:
            return ""
            
        # Combine relevant chunks
        context = "\n---\n".join(results['documents'][0])
        return context
