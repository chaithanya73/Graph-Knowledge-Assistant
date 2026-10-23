"""
Vector Repository — Store and search vector embeddings in Neo4j.

Neo4j 5.x has native vector index support, so we don't need a separate vector database
(Pinecone, Weaviate, ChromaDB). This simplifies our architecture — one database for both
graph structure AND vector search.

INTERVIEW: "Why not use a separate vector database?"
- Fewer moving parts = simpler deployment, fewer failure points, less operational cost.
- Neo4j's vector index uses HNSW (Hierarchical Navigable Small World) algorithm internally.
- For enterprise scale (millions of vectors), a dedicated vector DB might perform better.
- For our use case (thousands to tens of thousands of chunks), Neo4j is more than sufficient.

INTERVIEW: "What is HNSW?"
- Approximate Nearest Neighbor (ANN) algorithm — doesn't check ALL vectors, uses a graph
  structure to quickly navigate to the region of similar vectors.
- O(log N) average search time instead of O(N) brute-force.
- Trade-off: ~95-99% recall (might miss the absolute best match) but 100x faster.
"""

from typing import List, Dict, Any, Optional
from app.repositories.neo4j_connection import neo4j_connection
from app.services.embedding_service import generate_embedding, generate_embeddings_batch


class VectorRepository:
    """
    Manages vector storage and similarity search using Neo4j's native vector index.

    Stores text chunks as :Chunk nodes with an 'embedding' property,
    and creates a vector index for fast similarity search.
    """

    def ensure_vector_index(self) -> None:
        """
        Create the vector index if it doesn't exist.
        Called once at app startup.

        INTERVIEW: "What does this Cypher do?"
        - Creates an index named 'chunk_embedding_index' on :Chunk nodes.
        - Uses the 'embedding' property for the vector data.
        - 384 dimensions (matching our MiniLM model).
        - Cosine similarity metric (best for normalized embeddings).
        """
        query = """
        CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
        FOR (c:Chunk)
        ON (c.embedding)
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }
        }
        """
        try:
            with neo4j_connection.get_session() as session:
                session.run(query)
            print("[VECTOR] Vector index 'chunk_embedding_index' ensured")
        except Exception as e:
            print(f"[VECTOR] Could not create vector index: {e}")

    def store_chunk(
        self,
        chunk_id: str,
        content: str,
        document_title: str,
        embedding: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Store a text chunk with its vector embedding as a :Chunk node in Neo4j.
        Also links it to its source document.

        INTERVIEW: "Why store the raw text alongside the embedding?"
        - The embedding is for SEARCH (finding similar chunks).
        - The raw text is for DISPLAY (showing the user what was found).
        - You can't reconstruct readable text from an embedding vector.
        """
        if embedding is None:
            embedding = generate_embedding(content)

        query = """
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c.content = $content,
            c.document_title = $document_title,
            c.embedding = $embedding
        WITH c
        MERGE (d:Entity {name: $document_title})
        MERGE (d)-[:HAS_CHUNK]->(c)
        RETURN c.chunk_id AS chunk_id
        """
        with neo4j_connection.get_session() as session:
            result = session.run(
                query,
                chunk_id=chunk_id,
                content=content,
                document_title=document_title,
                embedding=embedding,
            )
            record = result.single()
            return dict(record) if record else {}

    def store_chunks_batch(
        self,
        chunks: List[Dict[str, str]],
        document_title: str,
    ) -> int:
        """
        Batch store multiple chunks with their embeddings.

        INTERVIEW: "Why batch?"
        - Generate all embeddings in one GPU pass (batch_encode).
        - Store all chunks in one Neo4j transaction (UNWIND).
        - For 50 chunks: batch = ~2 seconds, individual = ~25 seconds.
        """
        if not chunks:
            return 0

        # Generate embeddings for all chunks in one batch
        texts = [c["content"] for c in chunks]
        embeddings = generate_embeddings_batch(texts)

        # Prepare data for UNWIND
        chunk_data = [
            {
                "chunk_id": c["chunk_id"],
                "content": c["content"],
                "document_title": document_title,
                "embedding": emb,
            }
            for c, emb in zip(chunks, embeddings)
        ]

        query = """
        UNWIND $chunks AS chunk
        MERGE (c:Chunk {chunk_id: chunk.chunk_id})
        SET c.content = chunk.content,
            c.document_title = chunk.document_title,
            c.embedding = chunk.embedding
        WITH c, chunk
        MERGE (d:Entity {name: chunk.document_title})
        MERGE (d)-[:HAS_CHUNK]->(c)
        RETURN count(c) AS stored_count
        """
        with neo4j_connection.get_session() as session:
            result = session.run(query, chunks=chunk_data)
            record = result.single()
            return record["stored_count"] if record else 0

    def vector_search(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search using Neo4j's vector index.

        INTERVIEW: "Walk me through vector search."
        1. Convert the user's query text into an embedding vector (384 dims).
        2. Use Neo4j's db.index.vector.queryNodes() to find the top-K most similar chunks.
        3. Neo4j internally uses HNSW to efficiently find nearest neighbors.
        4. Return the chunks ranked by cosine similarity score.

        INTERVIEW: "What is cosine similarity?"
        - Measures the angle between two vectors, not the distance.
        - cos(0) = 1.0 (identical direction = identical meaning).
        - cos(90) = 0.0 (perpendicular = unrelated).
        - We use normalized vectors, so cosine similarity = dot product.
        """
        query_embedding = generate_embedding(query_text)

        query = """
        CALL db.index.vector.queryNodes('chunk_embedding_index', $top_k, $query_embedding)
        YIELD node, score
        RETURN node.chunk_id AS chunk_id,
               node.content AS content,
               node.document_title AS document_title,
               score
        ORDER BY score DESC
        """
        with neo4j_connection.get_session() as session:
            result = session.run(
                query,
                top_k=top_k,
                query_embedding=query_embedding,
            )
            return [dict(record) for record in result]


# Singleton
vector_repository = VectorRepository()
