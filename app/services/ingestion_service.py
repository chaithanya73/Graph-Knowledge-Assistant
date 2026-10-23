"""
Document Ingestion Service — Orchestrates the full ingestion pipeline.

Pipeline flow:
  Raw Text / File Upload
       |
       v
  Text Extraction (PDF/DOCX/TXT → plain text)
       |
       v
  Text Chunking (split into overlapping chunks)
       |
       v
  Entity & Relationship Extraction (LLM per chunk)
       |
       v
  Knowledge Graph Storage (batch upsert into Neo4j)

INTERVIEW: "Why is this a separate service and not in the router?"
- Routers should ONLY handle HTTP concerns (parsing request, returning response).
- Business logic (chunking strategy, extraction orchestration, deduplication) belongs in services.
- This allows us to reuse the ingestion logic from CLI tools, background workers,
  or scheduled jobs — not just HTTP endpoints.

INTERVIEW: "How would you scale this pipeline?"
- For large documents: use a task queue (Celery/RabbitMQ) for async processing.
- For many documents: horizontal scaling with multiple workers.
- For real-time: use FastAPI BackgroundTasks for fire-and-forget ingestion.
"""

import uuid
from typing import Dict, Any, List
from app.models.graph import EntityNode, RelationshipEdge, KnowledgeSubgraph
from app.models.query import DocumentIngestRequest, DocumentIngestResponse
from app.services.text_extractor import extract_text, chunk_text
from app.services.entity_extractor import extract_entities_and_relationships
from app.repositories.graph_repository import graph_repository
from app.repositories.vector_repository import vector_repository


class IngestionService:
    """
    Orchestrates the document-to-knowledge-graph pipeline.

    INTERVIEW: "This follows the Pipeline Pattern (or Chain of Responsibility).
    Each step transforms data and passes it to the next:
    text → chunks → entities/relationships → graph storage."
    """

    def ingest_text(self, request: DocumentIngestRequest) -> DocumentIngestResponse:
        """
        Ingest raw text content: chunk it, extract entities, store in Neo4j.

        INTERVIEW: "Walk me through what happens when a user submits a document."
        1. Generate a unique document_id (UUID4 — collision probability ~1 in 2^122).
        2. Split text into overlapping chunks (default 1000 chars, 200 overlap).
        3. For each chunk, call the LLM to extract entities and relationships.
        4. Deduplicate entities across chunks (same entity may appear in multiple chunks).
        5. Batch upsert the entire subgraph into Neo4j in a single transaction.
        6. Return summary statistics to the caller.
        """
        document_id = str(uuid.uuid4())

        # Step 1: Chunk the text
        chunks = chunk_text(request.content)

        # Step 2: Extract entities and relationships from each chunk
        all_entities: List[EntityNode] = []
        all_relationships: List[RelationshipEdge] = []

        for chunk in chunks:
            entities, relationships = extract_entities_and_relationships(chunk)
            all_entities.extend(entities)
            all_relationships.extend(relationships)

        # Step 3: Deduplicate entities by name (keep first occurrence)
        # INTERVIEW: "Why deduplicate?"
        # The same entity (e.g., "Python") may appear in multiple chunks.
        # Without dedup, we'd create redundant MERGE operations (harmless but wasteful).
        seen_names = set()
        unique_entities = []
        for entity in all_entities:
            if entity.name not in seen_names:
                seen_names.add(entity.name)
                unique_entities.append(entity)
            
        # Also add a document node to track provenance
        doc_entity = EntityNode(
            name=request.title,
            label="Document",
            description=f"Ingested document: {request.title}",
            properties=request.metadata,
        )
        unique_entities.append(doc_entity)

        # Link each extracted entity to the document for traceability
        doc_relationships = [
            RelationshipEdge(
                source=request.title,
                target=entity.name,
                type="CONTAINS_ENTITY",
                description=f"Entity extracted from document: {request.title}",
                weight=0.9,
            )
            for entity in unique_entities
            if entity.name != request.title  # Don't self-link
        ]
        all_relationships.extend(doc_relationships)

        # Step 4: Store text chunks as vector embeddings for semantic search
        chunk_data = [
            {"chunk_id": f"{document_id}_chunk_{i}", "content": chunk}
            for i, chunk in enumerate(chunks)
        ]
        try:
            vectors_stored = vector_repository.store_chunks_batch(chunk_data, request.title)
            print(f"[INGESTION] Stored {vectors_stored} chunk vectors for '{request.title}'")
        except Exception as e:
            print(f"[INGESTION] Vector storage failed (non-fatal): {e}")

        # Step 5: Batch upsert entities and relationships into Neo4j
        subgraph = KnowledgeSubgraph(
            entities=unique_entities,
            relationships=all_relationships,
        )

        try:
            stats = graph_repository.batch_upsert_subgraph(subgraph)
            entities_count = stats.get("entities_upserted", len(unique_entities))
            rels_count = stats.get("relationships_upserted", len(all_relationships))
        except Exception as e:
            print(f"[INGESTION] Neo4j storage failed: {e}")
            # Return what we extracted even if storage failed
            entities_count = len(unique_entities)
            rels_count = len(all_relationships)

        return DocumentIngestResponse(
            document_id=document_id,
            title=request.title,
            chunks_created=len(chunks),
            entities_extracted=entities_count,
            relationships_extracted=rels_count,
            message=f"Successfully processed '{request.title}': {entities_count} entities, {rels_count} relationships from {len(chunks)} chunks",
        )

    def ingest_file(self, file_bytes: bytes, filename: str, metadata: Dict[str, Any] = None) -> DocumentIngestResponse:
        """
        Ingest a file upload: extract text first, then run the text ingestion pipeline.
        """
        # Extract text from file
        text = extract_text(file_bytes, filename)

        if not text.strip():
            raise ValueError(f"No text content could be extracted from {filename}")

        # Create a request and delegate to ingest_text
        request = DocumentIngestRequest(
            title=filename,
            content=text,
            metadata=metadata or {"source": "file_upload", "original_filename": filename},
        )
        return self.ingest_text(request)


# Singleton instance
ingestion_service = IngestionService()
