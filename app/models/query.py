"""
API Request and Response Models for Ingestion and Retrieval.

SDE INTERVIEW CONCEPT: Strict Contract Boundaries.
Using strongly typed response models ensures backward compatibility
and prevents internal database details from leaking directly to API clients.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.models.graph import EntityNode, RelationshipEdge, KnowledgeSubgraph


# ==========================================
# Ingestion Models
# ==========================================

class DocumentIngestRequest(BaseModel):
    """Payload when sending raw text directly for graph extraction."""
    title: str = Field(..., description="Document title", example="Microservices Architecture Overview")
    content: str = Field(..., description="Raw text content to process into knowledge graph")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata tags (author, department, etc.)")


class DocumentIngestResponse(BaseModel):
    """Response after extracting and persisting graph entities."""
    document_id: str = Field(..., description="Unique generated document identifier")
    title: str = Field(..., description="Document title")
    chunks_created: int = Field(..., description="Number of text chunks created for vector storage")
    entities_extracted: int = Field(..., description="Total entity nodes identified")
    relationships_extracted: int = Field(..., description="Total relationship edges identified")
    message: str = Field(default="Document successfully ingested into knowledge graph")


# ==========================================
# Retrieval & Query Models
# ==========================================

class MultiHopQueryRequest(BaseModel):
    """Request for exploring connected knowledge starting from an entity."""
    start_entity: str = Field(..., description="The seed entity name to begin traversal from", example="FastAPI")
    max_hops: int = Field(default=2, ge=1, le=5, description="Graph traversal depth (multi-hop path length)")
    relationship_types: Optional[List[str]] = Field(default=None, description="Optional filter for edge types")


class VectorSearchRequest(BaseModel):
    """Request for semantic vector search over document chunks."""
    query: str = Field(..., description="Natural language search query", example="How does dependency injection work?")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of top matches to return")


class VectorSearchResultItem(BaseModel):
    """Single item retrieved via vector similarity search."""
    chunk_id: str
    content: str
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    document_title: Optional[str] = None


class GraphRAGQueryRequest(BaseModel):
    """
    Unified GraphRAG query request:
    Combines both vector search and graph multi-hop traversal to synthesize a comprehensive answer.
    """
    question: str = Field(..., description="The user's natural language question", example="What technologies depend on Python in our stack?")
    max_graph_hops: int = Field(default=2, ge=1, le=4, description="Depth of graph expansion for context")
    include_subgraph: bool = Field(default=True, description="Whether to return the subgraph for visual explainability")


class GraphRAGQueryResponse(BaseModel):
    """
    Unified GraphRAG response.
    Provides explainable retrieval: the synthesized answer + direct text citations + graph connections.
    """
    question: str
    answer: str = Field(..., description="Synthesized answer generated with context-aware LLM reasoning")
    supporting_facts: List[str] = Field(default_factory=list, description="Extracted factual claims from the graph")
    vector_citations: List[VectorSearchResultItem] = Field(default_factory=list, description="Relevant text chunks retrieved")
    subgraph: Optional[KnowledgeSubgraph] = Field(default=None, description="Graph context used for explainable retrieval")
