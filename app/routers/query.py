"""
Query Router — The unified GraphRAG query interface.

This is the "star" endpoint of the project — the one you demo in interviews.
It takes a natural language question and returns an answer grounded in both
vector search results AND knowledge graph context.

INTERVIEW: "What makes this different from ChatGPT?"
- ChatGPT answers from its training data (can hallucinate, data is stale).
- Our system answers from YOUR documents stored in a knowledge graph.
- Every answer comes with citations (text chunks) and a reasoning path (subgraph).
- This is enterprise-grade: auditable, explainable, and grounded in your data.
"""

from fastapi import APIRouter, HTTPException
from app.models.query import (
    GraphRAGQueryRequest,
    GraphRAGQueryResponse,
    VectorSearchRequest,
    VectorSearchResultItem,
)
from app.services.retriever_service import retriever_service
from typing import List

router = APIRouter(
    prefix="/query",
    tags=["GraphRAG Query"],
)


@router.post(
    "/ask",
    response_model=GraphRAGQueryResponse,
    summary="Ask a question using GraphRAG",
    description="Submit a natural language question. The system performs vector search "
                "for relevant text chunks AND graph traversal for entity relationships, "
                "then synthesizes an answer using an LLM. Returns the answer with citations "
                "and the supporting knowledge subgraph.",
)
async def graphrag_query(request: GraphRAGQueryRequest):
    """
    The unified GraphRAG query endpoint.

    INTERVIEW: "Walk me through what happens when I call this endpoint."
    1. Embed the question → vector (384 dims).
    2. Vector search → find top-5 relevant text chunks from ingested documents.
    3. Identify seed entities → scan question and chunks for known entity names.
    4. Multi-hop traversal → expand from seed entities through the knowledge graph.
    5. Assemble context → merge text chunks + graph relationships into a prompt.
    6. LLM synthesis → GPT generates a grounded answer citing both sources.
    7. Return → answer + supporting_facts + vector_citations + subgraph.

    The subgraph enables the frontend to render an interactive graph visualization
    showing exactly HOW the system arrived at the answer (explainable AI).
    """
    try:
        result = retriever_service.graphrag_query(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GraphRAG query failed: {str(e)}")


@router.post(
    "/search",
    response_model=List[VectorSearchResultItem],
    summary="Semantic vector search",
    description="Search for relevant document chunks using semantic similarity. "
                "Returns ranked results with cosine similarity scores.",
)
async def vector_search(request: VectorSearchRequest):
    """
    Pure vector search without graph context or LLM synthesis.
    Useful when you just want to find relevant text, not a synthesized answer.

    INTERVIEW: "When would you use /search instead of /ask?"
    - /search: Quick lookup, browsing results, debugging retrieval quality.
    - /ask: Full question answering with reasoning and explainability.
    """
    try:
        results = retriever_service.vector_search(request)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")
