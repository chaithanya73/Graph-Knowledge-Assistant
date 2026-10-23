"""
GraphRAG Retriever Service — The core intelligence of the system.

Combines three retrieval strategies and synthesizes a final answer using an LLM:

1. Vector Search: Find semantically similar text chunks (traditional RAG).
2. Graph Traversal: Follow entity relationships for multi-hop context (GraphRAG).
3. LLM Synthesis: Feed both contexts to an LLM to generate a grounded, explainable answer.

INTERVIEW: "What is GraphRAG and how is it different from regular RAG?"
- Regular RAG: query → vector search → top-K text chunks → LLM generates answer.
  Problem: Only sees isolated text snippets. No understanding of HOW entities relate.
- GraphRAG: query → vector search + graph traversal → text chunks + entity relationships → LLM.
  Advantage: The LLM sees "FastAPI USES Pydantic, Pydantic VALIDATES data" — structured context.
  This enables multi-hop reasoning: "What does FastAPI use for validation?" → follows the path.

INTERVIEW: "Why combine vector AND graph retrieval?"
- Vector search excels at finding RELEVANT text (semantic similarity).
- Graph traversal excels at finding CONNECTED context (structural relationships).
- Example: "What are the dependencies of our ML pipeline?"
  - Vector search finds chunks mentioning "ML pipeline".
  - Graph traversal follows: ML Pipeline → TensorFlow → CUDA → GPU Driver.
  - Together, the LLM can give a complete dependency chain, not just text fragments.

INTERVIEW: "How do you handle the case where vector and graph results conflict?"
- We don't resolve conflicts — we present BOTH to the LLM with source attribution.
- The LLM acts as a "judge" that weighs evidence from both sources.
- The response includes both vector_citations and the subgraph, so the user
  can verify the answer themselves (explainable AI).
"""

import json
from typing import List, Optional
from openai import OpenAI
from app.config import settings
from app.models.graph import KnowledgeSubgraph, EntityNode
from app.models.query import (
    GraphRAGQueryRequest,
    GraphRAGQueryResponse,
    VectorSearchRequest,
    VectorSearchResultItem,
    MultiHopQueryRequest,
)
from app.repositories.graph_repository import graph_repository
from app.repositories.vector_repository import vector_repository


# System prompt for the answer synthesis LLM
ANSWER_SYNTHESIS_PROMPT = """You are a knowledge assistant that answers questions using provided context.

You will receive two types of context:
1. TEXT CONTEXT: Relevant text chunks found via semantic search.
2. GRAPH CONTEXT: Entity relationships from a knowledge graph.

Rules:
1. ONLY answer based on the provided context. If the context doesn't contain enough information, say so.
2. Cite your sources: mention which text chunks or graph relationships support your answer.
3. If the graph context reveals multi-hop connections, explain the reasoning chain.
4. Be concise but thorough.
5. If text and graph context provide complementary information, synthesize them.
"""


class RetrieverService:
    """
    Unified GraphRAG retriever that combines vector search, graph traversal,
    and LLM synthesis into a single query interface.
    """

    def vector_search(self, request: VectorSearchRequest) -> List[VectorSearchResultItem]:
        """
        Perform semantic vector search over stored document chunks.

        INTERVIEW: "What happens under the hood?"
        1. User query → embedding vector (384 dims via sentence-transformers).
        2. Neo4j HNSW index finds top-K nearest chunk vectors (cosine similarity).
        3. Return chunks with scores, ordered by relevance.
        """
        results = vector_repository.vector_search(
            query_text=request.query,
            top_k=request.top_k,
        )

        return [
            VectorSearchResultItem(
                chunk_id=r.get("chunk_id", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                document_title=r.get("document_title"),
            )
            for r in results
        ]

    def graph_traversal(self, request: MultiHopQueryRequest) -> KnowledgeSubgraph:
        """
        Perform multi-hop graph traversal from a starting entity.

        INTERVIEW: "Give an example of multi-hop reasoning."
        Query: "What technologies depend on Python?"
        Hop 1: Python → FastAPI, Python → Django, Python → NumPy
        Hop 2: FastAPI → Pydantic, FastAPI → Uvicorn, Django → PostgreSQL
        Result: A subgraph showing Python's entire dependency tree up to N hops.
        """
        return graph_repository.multi_hop_traversal(
            start_name=request.start_entity,
            max_hops=request.max_hops,
            relationship_types=request.relationship_types,
        )

    def graphrag_query(self, request: GraphRAGQueryRequest) -> GraphRAGQueryResponse:
        """
        The main GraphRAG pipeline — combines everything.

        INTERVIEW: "Walk me through the full GraphRAG query flow."

        Step 1: Vector Search
        - Convert the question to an embedding and find the top-5 most relevant text chunks.

        Step 2: Entity Identification
        - From the vector search results, identify entity names mentioned in the chunks.
        - These become "seed entities" for graph traversal.

        Step 3: Graph Traversal
        - For each seed entity, perform multi-hop traversal (default 2 hops).
        - Collect all connected entities and relationships into a subgraph.

        Step 4: Context Assembly
        - Combine text chunks (from vector search) and graph relationships into a prompt.

        Step 5: LLM Synthesis
        - Send the assembled context + user question to the LLM.
        - The LLM generates a grounded answer with citations.

        Step 6: Response
        - Return the answer + supporting facts + vector citations + the subgraph.
        - The subgraph enables the frontend to visualize the reasoning path.
        """
        # Step 1: Vector search for relevant text chunks
        vector_results = vector_repository.vector_search(
            query_text=request.question,
            top_k=5,
        )

        vector_citations = [
            VectorSearchResultItem(
                chunk_id=r.get("chunk_id", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                document_title=r.get("document_title"),
            )
            for r in vector_results
        ]

        # Step 2: Identify seed entities from vector results and graph
        seed_entities = self._identify_seed_entities(request.question, vector_results)

        # Step 3: Multi-hop graph traversal from each seed entity
        combined_subgraph = KnowledgeSubgraph(entities=[], relationships=[])
        seen_entity_names = set()
        seen_rel_keys = set()

        for entity_name in seed_entities:
            subgraph = graph_repository.multi_hop_traversal(
                start_name=entity_name,
                max_hops=request.max_graph_hops,
            )
            # Merge into combined subgraph (deduplicate)
            for entity in subgraph.entities:
                if entity.name not in seen_entity_names:
                    seen_entity_names.add(entity.name)
                    combined_subgraph.entities.append(entity)
            for rel in subgraph.relationships:
                rel_key = f"{rel.source}-{rel.type}-{rel.target}"
                if rel_key not in seen_rel_keys:
                    seen_rel_keys.add(rel_key)
                    combined_subgraph.relationships.append(rel)

        # Step 4 & 5: Assemble context and generate answer with LLM
        answer, supporting_facts = self._synthesize_answer(
            question=request.question,
            vector_citations=vector_citations,
            subgraph=combined_subgraph,
        )

        return GraphRAGQueryResponse(
            question=request.question,
            answer=answer,
            supporting_facts=supporting_facts,
            vector_citations=vector_citations,
            subgraph=combined_subgraph if request.include_subgraph else None,
        )

    def _identify_seed_entities(
        self,
        question: str,
        vector_results: List[dict],
    ) -> List[str]:
        """
        Identify entity names to use as starting points for graph traversal.

        Strategy:
        1. Search the graph for entities whose names appear in the question.
        2. Also check entities mentioned in the top vector search results.
        3. Return unique entity names (capped at 5 to limit traversal scope).

        INTERVIEW: "Why cap at 5 seed entities?"
        - Each seed entity triggers a multi-hop traversal (O(b^d) complexity).
        - 5 seeds × 2 hops × branching factor 10 = up to 500 nodes explored.
        - More seeds = more context but slower response and more noise.
        """
        seed_entities = set()

        # Check which known entities appear in the question
        all_entities = graph_repository.get_all_entities(limit=500)
        question_lower = question.lower()
        for entity in all_entities:
            if entity["name"].lower() in question_lower:
                seed_entities.add(entity["name"])

        # Also extract entity names from top vector results
        for result in vector_results[:3]:
            content = result.get("content", "")
            for entity in all_entities:
                if entity["name"].lower() in content.lower():
                    seed_entities.add(entity["name"])

        return list(seed_entities)[:5]

    def _synthesize_answer(
        self,
        question: str,
        vector_citations: List[VectorSearchResultItem],
        subgraph: KnowledgeSubgraph,
    ) -> tuple:
        """
        Use LLM to synthesize a final answer from vector and graph context.

        Returns:
            Tuple of (answer_text, list_of_supporting_facts)
        """
        # Build text context from vector search results
        text_context = ""
        if vector_citations:
            text_context = "TEXT CONTEXT (from semantic search):\n"
            for i, citation in enumerate(vector_citations, 1):
                text_context += f"\n[Chunk {i}] (relevance: {citation.score:.2f}):\n{citation.content}\n"

        # Build graph context from subgraph
        graph_context = ""
        if subgraph.relationships:
            graph_context = "\nGRAPH CONTEXT (from knowledge graph):\n"
            graph_context += "Entities: " + ", ".join(
                f"{e.name} ({e.label})" for e in subgraph.entities
            ) + "\n"
            graph_context += "Relationships:\n"
            for rel in subgraph.relationships:
                graph_context += f"  - {rel.source} --[{rel.type}]--> {rel.target}"
                if rel.description:
                    graph_context += f" ({rel.description})"
                graph_context += "\n"

        # Extract supporting facts from graph
        supporting_facts = [
            f"{rel.source} {rel.type.replace('_', ' ').lower()} {rel.target}"
            for rel in subgraph.relationships
        ]

        # If no OpenAI key, return a structured summary without LLM
        if not settings.openai_api_key:
            return self._fallback_answer(question, text_context, graph_context, supporting_facts)

        # LLM synthesis
        try:
            client = OpenAI(api_key=settings.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": ANSWER_SYNTHESIS_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\n"
                            f"{text_context}\n"
                            f"{graph_context}\n"
                            f"Please answer the question based on the above context."
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=1000,
            )
            answer = response.choices[0].message.content.strip()
            return answer, supporting_facts

        except Exception as e:
            print(f"[RETRIEVER] LLM synthesis failed: {e}")
            return self._fallback_answer(question, text_context, graph_context, supporting_facts)

    def _fallback_answer(
        self,
        question: str,
        text_context: str,
        graph_context: str,
        supporting_facts: List[str],
    ) -> tuple:
        """
        Generate a structured answer without LLM when OpenAI is unavailable.
        """
        parts = [f"Based on the available knowledge about '{question}':\n"]

        if supporting_facts:
            parts.append("From the knowledge graph:")
            for fact in supporting_facts[:10]:
                parts.append(f"  - {fact}")

        if text_context:
            parts.append("\nRelevant text excerpts were found via semantic search.")
            parts.append("(Full LLM synthesis unavailable — set OPENAI_API_KEY for detailed answers)")

        if not supporting_facts and not text_context:
            parts.append("No relevant information found in the knowledge base for this query.")
            parts.append("Try ingesting documents first using POST /ingest/text or POST /ingest/file.")

        answer = "\n".join(parts)
        return answer, supporting_facts


# Singleton
retriever_service = RetrieverService()
