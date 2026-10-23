"""
Graph Router — API endpoints for exploring and managing the knowledge graph.

Provides endpoints for:
- Viewing graph statistics
- Looking up individual entities and their relationships
- Multi-hop graph traversal
- Listing all entities
- Deleting entities

INTERVIEW: "Why separate graph endpoints from query endpoints?"
- Graph endpoints are for EXPLORATION (browsing the graph, debugging, admin).
- Query endpoints are for RETRIEVAL (asking questions, getting answers).
- Different audiences: graph endpoints for developers/admins, query endpoints for end users.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from app.models.graph import EntityNode, RelationshipEdge, KnowledgeSubgraph
from app.models.query import MultiHopQueryRequest
from app.repositories.graph_repository import graph_repository

router = APIRouter(
    prefix="/graph",
    tags=["Knowledge Graph"],
)


@router.get(
    "/stats",
    summary="Get knowledge graph statistics",
    description="Returns total entity and relationship counts in the knowledge graph.",
)
async def get_graph_stats():
    """
    Returns high-level stats: total entities, total relationships.
    Useful for monitoring ingestion progress and graph health.
    """
    try:
        stats = graph_repository.get_graph_stats()
        return {"status": "ok", **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch graph stats: {str(e)}")


@router.get(
    "/entities",
    summary="List all entities in the knowledge graph",
    description="Returns a paginated list of all entity nodes. Use limit parameter to control page size.",
)
async def list_entities(
    limit: int = Query(default=50, ge=1, le=500, description="Maximum number of entities to return"),
):
    """
    INTERVIEW: "Why have a limit parameter?"
    - Without it, a graph with 100K entities would return a 50MB JSON response.
    - Default limit protects both the database (query execution) and the client (parsing).
    - In production, you'd add cursor-based pagination for large datasets.
    """
    try:
        entities = graph_repository.get_all_entities(limit=limit)
        return {"count": len(entities), "entities": entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch entities: {str(e)}")


@router.get(
    "/entities/{entity_name}",
    summary="Get a specific entity and its relationships",
    description="Fetch an entity by name and all its incoming/outgoing relationships.",
)
async def get_entity(entity_name: str):
    """
    Returns an entity's details plus all connected relationships.
    Useful for exploring what an entity is connected to before doing multi-hop traversal.
    """
    try:
        entity = graph_repository.get_entity(entity_name)
        if not entity:
            raise HTTPException(status_code=404, detail=f"Entity '{entity_name}' not found in the knowledge graph")

        relationships = graph_repository.get_entity_relationships(entity_name)
        return {
            "entity": entity,
            "relationships": relationships,
            "relationship_count": len(relationships),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch entity: {str(e)}")


@router.post(
    "/traverse",
    response_model=KnowledgeSubgraph,
    summary="Multi-hop graph traversal",
    description="Starting from a seed entity, traverse the knowledge graph up to N hops deep. "
                "Returns all reachable entities and relationships as a subgraph.",
)
async def multi_hop_traverse(request: MultiHopQueryRequest):
    """
    INTERVIEW: "What is multi-hop traversal?"
    - Start from entity X, follow all relationships 1 hop, then from those results
      follow another hop, up to max_hops.
    - Returns a subgraph (set of nodes + edges) that represents the entity's neighborhood.
    - This is essentially BFS on the knowledge graph, bounded by depth.

    INTERVIEW: "What's the time complexity?"
    - O(b^d) where b = average branching factor, d = max_hops.
    - We cap max_hops at 5 to prevent exponential blowup.
    """
    try:
        subgraph = graph_repository.multi_hop_traversal(
            start_name=request.start_entity,
            max_hops=request.max_hops,
            relationship_types=request.relationship_types,
        )

        if not subgraph.entities:
            raise HTTPException(
                status_code=404,
                detail=f"Entity '{request.start_entity}' not found or has no connections within {request.max_hops} hops",
            )

        return subgraph
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Traversal failed: {str(e)}")


@router.delete(
    "/entities/{entity_name}",
    summary="Delete an entity and its relationships",
    description="Removes an entity node and all connected edges from the knowledge graph.",
)
async def delete_entity(entity_name: str):
    """
    DETACH DELETE in Neo4j removes the node AND all connected relationships.
    This prevents orphan edges (edges pointing to non-existent nodes).
    """
    try:
        deleted = graph_repository.delete_entity(entity_name)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Entity '{entity_name}' not found")
        return {"message": f"Entity '{entity_name}' and its relationships deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@router.post(
    "/entities",
    summary="Manually add an entity to the knowledge graph",
    description="Create or update an entity node. Uses MERGE (upsert) — safe to call multiple times.",
)
async def create_entity(entity: EntityNode):
    """Manually add entities for testing or manual curation."""
    try:
        result = graph_repository.create_or_merge_entity(entity)
        return {"message": f"Entity '{entity.name}' upserted successfully", "entity": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create entity: {str(e)}")


@router.post(
    "/relationships",
    summary="Manually add a relationship between entities",
    description="Create or update a directed relationship. Both source and target entities are auto-created if missing.",
)
async def create_relationship(relationship: RelationshipEdge):
    """Manually add relationships for testing or manual graph curation."""
    try:
        result = graph_repository.create_or_merge_relationship(relationship)
        return {"message": f"Relationship '{relationship.source} -> {relationship.target}' upserted", "relationship": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create relationship: {str(e)}")
