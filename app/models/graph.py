"""
Knowledge Graph Domain Models.

WHY SEPARATE GRAPH PRIMITIVES?
- Represents nodes and relationships independently from the underlying database (Neo4j).
- Decouples our domain models from Neo4j driver types, allowing us to swap databases
  or mock them easily in unit tests without standing up a database instance.

INTERVIEW CONCEPT: Data Transfer Objects (DTO) & Domain-Driven Design (DDD).
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EntityNode(BaseModel):
    """
    Represents an entity (node) extracted from documents.
    Example: EntityNode(name="FastAPI", label="Technology", properties={"category": "Web Framework"})
    """
    name: str = Field(..., description="Unique canonical name of the entity", example="FastAPI")
    label: str = Field(default="Entity", description="Category or node label (e.g., Concept, Technology, Person)")
    description: Optional[str] = Field(default=None, description="Brief description or context of the entity")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary entity metadata")


class RelationshipEdge(BaseModel):
    """
    Represents a directed relationship (edge) between two entities in the knowledge graph.
    Example: FastAPI --[BUILT_WITH]--> Python
    """
    source: str = Field(..., description="Canonical name of the source entity", example="FastAPI")
    target: str = Field(..., description="Canonical name of the target entity", example="Python")
    type: str = Field(..., description="Type of relationship in uppercase", example="BUILT_WITH")
    description: Optional[str] = Field(default=None, description="Contextual explanation of this connection")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence or relationship weight score")


class KnowledgeSubgraph(BaseModel):
    """
    A connected component or subgraph retrieved during multi-hop graph traversals.
    Used for explainable retrieval: shows the user exactly which path in the graph supported the answer.
    """
    entities: List[EntityNode] = Field(default_factory=list, description="Set of nodes in the subgraph")
    relationships: List[RelationshipEdge] = Field(default_factory=list, description="Set of edges connecting the nodes")
