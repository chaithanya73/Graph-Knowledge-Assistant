"""
Graph Repository — Data Access Layer for the Neo4j Knowledge Graph.

This module encapsulates ALL Cypher queries. No other layer (routers, services)
should write raw Cypher. This is the Repository Pattern.

INTERVIEW: "Why the Repository Pattern?"
1. Testability: Mock this class in unit tests without a running Neo4j instance.
2. Single Responsibility: Only this file knows how Neo4j works internally.
3. Swappability: If we migrate from Neo4j to Amazon Neptune, we only change this file.

INTERVIEW: "Why parameterized queries ($name) instead of f-strings?"
- Cypher injection prevention (same concept as SQL injection).
- f"MATCH (n {{name: '{user_input}'}}) RETURN n" → attacker sends: ' }) DETACH DELETE n //
- Parameterized: session.run("MATCH (n {name: $name})", name=user_input) → safe.
"""

from typing import List, Optional, Dict, Any
from app.models.graph import EntityNode, RelationshipEdge, KnowledgeSubgraph
from app.repositories.neo4j_connection import neo4j_connection


class GraphRepository:
    """
    Encapsulates all Neo4j Cypher operations for the knowledge graph.

    Methods follow CRUD + traversal:
    - create_or_merge_entity / create_or_merge_relationship (Create/Update)
    - get_entity / get_all_entities (Read)
    - multi_hop_traversal (Graph Traversal — the core of GraphRAG)
    - batch_upsert_subgraph (Bulk Write)
    - delete_entity (Delete)
    - get_graph_stats (Analytics)
    """

    # ==========================================
    # CREATE / UPDATE (MERGE = upsert in Neo4j)
    # ==========================================

    def create_or_merge_entity(self, entity: EntityNode) -> Dict[str, Any]:
        """
        Create an entity node or update it if it already exists (MERGE = upsert).

        INTERVIEW: "Why MERGE instead of CREATE?"
        - CREATE always inserts a new node → duplicates if called twice with same name.
        - MERGE first checks if a matching node exists:
          - If yes → updates properties (SET).
          - If no  → creates a new node.
        - This is idempotent — calling it 10 times produces the same result as calling it once.
          Idempotency is critical for retry-safe ingestion pipelines.
        """
        query = """
        MERGE (e:Entity {name: $name})
        SET e.label = $label,
            e.description = $description,
            e.properties = $properties
        RETURN e.name AS name, e.label AS label, e.description AS description
        """
        with neo4j_connection.get_session() as session:
            result = session.run(
                query,
                name=entity.name,
                label=entity.label,
                description=entity.description,
                properties=str(entity.properties),  # Neo4j stores as string; deserialize on read
            )
            record = result.single()
            return dict(record) if record else {}

    def create_or_merge_relationship(self, rel: RelationshipEdge) -> Dict[str, Any]:
        """
        Create or update a directed relationship between two entities.
        Also ensures both source and target entities exist (MERGE on both).

        INTERVIEW: "Why MERGE both nodes AND the relationship?"
        - Documents may reference entities we haven't seen before.
        - MERGE guarantees nodes exist before creating the edge.
        - Without this, CREATE (a)-[:REL]->(b) fails if a or b doesn't exist.
        """
        query = """
        MERGE (source:Entity {name: $source})
        MERGE (target:Entity {name: $target})
        MERGE (source)-[r:RELATES_TO {type: $rel_type}]->(target)
        SET r.description = $description,
            r.weight = $weight
        RETURN source.name AS source, target.name AS target, 
               r.type AS type, r.weight AS weight
        """
        with neo4j_connection.get_session() as session:
            result = session.run(
                query,
                source=rel.source,
                target=rel.target,
                rel_type=rel.type,
                description=rel.description,
                weight=rel.weight,
            )
            record = result.single()
            return dict(record) if record else {}

    def batch_upsert_subgraph(self, subgraph: KnowledgeSubgraph) -> Dict[str, int]:
        """
        Batch insert/update an entire subgraph (entities + relationships) in a single transaction.

        INTERVIEW: "Why batch operations?"
        - Individual MERGE calls = N network round-trips (slow for large documents).
        - Batch = 1 transaction with UNWIND → single round-trip for N operations.
        - UNWIND converts a list parameter into rows, so Neo4j processes them in bulk.
        - This is critical for ingestion pipelines processing hundreds of entities per document.
        """
        entities_query = """
        UNWIND $entities AS entity
        MERGE (e:Entity {name: entity.name})
        SET e.label = entity.label,
            e.description = entity.description
        RETURN count(e) AS entities_created
        """

        relationships_query = """
        UNWIND $relationships AS rel
        MERGE (source:Entity {name: rel.source})
        MERGE (target:Entity {name: rel.target})
        MERGE (source)-[r:RELATES_TO {type: rel.type}]->(target)
        SET r.description = rel.description,
            r.weight = rel.weight
        RETURN count(r) AS relationships_created
        """

        entities_data = [
            {"name": e.name, "label": e.label, "description": e.description}
            for e in subgraph.entities
        ]
        relationships_data = [
            {
                "source": r.source,
                "target": r.target,
                "type": r.type,
                "description": r.description,
                "weight": r.weight,
            }
            for r in subgraph.relationships
        ]

        with neo4j_connection.get_session() as session:
            # Use a single transaction for atomicity
            with session.begin_transaction() as tx:
                entity_result = tx.run(entities_query, entities=entities_data)
                entity_count = entity_result.single()["entities_created"]

                rel_result = tx.run(relationships_query, relationships=relationships_data)
                rel_count = rel_result.single()["relationships_created"]

                tx.commit()

            return {
                "entities_upserted": entity_count,
                "relationships_upserted": rel_count,
            }

    # ==========================================
    # READ
    # ==========================================

    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        """Fetch a single entity node by name."""
        query = """
        MATCH (e:Entity {name: $name})
        RETURN e.name AS name, e.label AS label, 
               e.description AS description, e.properties AS properties
        """
        with neo4j_connection.get_session() as session:
            result = session.run(query, name=name)
            record = result.single()
            return dict(record) if record else None

    def get_all_entities(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch all entity nodes with a safety limit to prevent memory exhaustion."""
        query = """
        MATCH (e:Entity)
        RETURN e.name AS name, e.label AS label, e.description AS description
        ORDER BY e.name
        LIMIT $limit
        """
        with neo4j_connection.get_session() as session:
            result = session.run(query, limit=limit)
            return [dict(record) for record in result]

    def get_entity_relationships(self, name: str) -> List[Dict[str, Any]]:
        """
        Get all relationships (incoming and outgoing) for a specific entity.
        Uses undirected match pattern (source)-[r]-(target) to get both directions.
        """
        query = """
        MATCH (e:Entity {name: $name})-[r:RELATES_TO]-(other:Entity)
        RETURN e.name AS entity, r.type AS relationship_type, 
               other.name AS connected_to, r.weight AS weight,
               r.description AS description,
               CASE WHEN startNode(r) = e THEN 'outgoing' ELSE 'incoming' END AS direction
        """
        with neo4j_connection.get_session() as session:
            result = session.run(query, name=name)
            return [dict(record) for record in result]

    # ==========================================
    # GRAPH TRAVERSAL (Core of GraphRAG)
    # ==========================================

    def multi_hop_traversal(
        self,
        start_name: str,
        max_hops: int = 2,
        relationship_types: Optional[List[str]] = None,
    ) -> KnowledgeSubgraph:
        """
        Perform multi-hop traversal starting from an entity.
        Returns a KnowledgeSubgraph containing all reachable nodes and edges within max_hops.

        INTERVIEW: "What is multi-hop traversal? Why is it important for GraphRAG?"
        - Multi-hop = follow relationships N levels deep from a starting node.
        - Example (2-hop from "Python"):
          Hop 1: Python → FastAPI, Python → Django, Python → NumPy
          Hop 2: FastAPI → Pydantic, Django → ORM, NumPy → SciPy
        - This provides CONTEXTUAL understanding that vector search alone cannot:
          "What depends on Python?" requires graph structure, not just text similarity.

        INTERVIEW: "Why use variable-length paths *1..N instead of fixed paths?"
        - Fixed: MATCH (a)-[r1]->(b)-[r2]->(c) only finds exactly 2-hop paths.
        - Variable: MATCH path = (a)-[*1..N]->(b) finds ALL paths from 1 to N hops.
        - More flexible and handles graphs with varying depth.

        INTERVIEW: "What is the time complexity?"
        - Worst case: O(b^d) where b = average branching factor, d = max_hops.
        - For b=10, d=3 → up to 1000 nodes explored. This is why we cap max_hops.
        """
        # Build relationship filter if specified
        rel_filter = ""
        if relationship_types:
            types_str = "|".join(relationship_types)
            rel_filter = f":{types_str}"

        query = f"""
        MATCH path = (start:Entity {{name: $start_name}})-[r:RELATES_TO{rel_filter}*1..{max_hops}]-(end:Entity)
        WITH nodes(path) AS path_nodes, relationships(path) AS path_rels
        UNWIND path_nodes AS node
        WITH COLLECT(DISTINCT node) AS all_nodes, path_rels
        UNWIND path_rels AS rel
        WITH all_nodes, COLLECT(DISTINCT rel) AS all_rels
        UNWIND all_nodes AS n
        WITH COLLECT(DISTINCT {{
            name: n.name, 
            label: n.label, 
            description: n.description
        }}) AS entities, all_rels
        UNWIND all_rels AS r
        RETURN entities,
               COLLECT(DISTINCT {{
                   source: startNode(r).name, 
                   target: endNode(r).name, 
                   type: r.type, 
                   description: r.description, 
                   weight: r.weight
               }}) AS relationships
        """

        with neo4j_connection.get_session() as session:
            result = session.run(query, start_name=start_name)
            record = result.single()

            if not record:
                return KnowledgeSubgraph(entities=[], relationships=[])

            entities = [
                EntityNode(
                    name=e["name"],
                    label=e.get("label", "Entity"),
                    description=e.get("description"),
                )
                for e in record["entities"]
            ]

            relationships = [
                RelationshipEdge(
                    source=r["source"],
                    target=r["target"],
                    type=r.get("type", "RELATES_TO"),
                    description=r.get("description"),
                    weight=r.get("weight", 1.0),
                )
                for r in record["relationships"]
            ]

            return KnowledgeSubgraph(entities=entities, relationships=relationships)

    # ==========================================
    # DELETE
    # ==========================================

    def delete_entity(self, name: str) -> bool:
        """
        Delete an entity and all its relationships.
        DETACH DELETE removes the node AND all connected edges (prevents orphan edges).
        """
        query = """
        MATCH (e:Entity {name: $name})
        DETACH DELETE e
        RETURN count(e) AS deleted
        """
        with neo4j_connection.get_session() as session:
            result = session.run(query, name=name)
            record = result.single()
            return record["deleted"] > 0 if record else False

    # ==========================================
    # ANALYTICS / STATS
    # ==========================================

    def get_graph_stats(self) -> Dict[str, int]:
        """Get high-level graph statistics — useful for monitoring and dashboard."""
        query = """
        MATCH (e:Entity)
        WITH count(e) AS entity_count
        OPTIONAL MATCH ()-[r:RELATES_TO]->()
        RETURN entity_count, count(r) AS relationship_count
        """
        with neo4j_connection.get_session() as session:
            result = session.run(query)
            record = result.single()
            if record:
                return {
                    "total_entities": record["entity_count"],
                    "total_relationships": record["relationship_count"],
                }
            return {"total_entities": 0, "total_relationships": 0}


# Singleton instance
graph_repository = GraphRepository()
