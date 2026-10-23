"""
Entity & Relationship Extraction Service using OpenAI LLM.

This is the "intelligence" layer — it reads raw text and identifies:
1. Entities (people, technologies, concepts, organizations)
2. Relationships between them (uses, depends_on, created_by, etc.)

INTERVIEW: "Why use an LLM for extraction instead of NER (Named Entity Recognition)?"
- Traditional NER (spaCy, NLTK) only extracts entity NAMES and types (PERSON, ORG).
- It does NOT extract RELATIONSHIPS between entities.
- LLMs can understand context and output structured JSON with both entities AND relationships.
- Example: "Guido van Rossum created Python in 1991" →
  Entities: [Guido van Rossum (Person), Python (Language)]
  Relationships: [Guido van Rossum --CREATED--> Python]

INTERVIEW: "What about hallucination?"
- The LLM might invent relationships not in the text. We mitigate this by:
  1. Instructing it to ONLY extract from the given text (grounding).
  2. Assigning confidence weights to relationships.
  3. In production, adding a human-in-the-loop validation step.
"""

import json
from typing import Tuple, List
from openai import OpenAI
from app.config import settings
from app.models.graph import EntityNode, RelationshipEdge


# System prompt that instructs the LLM on extraction format
EXTRACTION_SYSTEM_PROMPT = """You are a knowledge graph extraction expert.
Given a text chunk, extract ALL entities and relationships mentioned.

Rules:
1. ONLY extract information explicitly stated in the text. Do NOT infer or hallucinate.
2. Entity names should be canonical (e.g., "Python" not "python programming language").
3. Relationship types should be uppercase with underscores (e.g., USES, DEPENDS_ON, CREATED_BY).
4. Assign a confidence weight (0.0 to 1.0) to each relationship.

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{
    "entities": [
        {"name": "EntityName", "label": "Category", "description": "Brief description from text"}
    ],
    "relationships": [
        {"source": "Entity1", "target": "Entity2", "type": "RELATIONSHIP_TYPE", "description": "Context from text", "weight": 0.9}
    ]
}

Categories for labels: Person, Organization, Technology, Concept, Location, Event, Product, Other
"""


def extract_entities_and_relationships(
    text_chunk: str,
) -> Tuple[List[EntityNode], List[RelationshipEdge]]:
    """
    Use OpenAI GPT to extract entities and relationships from a text chunk.

    INTERVIEW: "Walk me through the extraction pipeline."
    1. We send the text chunk + a structured system prompt to GPT.
    2. GPT returns JSON with entities and relationships.
    3. We parse the JSON into our Pydantic domain models (EntityNode, RelationshipEdge).
    4. Pydantic validates the data — if GPT returns malformed JSON, we catch it gracefully.
    5. These validated models are then passed to GraphRepository for Neo4j storage.

    Returns:
        Tuple of (list of EntityNodes, list of RelationshipEdges)
    """
    if not settings.openai_api_key:
        # Fallback: if no API key, use simple rule-based extraction
        return _fallback_extraction(text_chunk)

    client = OpenAI(api_key=settings.openai_api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract entities and relationships from this text:\n\n{text_chunk}"},
            ],
            temperature=0.0,  # Deterministic output — no creativity needed for extraction
            max_tokens=2000,
        )

        raw_output = response.choices[0].message.content.strip()

        # Parse the JSON response
        # Handle cases where LLM wraps in ```json ... ```
        if raw_output.startswith("```"):
            raw_output = raw_output.split("```")[1]
            if raw_output.startswith("json"):
                raw_output = raw_output[4:]
            raw_output = raw_output.strip()

        data = json.loads(raw_output)

        entities = [
            EntityNode(
                name=e.get("name", "Unknown"),
                label=e.get("label", "Entity"),
                description=e.get("description"),
            )
            for e in data.get("entities", [])
        ]

        relationships = [
            RelationshipEdge(
                source=r.get("source", ""),
                target=r.get("target", ""),
                type=r.get("type", "RELATES_TO"),
                description=r.get("description"),
                weight=min(max(r.get("weight", 0.8), 0.0), 1.0),  # Clamp to [0, 1]
            )
            for r in data.get("relationships", [])
            if r.get("source") and r.get("target")  # Skip incomplete relationships
        ]

        return entities, relationships

    except json.JSONDecodeError as e:
        print(f"[ENTITY_EXTRACTOR] Failed to parse LLM JSON output: {e}")
        return _fallback_extraction(text_chunk)
    except Exception as e:
        print(f"[ENTITY_EXTRACTOR] LLM extraction failed: {e}")
        return _fallback_extraction(text_chunk)


def _fallback_extraction(
    text_chunk: str,
) -> Tuple[List[EntityNode], List[RelationshipEdge]]:
    """
    Simple rule-based fallback when OpenAI is unavailable.
    Extracts capitalized multi-word phrases as potential entities.

    INTERVIEW: "Why have a fallback?"
    - Graceful degradation: the system still works (partially) without the LLM.
    - Useful for development/testing without burning API credits.
    - In production, you'd log this as a degraded-mode alert.
    """
    import re

    # Find capitalized phrases (2+ words starting with uppercase)
    # e.g., "Machine Learning", "Graph Database", "Neo4j"
    pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    matches = list(set(re.findall(pattern, text_chunk)))

    # Filter out common English words that happen to be capitalized (sentence starters)
    stop_words = {
        "The", "This", "That", "These", "Those", "Here", "There",
        "When", "Where", "What", "Which", "How", "Why", "Who",
        "Each", "Every", "Some", "Many", "Most", "Other", "Such",
        "However", "Therefore", "Furthermore", "Moreover", "Also",
    }
    matches = [m for m in matches if m not in stop_words and len(m) > 2]

    entities = [
        EntityNode(name=name, label="Concept", description=f"Extracted from text: {name}")
        for name in matches[:20]  # Cap at 20 to prevent noise
    ]

    # No relationships in fallback mode — we need LLM for that
    return entities, []
