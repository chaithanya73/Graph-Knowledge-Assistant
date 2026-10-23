# Graph-Knowledge-Assistant

An enterprise knowledge assistant using **GraphRAG**, integrating Neo4j knowledge graphs, vector search, and LLMs to retrieve context-aware information from unstructured documents.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   FastAPI     │────▶│  Ingestion   │────▶│   Neo4j      │
│   REST API    │     │  Pipeline    │     │  (Knowledge  │
│               │     │  (Extract    │     │   Graph)     │
└──────┬───────┘     │  Entities)   │     └──────────────┘
       │              └──────────────┘             │
       │                                           │
       ▼                                           ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Query      │────▶│  Graph +     │────▶│   LLM        │
│   Endpoint   │     │  Vector      │     │  (Generate   │
│              │     │  Retrieval   │     │   Answer)    │
└──────────────┘     └──────────────┘     └──────────────┘
```

## Features

- **Document Ingestion**: Upload PDF, DOCX, or TXT files; automatically extracts entities and relationships using LLM
- **Knowledge Graph Construction**: Builds a Neo4j graph with entities, relationships, and document provenance
- **Vector Search**: Semantic similarity search using sentence-transformers embeddings and Neo4j vector index (HNSW)
- **Multi-hop Graph Traversal**: Explore connected knowledge up to N hops deep from any entity
- **GraphRAG Query**: Combines vector search + graph traversal + LLM synthesis for context-aware answers
- **Explainable Retrieval**: Every answer includes the supporting subgraph and text citations

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend Framework | FastAPI |
| Graph Database | Neo4j 5.x |
| Vector Index | Neo4j Native Vector (HNSW) |
| LLM | OpenAI GPT-3.5 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Document Parsing | PyPDF2, python-docx |

## Project Structure

```
app/
├── main.py                     # FastAPI entry point, lifecycle events
├── config.py                   # Centralized settings (pydantic-settings)
├── models/
│   ├── graph.py                # EntityNode, RelationshipEdge, KnowledgeSubgraph
│   └── query.py                # Request/Response DTOs
├── repositories/
│   ├── neo4j_connection.py     # Connection pool manager
│   ├── graph_repository.py     # Cypher queries: CRUD, batch, multi-hop
│   └── vector_repository.py    # Vector index, chunk storage, HNSW search
├── services/
│   ├── text_extractor.py       # PDF/DOCX/TXT extraction + chunking
│   ├── embedding_service.py    # sentence-transformers embeddings
│   ├── entity_extractor.py     # LLM entity/relationship extraction
│   ├── ingestion_service.py    # Full ingestion pipeline orchestrator
│   └── retriever_service.py    # GraphRAG: vector + graph + LLM synthesis
└── routers/
    ├── ingestion.py            # POST /ingest/text, POST /ingest/file
    ├── graph.py                # Graph exploration endpoints
    └── query.py                # POST /query/ask, POST /query/search
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API welcome + endpoint listing |
| `POST` | `/ingest/text` | Ingest raw text into knowledge graph |
| `POST` | `/ingest/file` | Upload and ingest PDF/DOCX/TXT file |
| `GET` | `/graph/stats` | Graph statistics (entity/relationship counts) |
| `GET` | `/graph/entities` | List all entities |
| `GET` | `/graph/entities/{name}` | Get entity details + relationships |
| `POST` | `/graph/entities` | Create/update an entity |
| `POST` | `/graph/relationships` | Create/update a relationship |
| `POST` | `/graph/traverse` | Multi-hop graph traversal |
| `DELETE` | `/graph/entities/{name}` | Delete entity and its edges |
| `POST` | `/query/search` | Semantic vector search |
| `POST` | `/query/ask` | Full GraphRAG query |

## Setup

### Prerequisites
- Python 3.11+
- Neo4j 5.x (local or cloud)
- OpenAI API key (optional, fallback extraction available)

### Installation

```bash
# Clone the repository
git clone https://github.com/chaithanya73/Graph-Knowledge-Assistant.git
cd Graph-Knowledge-Assistant

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Neo4j and OpenAI credentials

# Run the server
uvicorn app.main:app --reload --port 8000
```



## Design Patterns Used

- **Repository Pattern**: Database access encapsulated in repository classes
- **Pipeline Pattern**: Document ingestion flows through chunk → extract → dedupe → store
- **Strategy Pattern**: Different text extractors per file format
- **Singleton Pattern**: Module-level instances for connection and repository objects
- **Graceful Degradation**: App runs without Neo4j or OpenAI (fallback extraction)
