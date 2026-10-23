"""
FastAPI Application Entry Point.

WHY FastAPI over Flask/Django?
- Async support out of the box (important for I/O heavy operations like DB calls, LLM API calls)
- Auto-generated Swagger/OpenAPI docs at /docs
- Built-in request validation with Pydantic
- Type hints everywhere = fewer bugs, better IDE support

INTERVIEW TIP: "I chose FastAPI because our app is I/O bound --
we wait on Neo4j queries, LLM API calls, and file parsing.
Async allows us to handle multiple requests concurrently
without spawning threads."
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.repositories.neo4j_connection import neo4j_connection
from app.repositories.vector_repository import vector_repository
from app.routers import ingestion, graph, query

# Create the FastAPI application instance
app = FastAPI(
    title="Graph Knowledge Assistant",
    description=(
        "An enterprise knowledge assistant using GraphRAG. "
        "Integrates Neo4j knowledge graphs, vector search, and LLMs "
        "to retrieve context-aware information from unstructured documents."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routers
app.include_router(ingestion.router)
app.include_router(graph.router)
app.include_router(query.router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint -- API welcome message."""
    return {
        "message": "Welcome to the Graph Knowledge Assistant API",
        "docs": "/docs",
        "endpoints": {
            "ingest_text": "POST /ingest/text",
            "ingest_file": "POST /ingest/file",
            "graph_stats": "GET /graph/stats",
            "graph_entities": "GET /graph/entities",
            "graph_traverse": "POST /graph/traverse",
            "vector_search": "POST /query/search",
            "graphrag_ask": "POST /query/ask",
        },
    }


# Lifecycle events
@app.on_event("startup")
async def startup_event():
    """Initialize Neo4j connection pool and vector index on app startup."""
    print(f"[START] Starting Graph Knowledge Assistant in {settings.app_env} mode")
    try:
        neo4j_connection.connect()
        print("[NEO4J] Connection pool initialized successfully")
        # Ensure vector index exists for semantic search
        vector_repository.ensure_vector_index()
    except Exception as e:
        print(f"[NEO4J] WARNING: Could not connect to Neo4j: {e}")
        print("[NEO4J] App will run but graph/vector features will be unavailable")


@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully close Neo4j connections on app shutdown."""
    neo4j_connection.close()
    print("[STOP] Shutting down Graph Knowledge Assistant")
