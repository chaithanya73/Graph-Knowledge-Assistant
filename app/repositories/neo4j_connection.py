"""
Neo4j Connection Manager.

Manages the Neo4j driver lifecycle: initialization, session creation,
health verification, and graceful shutdown.

INTERVIEW: "Why a connection manager?"
- Neo4j's driver internally manages a connection pool (default ~100 connections).
- Creating a new driver per request wastes TCP sockets and authentication round-trips.
- We initialize ONCE at startup and reuse across all requests (Singleton pattern).

INTERVIEW: "What is connection pooling?"
- Instead of opening/closing a TCP connection per query, the driver maintains
  a pool of idle connections. When a query needs a connection, it borrows one
  from the pool, executes, and returns it. This reduces latency from ~50ms
  (TCP handshake + auth) to ~1ms (borrow from pool).
"""

from neo4j import GraphDatabase, Driver, Session
from app.config import settings


class Neo4jConnection:
    """
    Thread-safe Neo4j connection manager.

    Usage:
        neo4j_conn = Neo4jConnection()
        neo4j_conn.connect()          # Call once at app startup
        session = neo4j_conn.get_session()
        neo4j_conn.close()            # Call once at app shutdown
    """

    def __init__(self):
        self._driver: Driver | None = None

    def connect(self) -> None:
        """
        Initialize the Neo4j driver with connection pooling.
        Called once during FastAPI startup event.

        INTERVIEW: Why not connect in __init__?
        - We want to control WHEN the connection happens (startup event).
        - If Neo4j is down, __init__ would crash the import, making debugging harder.
        - Separating construction from initialization follows the "two-phase init" pattern.
        """
        if self._driver is not None:
            return  # Already connected, idempotent

        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        # Verify connectivity immediately — fail fast if Neo4j is unreachable
        self._driver.verify_connectivity()
        print(f"[NEO4J] Connected to {settings.neo4j_uri}")

    def close(self) -> None:
        """
        Gracefully close all pooled connections.
        Called during FastAPI shutdown event.
        """
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            print("[NEO4J] Connection closed")

    def get_session(self) -> Session:
        """
        Get a new session from the connection pool.

        INTERVIEW: "Session vs Driver?"
        - Driver = long-lived, manages the connection pool (one per app).
        - Session = short-lived, represents a logical unit of work (one per request/operation).
        - Sessions are lightweight — creating one just borrows a connection from the pool.
        """
        if self._driver is None:
            raise RuntimeError(
                "Neo4j driver not initialized. Call connect() first. "
                "This usually means the app startup event didn't run."
            )
        return self._driver.session()

    def is_connected(self) -> bool:
        """Check if Neo4j is reachable. Used for health checks."""
        if self._driver is None:
            return False
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False


# Singleton instance — imported and used across the application
# INTERVIEW: "Why a module-level singleton?"
# - Python modules are imported once and cached. This guarantees
#   exactly one Neo4jConnection instance exists across all imports.
# - Alternative: FastAPI's Depends() with a dependency provider function.
neo4j_connection = Neo4jConnection()
