# rag-core/rag_core/capabilities/structured_query.py
import sqlite3
import re
from typing import Protocol, runtime_checkable
from rag_core.types import StructuredQuery, StructuredResult


@runtime_checkable
class StructuredQueryEngine(Protocol):
    async def execute_readonly(self, query: StructuredQuery) -> StructuredResult:
        """Execute a read-only SQL query. Must reject any non-SELECT statements."""
        ...


class SQLiteQueryEngine:
    """Read-only SQLite query engine. Only allows SELECT statements."""

    def __init__(self, db_path: str = ":memory:"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL to set up tables (call once at init, not via execute_readonly)."""
        self._conn.executescript(ddl)
        self._conn.commit()

    def _is_readonly(self, sql: str) -> bool:
        cleaned = sql.strip().lstrip("(")
        first_word = re.split(r"\s+", cleaned, maxsplit=1)[0].upper()
        return first_word in ("SELECT", "WITH", "EXPLAIN", "PRAGMA")

    async def execute_readonly(self, query: StructuredQuery) -> StructuredResult:
        if not self._is_readonly(query.sql):
            raise ValueError(f"Only SELECT queries are allowed. Rejected: {query.sql[:80]}")

        cur = self._conn.execute(query.sql, query.params or {})
        rows = [list(row) for row in cur.fetchall()]
        columns = [desc[0] for desc in cur.description] if cur.description else []
        return StructuredResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
        )
