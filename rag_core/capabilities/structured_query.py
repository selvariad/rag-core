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


class SqlValidationError(ValueError):
    """SQL failed validation — should NOT fall back to RAG."""
    pass


class SqlExecutionError(ValueError):
    """SQL execution failed — may fall back to RAG."""
    pass


class SQLiteQueryEngine:
    """Read-only SQLite query engine with table/column allowlists."""

    def __init__(
        self,
        db_path: str = ":memory:",
        table_allowlist: list[str] | None = None,
        column_allowlist: dict[str, list[str]] | None = None,
        default_limit: int = 100,
    ):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # None = no allowlist (all tables allowed)
        # [] = deny all tables
        # ["x"] = only allow x
        self._table_allowlist = set(table_allowlist) if table_allowlist is not None else None
        self._column_allowlist = column_allowlist or {}
        self._default_limit = default_limit

    def get_tables(self) -> list[str]:
        cur = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [row[0] for row in cur.fetchall()]

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL to set up tables (call once at init, not via execute_readonly)."""
        self._conn.executescript(ddl)
        self._conn.commit()

    def _is_select_only(self, sql: str) -> bool:
        cleaned = sql.strip().lstrip("(")
        first_word = re.split(r"\s+", cleaned, maxsplit=1)[0].upper()
        return first_word in ("SELECT", "WITH")

    def _validate_tables(self, sql: str) -> str | None:
        """Check referenced tables against allowlist. Returns error string or None.
        None = no allowlist (all allowed), [] = deny all, ["x"] = whitelist.
        """
        if self._table_allowlist is None:
            return None
        tables = set(re.findall(
            r'\b(?:FROM|JOIN)\s+(\w+)',
            sql, re.IGNORECASE,
        ))
        if not tables:
            return None  # No tables referenced (e.g. SELECT 1)
        invalid = tables - self._table_allowlist
        if invalid:
            return f"Table(s) not in allowlist: {', '.join(sorted(invalid))}"
        return None

    def _add_default_limit(self, sql: str) -> tuple[str, bool]:
        """Returns (sql, limit_applied)."""
        if re.search(r'\bLIMIT\b', sql, re.IGNORECASE):
            return sql, False
        return f"{sql.rstrip(';')} LIMIT {self._default_limit}", True

    async def execute_readonly(self, query: StructuredQuery) -> StructuredResult:
        sql = query.sql.strip()

        if not sql:
            raise SqlValidationError("Empty SQL query")

        if not self._is_select_only(sql):
            raise SqlValidationError(f"Only SELECT queries are allowed. Rejected: {sql[:80]}")

        # Table allowlist check
        err = self._validate_tables(sql)
        if err:
            raise SqlValidationError(err)

        sql, limit_applied = self._add_default_limit(sql)

        cur = self._conn.execute(sql, query.params or {})
        rows = [list(row) for row in cur.fetchall()]
        columns = [desc[0] for desc in cur.description] if cur.description else []
        return StructuredResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            limit_applied=limit_applied,
        )
