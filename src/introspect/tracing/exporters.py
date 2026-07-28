"""
exporters.py — Custom OpenTelemetry span exporters.

SQLiteSpanExporter: Writes spans directly to the SQLite metrics store
for dashboard consumption.

ConsoleSpanExporter: Pretty-prints span data for CLI debugging with
color-coded output and tree-style nesting.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class SQLiteSpanExporter(SpanExporter):
    """Exports OpenTelemetry spans to a SQLite database.

    Creates and writes to a `spans` table with full attribute
    serialization, enabling time-range queries from the API server
    and dashboard.

    Thread-safe: uses a dedicated connection with WAL journal mode.
    """

    def __init__(self, db_path: str) -> None:
        """Initialize with a path to the SQLite database.

        Args:
            db_path: Filesystem path for the SQLite database file.
        """
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_table()

    def _create_table(self) -> None:
        """Create the spans table if it doesn't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                span_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                parent_span_id TEXT,
                name TEXT NOT NULL,
                start_time_ns INTEGER NOT NULL,
                end_time_ns INTEGER NOT NULL,
                duration_ms REAL NOT NULL,
                status_code TEXT NOT NULL,
                status_description TEXT,
                attributes_json TEXT,
                events_json TEXT,
                created_at REAL NOT NULL DEFAULT (unixepoch('now'))
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON spans(trace_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_spans_name ON spans(name)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_spans_start_time ON spans(start_time_ns)
        """)
        self._conn.commit()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export a batch of spans to SQLite.

        Args:
            spans: Sequence of completed OpenTelemetry spans.

        Returns:
            SpanExportResult.SUCCESS or SpanExportResult.FAILURE.
        """
        try:
            rows = []
            for span in spans:
                ctx = span.get_span_context()
                parent_ctx = span.parent

                start_ns = span.start_time or 0
                end_ns = span.end_time or 0
                duration_ms = (end_ns - start_ns) / 1_000_000

                # Serialize attributes to JSON.
                attrs = dict(span.attributes) if span.attributes else {}
                attrs_json = json.dumps(attrs, default=str)

                # Serialize events.
                events = []
                if span.events:
                    for event in span.events:
                        events.append({
                            "name": event.name,
                            "timestamp_ns": event.timestamp,
                            "attributes": dict(event.attributes) if event.attributes else {},
                        })
                events_json = json.dumps(events, default=str)

                status_code = span.status.status_code.name if span.status else "UNSET"
                status_desc = span.status.description if span.status else None

                rows.append((
                    format(ctx.span_id, "016x"),
                    format(ctx.trace_id, "032x"),
                    format(parent_ctx.span_id, "016x") if parent_ctx else None,
                    span.name,
                    start_ns,
                    end_ns,
                    round(duration_ms, 4),
                    status_code,
                    status_desc,
                    attrs_json,
                    events_json,
                ))

            self._conn.executemany(
                """INSERT OR REPLACE INTO spans
                   (span_id, trace_id, parent_span_id, name, start_time_ns,
                    end_time_ns, duration_ms, status_code, status_description,
                    attributes_json, events_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            self._conn.commit()
            return SpanExportResult.SUCCESS

        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Flush pending writes."""
        try:
            self._conn.commit()
            return True
        except Exception:
            return False


class PrettyConsoleSpanExporter(SpanExporter):
    """Pretty-prints spans to the console for CLI debugging.

    Outputs color-coded, human-readable span summaries with key
    attributes highlighted.
    """

    # ANSI color codes.
    _RESET = "\033[0m"
    _BOLD = "\033[1m"
    _DIM = "\033[2m"
    _CYAN = "\033[36m"
    _GREEN = "\033[32m"
    _YELLOW = "\033[33m"
    _RED = "\033[31m"
    _MAGENTA = "\033[35m"

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Print each span with formatted attributes."""
        for span in spans:
            self._print_span(span)
        return SpanExportResult.SUCCESS

    def _print_span(self, span: ReadableSpan) -> None:
        """Format and print a single span."""
        ctx = span.get_span_context()
        start_ns = span.start_time or 0
        end_ns = span.end_time or 0
        duration_ms = (end_ns - start_ns) / 1_000_000

        status = span.status.status_code.name if span.status else "UNSET"
        status_color = self._GREEN if status == "OK" else self._RED

        # Header line.
        print(
            f"{self._CYAN}{self._BOLD}▸ {span.name}{self._RESET}  "
            f"{self._DIM}[{format(ctx.span_id, '016x')[:8]}]{self._RESET}  "
            f"{status_color}{status}{self._RESET}  "
            f"{self._YELLOW}{duration_ms:.2f}ms{self._RESET}"
        )

        # Key attributes.
        if span.attributes:
            for key, value in span.attributes.items():
                print(
                    f"  {self._DIM}├─{self._RESET} "
                    f"{self._MAGENTA}{key}{self._RESET}: {value}"
                )

        print()

    def shutdown(self) -> None:
        """No cleanup needed for console output."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Flush stdout."""
        sys.stdout.flush()
        return True
