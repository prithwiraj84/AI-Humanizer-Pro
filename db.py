"""
Database access for SQLite (local development) and PostgreSQL (production).

Set DATABASE_URL (postgres://... or postgresql://...) to use PostgreSQL, for example a free
Neon database. Without it the app uses a local SQLite file, which is fine for development but
is wiped on every deploy and restart on hosts with ephemeral disks such as Render's free tier.

Queries are written once with "?" placeholders and portable SQL; this module adapts them.
"""
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IntegrityError = (sqlite3.IntegrityError,)

_lock = threading.Lock()
_backend = None   # 'sqlite' or 'postgres', decided on first use (after .env has been loaded)
_pool = None
_sqlite_path = None


class Row(dict):
    """A result row readable by column name or position, like sqlite3.Row."""
    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


def _postgres_row_factory(cursor):
    # description is None for statements without results (e.g. the pool's health check)
    names = [column.name for column in cursor.description or ()]
    return lambda values: Row(zip(names, values))


def _setup():
    global _backend, _pool, _sqlite_path, IntegrityError
    with _lock:
        if _backend:
            return
        url = os.environ.get('DATABASE_URL', '')
        if url.startswith(('postgres://', 'postgresql://')):
            import psycopg
            from psycopg_pool import ConnectionPool
            _pool = ConnectionPool(
                url, min_size=1, max_size=int(os.environ.get('DB_POOL_SIZE', 5)), timeout=15,
                # GMT (= UTC) is built into every PostgreSQL, even builds without the timezone database
                kwargs={'row_factory': _postgres_row_factory, 'options': '-c timezone=GMT'},
                check=ConnectionPool.check_connection, open=True,
            )
            IntegrityError = (psycopg.IntegrityError,)
            _backend = 'postgres'
            logger.info('Database: PostgreSQL')
        else:
            _sqlite_path = os.environ.get('DATABASE_PATH') or os.path.join(BASE_DIR, 'humanizer.db')
            _backend = 'sqlite'
            logger.info('Database: SQLite (%s)', _sqlite_path)


def is_postgres():
    _setup()
    return _backend == 'postgres'


def _adapt_param(value):
    # SQLite has no datetime type: store a sortable ISO string (Python 3.12 deprecates the implicit adapter)
    if _backend == 'sqlite' and isinstance(value, datetime):
        return value.isoformat(sep=' ', timespec='microseconds')
    return value


class Connection:
    """One request's connection, with a sqlite3-style execute() on both backends."""

    def __init__(self):
        _setup()
        if _backend == 'postgres':
            self._raw = _pool.getconn()
        else:
            self._raw = sqlite3.connect(_sqlite_path, timeout=10)
            self._raw.row_factory = sqlite3.Row

    def execute(self, sql, params=()):
        params = tuple(_adapt_param(p) for p in params)
        if _backend == 'postgres':
            sql = sql.replace('%', '%%').replace('?', '%s')
        return self._raw.execute(sql, params)

    def commit(self):
        self._raw.commit()

    def rollback(self):
        self._raw.rollback()

    def close(self):
        if _backend == 'postgres':
            import psycopg
            if self._raw.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
                self._raw.rollback()
            _pool.putconn(self._raw)
        else:
            self._raw.close()


def connect():
    return Connection()


def utcnow():
    """Naive UTC, matching how timestamps are stored."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def ensure_column(conn, table, column, definition):
    """Add a column to an existing table if it's missing (migrations for databases made by older versions)."""
    if _backend == 'postgres':
        conn.execute(f'ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {definition}')
        return
    existing = [row[1] for row in conn.execute(f'PRAGMA table_info({table})')]
    if column not in existing:
        conn.execute(f'ALTER TABLE {table} ADD COLUMN {column} {definition}')


def ddl(sql):
    """Adapt a CREATE TABLE statement written for SQLite to the active backend."""
    _setup()
    if _backend == 'postgres':
        return sql.replace('INTEGER PRIMARY KEY AUTOINCREMENT', 'SERIAL PRIMARY KEY')
    return sql


def to_datetime(value):
    """Timestamps come back as datetime from PostgreSQL and as ISO strings from SQLite."""
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def timestamp_str(value):
    """UTC timestamp as 'YYYY-MM-DD HH:MM:SS', the format the dashboard expects."""
    value = to_datetime(value)
    return value.strftime('%Y-%m-%d %H:%M:%S') if value else None
