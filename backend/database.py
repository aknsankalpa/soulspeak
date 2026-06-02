"""
database.py — SoulSpeak PostgreSQL Database Layer (Supabase)
============================================================
Migrated from SQLite to PostgreSQL. Connection string is read
from SUPABASE_DB_URL environment variable.

Tables:
  users        — user profiles (email, name, profession, goals, settings)
  sessions     — voice analysis sessions (scores, transcript, insights, audio_url)
  chat_logs    — chat message history per user / session
"""

import psycopg2
import psycopg2.extras
import json
import uuid
import threading
import logging
import os
from datetime import datetime

log = logging.getLogger("soulspeak.db")

SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL", "")

_local = threading.local()


def _get_conn():
    if not hasattr(_local, "conn") or _local.conn is None or _local.conn.closed:
        if not SUPABASE_DB_URL:
            raise RuntimeError("SUPABASE_DB_URL environment variable is not set")
        _local.conn = psycopg2.connect(SUPABASE_DB_URL)
    # Reconnect if connection dropped
    try:
        _local.conn.isolation_level
    except psycopg2.InterfaceError:
        _local.conn = psycopg2.connect(SUPABASE_DB_URL)
    return _local.conn


def _q(sql, params=(), fetchone=False, fetchall=False, commit=False):
    for attempt in range(2):
        try:
            conn = _get_conn()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(sql, params)
            if commit:
                conn.commit()
            if fetchone:
                row = cur.fetchone()
                return dict(row) if row else None
            if fetchall:
                return [dict(r) for r in cur.fetchall()]
            return cur
        except psycopg2.OperationalError:
            if attempt == 0:
                # Stale connection (e.g. post-fork SSL error) — drop and retry
                _local.conn = None
                continue
            raise


# ── Schema ─────────────────────────────────────────────────────────────────────

_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS users (
        id          TEXT PRIMARY KEY,
        email       TEXT UNIQUE NOT NULL,
        name        TEXT NOT NULL DEFAULT '',
        password    TEXT NOT NULL DEFAULT '',
        profession  TEXT NOT NULL DEFAULT '',
        goals       TEXT NOT NULL DEFAULT '[]',
        settings    TEXT NOT NULL DEFAULT '{}',
        created_at  TEXT NOT NULL,
        token       TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS sessions (
        id              TEXT PRIMARY KEY,
        user_id         TEXT,
        timestamp       TEXT NOT NULL,
        date_str        TEXT NOT NULL,
        time_str        TEXT NOT NULL,
        duration        INTEGER NOT NULL DEFAULT 0,
        duration_fmt    TEXT NOT NULL DEFAULT '00:00',
        filename        TEXT NOT NULL DEFAULT '',
        profession      TEXT NOT NULL DEFAULT '',
        prompt_type     TEXT NOT NULL DEFAULT 'passage',
        scores          TEXT NOT NULL DEFAULT '{}',
        labels          TEXT NOT NULL DEFAULT '{}',
        overall         INTEGER NOT NULL DEFAULT 0,
        transcript      TEXT NOT NULL DEFAULT '',
        insights        TEXT NOT NULL DEFAULT '[]',
        source          TEXT NOT NULL DEFAULT '',
        models_used     TEXT NOT NULL DEFAULT '{}',
        audio_url            TEXT NOT NULL DEFAULT '',
        profession_benchmarks TEXT NOT NULL DEFAULT '{}',
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
    )""",
    """CREATE TABLE IF NOT EXISTS chat_logs (
        id              TEXT PRIMARY KEY,
        user_id         TEXT,
        session_id      TEXT,
        conversation_id TEXT,
        role            TEXT NOT NULL,
        content         TEXT NOT NULL,
        model_name      TEXT NOT NULL DEFAULT '',
        timestamp       TEXT NOT NULL,
        FOREIGN KEY (user_id)    REFERENCES users(id)    ON DELETE CASCADE,
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_sessions_user    ON sessions  (user_id, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_chatlogs_user    ON chat_logs (user_id, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_chatlogs_session ON chat_logs (session_id)",
    "CREATE INDEX IF NOT EXISTS idx_chatlogs_conv    ON chat_logs (conversation_id)",
]

_MIGRATIONS = [
    "ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS conversation_id TEXT",
    "ALTER TABLE users     ADD COLUMN IF NOT EXISTS token TEXT",
    "ALTER TABLE sessions  ADD COLUMN IF NOT EXISTS audio_url TEXT DEFAULT ''",
    "ALTER TABLE sessions  ADD COLUMN IF NOT EXISTS profession_benchmarks TEXT DEFAULT '{}'",
]


def init():
    """Create tables and run migrations. Call once at app startup."""
    conn = _get_conn()
    cur = conn.cursor()
    for stmt in _SCHEMA:
        cur.execute(stmt)
    conn.commit()
    for migration in _MIGRATIONS:
        try:
            cur.execute(migration)
            conn.commit()
        except Exception:
            conn.rollback()
    log.info("PostgreSQL database ready (Supabase)")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _j(obj):
    return json.dumps(obj, ensure_ascii=False)

def _pj(s):
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}

def _now():
    return datetime.now().isoformat()

def _new_id():
    return str(uuid.uuid4())

def _hydrate_session(row):
    if not row:
        return row
    row["scores"]               = _pj(row.get("scores",               "{}"))
    row["labels"]               = _pj(row.get("labels",               "{}"))
    row["insights"]             = _pj(row.get("insights",             "[]"))
    row["models_used"]          = _pj(row.get("models_used",          "{}"))
    row["profession_benchmarks"] = _pj(row.get("profession_benchmarks", "{}"))
    row["date"]  = row.pop("date_str",  row.get("date",  ""))
    row["time"]  = row.pop("time_str",  row.get("time",  ""))
    return row

def _hydrate_user(row):
    if not row:
        return row
    row["goals"]    = _pj(row.get("goals",    "[]"))
    row["settings"] = _pj(row.get("settings", "{}"))
    row.pop("password", None)
    return row


# ── Token persistence ──────────────────────────────────────────────────────────

def store_token(token, user_id):
    _q("UPDATE users SET token=%s WHERE id=%s", (token, user_id), commit=True)

def get_user_id_for_token(token):
    row = _q("SELECT id FROM users WHERE token=%s", (token,), fetchone=True)
    return row["id"] if row else None

def clear_user_token(user_id):
    _q("UPDATE users SET token=NULL WHERE id=%s", (user_id,), commit=True)


# ── Users ──────────────────────────────────────────────────────────────────────

def create_user(email, name, password, profession="", goals=None, settings=None):
    if get_user_by_email(email):
        raise ValueError("Email already registered")
    uid = _new_id()
    _q("""INSERT INTO users
          (id,email,name,password,profession,goals,settings,created_at)
          VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
       (uid, email.lower().strip(), name.strip(), password,
        profession, _j(goals or []), _j(settings or {}), _now()),
       commit=True)
    log.info("New user: %s (%s)", name, email)
    return get_user_by_id(uid)


def get_user_by_email(email):
    row = _q("SELECT * FROM users WHERE email=%s",
             (email.lower().strip(),), fetchone=True)
    return _hydrate_user(row) if row else None


def get_user_by_id(uid):
    row = _q("SELECT * FROM users WHERE id=%s", (uid,), fetchone=True)
    return _hydrate_user(row) if row else None


def verify_user(email, password):
    row = _q("SELECT * FROM users WHERE email=%s",
             (email.lower().strip(),), fetchone=True)
    if row and row["password"] == password:
        return _hydrate_user(row)
    return None


def verify_password(uid, password):
    row = _q("SELECT password FROM users WHERE id=%s", (uid,), fetchone=True)
    if not row:
        return False
    return row.get("password", "") == password


def delete_user(uid):
    _q("DELETE FROM chat_logs WHERE user_id=%s", (uid,), commit=True)
    _q("DELETE FROM sessions WHERE user_id=%s",  (uid,), commit=True)
    _q("DELETE FROM users WHERE id=%s",           (uid,), commit=True)
    log.info("Deleted user %s and all associated data", uid[:8])


def update_user(uid, **fields):
    allowed = {"name", "profession", "goals", "settings"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return get_user_by_id(uid)
    set_parts, vals = [], []
    for k, v in updates.items():
        set_parts.append(f"{k}=%s")
        vals.append(_j(v) if isinstance(v, (dict, list)) else v)
    vals.append(uid)
    _q(f"UPDATE users SET {','.join(set_parts)} WHERE id=%s", vals, commit=True)
    return get_user_by_id(uid)


# ── Sessions ───────────────────────────────────────────────────────────────────

def save_session(session, user_id=None):
    sid = session.get("id") or _new_id()
    now = datetime.now()
    dur = int(session.get("duration", 0))
    _q("""INSERT INTO sessions
          (id,user_id,timestamp,date_str,time_str,duration,duration_fmt,
           filename,profession,prompt_type,
           scores,labels,overall,transcript,insights,source,models_used,audio_url,
           profession_benchmarks)
          VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
          ON CONFLICT (id) DO UPDATE SET
            user_id               = EXCLUDED.user_id,
            timestamp             = EXCLUDED.timestamp,
            date_str              = EXCLUDED.date_str,
            time_str              = EXCLUDED.time_str,
            duration              = EXCLUDED.duration,
            duration_fmt          = EXCLUDED.duration_fmt,
            filename              = EXCLUDED.filename,
            profession            = EXCLUDED.profession,
            prompt_type           = EXCLUDED.prompt_type,
            scores                = EXCLUDED.scores,
            labels                = EXCLUDED.labels,
            overall               = EXCLUDED.overall,
            transcript            = EXCLUDED.transcript,
            insights              = EXCLUDED.insights,
            source                = EXCLUDED.source,
            models_used           = EXCLUDED.models_used,
            audio_url             = EXCLUDED.audio_url,
            profession_benchmarks = EXCLUDED.profession_benchmarks""",
       (sid, user_id,
        session.get("timestamp", now.isoformat()),
        session.get("date",  now.strftime("%B %d, %Y")),
        session.get("time",  now.strftime("%H:%M")),
        dur,
        session.get("duration_fmt", f"{dur//60:02d}:{dur%60:02d}"),
        session.get("filename",    ""),
        session.get("profession",  ""),
        session.get("prompt_type", "passage"),
        _j(session.get("scores",               {})),
        _j(session.get("labels",               {})),
        int(session.get("overall",              0)),
        session.get("transcript",  ""),
        _j(session.get("insights",             [])),
        session.get("source",      ""),
        _j(session.get("models_used",          {})),
        session.get("audio_url",   ""),
        _j(session.get("profession_benchmarks", {})),
       ), commit=True)
    log.info("Session saved: %s (user=%s)", sid[:8], user_id or "anon")
    return get_session(sid)


def get_session(sid):
    row = _q("SELECT * FROM sessions WHERE id=%s", (sid,), fetchone=True)
    return _hydrate_session(row) if row else None


def list_sessions(user_id=None, limit=50):
    if user_id:
        rows = _q("SELECT * FROM sessions WHERE user_id=%s ORDER BY timestamp DESC LIMIT %s",
                  (user_id, limit), fetchall=True)
    else:
        rows = _q("SELECT * FROM sessions ORDER BY timestamp DESC LIMIT %s",
                  (limit,), fetchall=True)
    return [_hydrate_session(r) for r in rows]


def delete_session(sid):
    cur = _q("DELETE FROM sessions WHERE id=%s", (sid,), commit=True)
    return cur.rowcount > 0


# ── Chat logs ──────────────────────────────────────────────────────────────────

def save_chat_message(role, content, user_id=None, session_id=None,
                      conversation_id=None, model_name=""):
    mid = _new_id()
    _q("""INSERT INTO chat_logs
          (id,user_id,session_id,conversation_id,role,content,model_name,timestamp)
          VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
       (mid, user_id, session_id, conversation_id, role, content, model_name, _now()),
       commit=True)
    return {"id": mid, "role": role, "content": content,
            "model_name": model_name, "timestamp": _now()}


def get_chat_history(user_id=None, session_id=None, conversation_id=None, limit=100):
    if conversation_id:
        rows = _q("SELECT * FROM chat_logs WHERE conversation_id=%s ORDER BY timestamp ASC LIMIT %s",
                  (conversation_id, limit), fetchall=True)
    elif session_id:
        rows = _q("SELECT * FROM chat_logs WHERE session_id=%s ORDER BY timestamp ASC LIMIT %s",
                  (session_id, limit), fetchall=True)
    elif user_id:
        rows = _q("SELECT * FROM chat_logs WHERE user_id=%s ORDER BY timestamp ASC LIMIT %s",
                  (user_id, limit), fetchall=True)
    else:
        rows = _q("SELECT * FROM chat_logs ORDER BY timestamp DESC LIMIT %s",
                  (limit,), fetchall=True)
        rows = list(reversed(rows))
    return rows


def delete_chat_history(user_id=None, session_id=None, conversation_id=None):
    if conversation_id:
        cur = _q("DELETE FROM chat_logs WHERE conversation_id=%s", (conversation_id,), commit=True)
    elif session_id:
        cur = _q("DELETE FROM chat_logs WHERE session_id=%s", (session_id,), commit=True)
    elif user_id:
        cur = _q("DELETE FROM chat_logs WHERE user_id=%s", (user_id,), commit=True)
    else:
        return 0
    return cur.rowcount


def list_chat_sessions(user_id, limit=50):
    rows = _q("""
        SELECT conversation_id as session_id,
               MIN(timestamp) as started_at,
               MAX(timestamp) as last_msg_at,
               COUNT(*) as msg_count
        FROM chat_logs
        WHERE user_id=%s AND conversation_id IS NOT NULL
        GROUP BY conversation_id
        ORDER BY last_msg_at DESC
        LIMIT %s
    """, (user_id, limit), fetchall=True)
    result = []
    for row in rows:
        first = _q("""
            SELECT content FROM chat_logs
            WHERE user_id=%s AND conversation_id=%s AND role='user'
            ORDER BY timestamp ASC LIMIT 1
        """, (user_id, row["session_id"]), fetchone=True)
        content = first["content"] if first else ""
        row["preview"] = (content[:75] + "…") if len(content) > 75 else (content or "New conversation")
        result.append(row)
    return result


def get_stats():
    return {
        "users":         _q("SELECT COUNT(*) as n FROM users",     fetchone=True)["n"],
        "sessions":      _q("SELECT COUNT(*) as n FROM sessions",  fetchone=True)["n"],
        "chat_messages": _q("SELECT COUNT(*) as n FROM chat_logs", fetchone=True)["n"],
        "db_type":       "PostgreSQL (Supabase)",
    }
