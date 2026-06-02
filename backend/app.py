"""
app.py — SoulSpeak Flask Backend v3
=====================================
All user data (profiles, sessions, chat history) is stored in SQLite
via database.py. Drop-in replaceable with PostgreSQL later.

Endpoints:
  GET  /                        Serve the SPA
  POST /api/auth/register       Register new user
  POST /api/auth/login          Login
  GET  /api/auth/profile        Get current user profile
  PATCH /api/auth/profile       Update profile (name, profession, goals)
  POST /api/analyze             Analyse audio → Big Five scores
  GET  /api/sessions            List sessions for logged-in user
  GET  /api/sessions/<id>       Get a single session
  DELETE /api/sessions/<id>     Delete session
  POST /api/chat                VOXMIND AI via Ollama (SSE stream)
  GET  /api/chat/history        Fetch chat history for current user
  DELETE /api/chat/history      Clear chat history
  GET  /api/ollama/models       List Ollama models
  GET  /api/status              Health check
"""

import os
import sys
import uuid
import json
import logging
import tempfile
from datetime import datetime

import requests
from flask import (Flask, request, jsonify, send_from_directory,
                   Response, stream_with_context)
from flask_cors import CORS

import database as db

# ── Supabase Storage ───────────────────────────────────────────────────────────
SUPABASE_URL         = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
AUDIO_BUCKET         = "voice-clips"

_supabase_client = None

def _get_supabase():
    global _supabase_client
    if _supabase_client is None and SUPABASE_URL and SUPABASE_SERVICE_KEY:
        from supabase import create_client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase_client

def _upload_audio(tmp_path, session_id, filename):
    """Upload audio to Supabase Storage. Returns public URL or empty string."""
    client = _get_supabase()
    if not client:
        return ""
    try:
        ext = os.path.splitext(filename)[-1].lower() or ".webm"
        mime_map = {".webm": "audio/webm", ".wav": "audio/wav",
                    ".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".m4a": "audio/mp4"}
        mime = mime_map.get(ext, "audio/webm")
        storage_path = f"{session_id}{ext}"
        with open(tmp_path, "rb") as f:
            data = f.read()
        client.storage.from_(AUDIO_BUCKET).upload(
            path=storage_path,
            file=data,
            file_options={"content-type": mime, "upsert": "true"},
        )
        return client.storage.from_(AUDIO_BUCKET).get_public_url(storage_path)
    except Exception as e:
        log.warning("Audio upload to Supabase Storage failed: %s", e)
        return ""

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("soulspeak")

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# ── Config ─────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _find_model(filename: str) -> str:
    """Locate a model file by checking common paths relative to this file."""
    candidates = [
        os.environ.get("WAV2VEC2_PATH" if "wav2vec2" in filename else "BERT_PATH", ""),
        filename,
        os.path.join(_BASE_DIR, filename),
        os.path.join(_BASE_DIR, "ml_models", filename),
        os.path.join(os.path.dirname(_BASE_DIR), "ml_models", filename),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return os.path.join(_BASE_DIR, "ml_models", filename)  # best-guess path for error messages

WAV2VEC2_PATH   = _find_model("wav2vec2_personality_best.pt")
BERT_PATH       = _find_model("bert_personality_best.pt")
WHISPER_SIZE    = os.environ.get("WHISPER_SIZE",   "base")
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY",   "")
GROQ_MODEL      = os.environ.get("GROQ_MODEL",     "llama-3.1-8b-instant")
ACOUSTIC_WEIGHT = float(os.environ.get("ACOUSTIC_WEIGHT", "0.5"))

MODEL_STATUS = {"wav2vec2": False, "bert": False, "whisper": False, "device": "cpu"}

# ── Model loader ───────────────────────────────────────────────────────────────

def _load_ml_models():
    global MODEL_STATUS
    try:
        from analyze import load_all_models
        MODEL_STATUS = load_all_models(WAV2VEC2_PATH, BERT_PATH, WHISPER_SIZE)
    except ImportError as e:
        log.warning("ML deps not installed (analysis will simulate): %s", e)
    except Exception as e:
        log.error("Model load error: %s", e)

# Load models at import time so Gunicorn workers have them ready
_load_ml_models()


# ── Simple session token store (in-memory — swap for JWT in production) ────────
# Maps token → user_id
_token_store = {}   # in-memory cache; DB is the source of truth

def _make_token(user_id):
    token = str(uuid.uuid4())
    _token_store[token] = user_id
    db.store_token(token, user_id)   # persist so restarts don't invalidate sessions
    return token

def _get_user_id():
    """Extract user_id from Bearer token — checks memory cache then DB."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth[7:].strip()
    uid = _token_store.get(token)
    if uid:
        return uid
    # Cache miss (e.g. after server restart) — look up in DB
    uid = db.get_user_id_for_token(token)
    if uid:
        _token_store[token] = uid   # repopulate cache
    return uid


# ── Groq helpers ───────────────────────────────────────────────────────────────

_GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

def _groq_available():
    return bool(GROQ_API_KEY)

def _get_groq_client():
    from groq import Groq
    return Groq(api_key=GROQ_API_KEY)


# ── JSON error handlers (so /api/* never returns HTML on 404/500) ──────────────

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not found", "path": request.path}), 404
    return e

@app.errorhandler(405)
def method_not_allowed(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Method not allowed"}), 405
    return e

@app.errorhandler(500)
def internal_error(e):
    if request.path.startswith("/api/"):
        log.exception("Internal error on %s", request.path)
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500
    return e


# ── Routes: Frontend ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    html_path = os.path.join(app.template_folder, "index.html")
    if not os.path.exists(html_path):
        return "Run <b>python setup.py</b> first to generate templates/index.html", 500
    with open(html_path, encoding="utf-8") as f:
        html = f.read()
    return Response(html, mimetype="text/html")

@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route("/style.css")
def serve_css():
    return send_from_directory(os.path.dirname(app.template_folder), "style.css")


# ── Routes: Auth ───────────────────────────────────────────────────────────────

@app.route("/api/auth/register", methods=["POST"])
def register():
    data       = request.get_json(force=True) or {}
    email      = data.get("email", "").lower().strip()
    name       = data.get("name", "").strip()
    password   = data.get("password", "")
    profession = data.get("profession", "")
    goals      = data.get("goals", [])

    if not email:
        return jsonify({"error": "Email is required"}), 400
    if not password:
        return jsonify({"error": "Password is required"}), 400

    import re as _re
    pw_issues = []
    if len(password) < 8:
        pw_issues.append("at least 8 characters")
    if not _re.search(r'[A-Z]', password):
        pw_issues.append("one uppercase letter (A–Z)")
    if not _re.search(r'[a-z]', password):
        pw_issues.append("one lowercase letter (a–z)")
    if not _re.search(r'\d', password):
        pw_issues.append("one number (0–9)")
    if pw_issues:
        return jsonify({"error": "Password must contain: " + ", ".join(pw_issues)}), 400

    try:
        user = db.create_user(email, name, password, profession, goals)
    except ValueError as e:
        return jsonify({"error": str(e)}), 409

    token = _make_token(user["id"])
    return jsonify({"success": True, "user": user, "token": token})


@app.route("/api/auth/login", methods=["POST"])
def login():
    data     = request.get_json(force=True) or {}
    email    = data.get("email", "").lower().strip()
    password = data.get("password", "")

    if not email:
        return jsonify({"error": "Email is required"}), 400

    # Try real auth first
    user = db.verify_user(email, password)

    # If no match but no password given — create a demo guest session
    if not user and not password:
        existing = db.get_user_by_email(email)
        if existing:
            user = existing
        else:
            # Auto-create guest user
            user = db.create_user(
                email=email,
                name=email.split("@")[0].replace(".", " ").title(),
                password="",
                profession="Healthcare Professional",
            )

    if not user:
        return jsonify({"error": "Invalid email or password"}), 401

    token = _make_token(user["id"])
    return jsonify({"success": True, "user": user, "token": token})


@app.route("/api/auth/profile", methods=["GET"])
def get_profile():
    uid = _get_user_id()
    if not uid:
        return jsonify({"error": "Not authenticated"}), 401
    user = db.get_user_by_id(uid)
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user)


@app.route("/api/auth/profile", methods=["PATCH"])
def update_profile():
    uid = _get_user_id()
    if not uid:
        return jsonify({"error": "Not authenticated"}), 401
    data = request.get_json(force=True) or {}
    allowed = {k: v for k, v in data.items() if k in ("name", "profession", "goals", "settings")}
    user = db.update_user(uid, **allowed)
    return jsonify({"success": True, "user": user})


@app.route("/api/auth/profile", methods=["DELETE"])
def delete_account():
    """Delete the authenticated user's account and ALL their data after password verification."""
    uid = _get_user_id()
    if not uid:
        return jsonify({"error": "Not authenticated"}), 401
    data     = request.get_json(force=True) or {}
    password = data.get("password", "")
    if not password:
        return jsonify({"error": "Password is required"}), 400
    if not db.verify_password(uid, password):
        return jsonify({"error": "Incorrect password — please try again"}), 401
    db.delete_user(uid)
    # Revoke tokens from cache and DB
    for token in [t for t, u in _token_store.items() if u == uid]:
        _token_store.pop(token, None)
    # token column is wiped by delete_user's DELETE FROM users
    log.info("Account deleted: user_id=%s", uid[:8])
    return jsonify({"success": True})


# ── Routes: Analysis ───────────────────────────────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def analyze():
    user_id     = _get_user_id()
    duration    = float(request.form.get("duration", 30))
    profession  = request.form.get("profession", "Healthcare Professional")
    prompt_type = request.form.get("prompt_type", "passage")
    audio_file  = request.files.get("audio")

    # If user is logged in, use their profession from profile
    if user_id:
        user = db.get_user_by_id(user_id)
        if user and user.get("profession"):
            profession = user["profession"]

    tmp_path = None
    filename = "live_recording.webm"
    session_id = str(uuid.uuid4())
    audio_url = ""

    if audio_file and audio_file.filename:
        filename = audio_file.filename
        suffix   = os.path.splitext(filename)[-1] or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            audio_file.save(tmp)
            tmp_path = tmp.name
        log.info("Audio file saved: %s", tmp_path)

    try:
        from analyze import (analyze_audio, _simulate_scores,
                             _build_insights, _trait_label, TRAITS)
        if tmp_path:
            result = analyze_audio(tmp_path, duration, profession, ACOUSTIC_WEIGHT)
            # Upload INSIDE try block — file still exists here
            audio_url = _upload_audio(tmp_path, session_id, filename)
            if audio_url:
                log.info("Audio uploaded to Supabase: %s", audio_url)
        else:
            fused  = _simulate_scores(duration, profession)
            scores = {t: int(round(float(fused[i]) * 100)) for i, t in enumerate(TRAITS)}
            result = {
                "scores":     scores,
                "labels":     {t: _trait_label(v / 100) for t, v in scores.items()},
                "overall":    int(round(sum(scores.values()) / len(TRAITS))),
                "transcript": "",
                "insights":   _build_insights(scores, profession),
                "source":     "simulation (no audio uploaded)",
                "models_used": {"wav2vec2": False, "bert": False, "whisper": False},
            }
    except ImportError:
        # analyze.py deps not installed — full simulation
        import random
        TRAITS_LIST = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
        random.seed(int(duration * 31 + sum(ord(c) for c in profession)))
        scores = {t: random.randint(45, 88) for t in TRAITS_LIST}
        random.seed()
        result = {
            "scores":     scores,
            "labels":     {t: "Moderate" for t in TRAITS_LIST},
            "overall":    int(sum(scores.values()) / len(scores)),
            "transcript": "",
            "insights":   [],
            "source":     "simulation (ML deps not installed)",
            "models_used": {"wav2vec2": False, "bert": False, "whisper": False},
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Generate ideal profession benchmark scores via Groq
    profession_benchmarks = None
    if _groq_available() and profession and result.get('scores'):
        try:
            _gc_bench = _get_groq_client()
            _bench_prompt = (
                f"What are the ideal Big Five personality trait scores (0-100) "
                f"for a highly successful {profession}? "
                f"Based on psychological research, give typical scores top {profession} professionals exhibit.\n"
                "Output ONLY JSON. Replace each INT with a realistic integer:\n"
                '{"Openness": INT, "Conscientiousness": INT, "Extraversion": INT, '
                '"Agreeableness": INT, "Neuroticism": INT}'
            )
            _rb = _gc_bench.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": _bench_prompt}],
                max_tokens=80,
                temperature=0.1,
            )
            import re as _reb
            _mb = _reb.search(r'\{[^}]+\}', _rb.choices[0].message.content.strip())
            if _mb:
                _pb = json.loads(_mb.group())
                _TL = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
                profession_benchmarks = {t: max(0, min(100, int(_pb.get(t, 50)))) for t in _TL}
        except Exception as _eb:
            log.warning("Groq profession benchmark failed: %s", _eb)
    if profession_benchmarks:
        result['profession_benchmarks'] = profession_benchmarks

    now = datetime.now()
    dur = int(duration)

    session_data = {
        "id":           session_id,
        "timestamp":    now.isoformat(),
        "date":         now.strftime("%B %d, %Y"),
        "time":         now.strftime("%H:%M"),
        "duration":     dur,
        "duration_fmt": f"{dur//60:02d}:{dur%60:02d}",
        "filename":     filename,
        "profession":   profession,
        "prompt_type":  prompt_type,
        "audio_url":    audio_url,
        **result,
    }

    saved = db.save_session(session_data, user_id=user_id)
    log.info("Analyzed: overall=%d%% source=%s", saved["overall"], result["source"])
    return jsonify(saved)


# ── Routes: Sessions ───────────────────────────────────────────────────────────

@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    user_id = _get_user_id()
    sessions = db.list_sessions(user_id=user_id)
    return jsonify(sessions)


@app.route("/api/sessions/<sid>", methods=["GET"])
def get_session(sid):
    s = db.get_session(sid)
    if not s:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(s)


@app.route("/api/sessions/<sid>", methods=["DELETE"])
def delete_session(sid):
    db.delete_session(sid)
    return jsonify({"success": True})


# ── Routes: Chat (Groq SSE streaming) ─────────────────────────────────────────

TRAITS_ORDER = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

def _trait_label_simple(v):
    pct = v / 100
    if pct >= 0.80: return "Very High"
    if pct >= 0.65: return "High"
    if pct >= 0.45: return "Moderate"
    if pct >= 0.30: return "Low"
    return "Very Low"


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    SoulSpeak AI Coach streaming chat via Groq API.
    Streams tokens back as Server-Sent Events.
    Also persists both user message and assistant reply to chat_logs table.
    """
    data            = request.get_json(force=True) or {}
    message         = data.get("message", "").strip()
    history         = data.get("history", [])
    personality     = data.get("personality", {})
    profession      = data.get("profession", "Professional")
    conversation_id = data.get("conversation_id")

    if not message:
        return jsonify({"error": "Empty message"}), 400

    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY is not configured on the server"}), 503

    try:
        user_id = _get_user_id()
        db.save_chat_message("user", message,
                             user_id=user_id, conversation_id=conversation_id,
                             model_name=GROQ_MODEL)
    except Exception as exc:
        log.exception("Chat pre-stream error")
        return jsonify({"error": str(exc)}), 500

    # Build system prompt
    if personality:
        scores_block = "\n".join(
            f"  • {t}: {personality.get(t, '?')}%  [{_trait_label_simple(personality.get(t, 50))}]"
            for t in TRAITS_ORDER
        )
    else:
        scores_block = "  (no voice analysis completed yet — invite user to try one)"

    user_name = "there"
    sessions_context = ""
    if user_id:
        u = db.get_user_by_id(user_id)
        if u and u.get("name"):
            user_name = u["name"].split()[0]
        if u and u.get("profession"):
            profession = u["profession"]
        all_sessions = db.list_sessions(user_id=user_id, limit=10)
        if all_sessions:
            sessions_context = "\n\nUser's full session history (most recent first):\n"
            for i, s in enumerate(all_sessions):
                sc = s.get("scores", {})
                overall = s.get("overall", "?")
                date_s  = s.get("date", s.get("date_str", "?"))
                dur_s   = s.get("duration_fmt", "?")
                prof_s  = s.get("profession", profession)
                trait_str = ", ".join(
                    f"{t}: {sc.get(t,'?')}%" for t in TRAITS_ORDER if sc.get(t)
                )
                sessions_context += (
                    f"  Session {i+1}: {date_s} | Duration: {dur_s} | "
                    f"Overall: {overall}% | Profession: {prof_s}\n"
                    f"    Scores — {trait_str}\n"
                )
            sessions_context += "\nUse this history to track the user's growth, compare sessions, and give longitudinal advice."

    system = f"""You are the SoulSpeak AI Voice Coach, a warm and expert personality and communication coach built into the SoulSpeak platform.
SoulSpeak uses the Big Five (OCEAN) personality model to analyse voice recordings and help people communicate more effectively in their professional lives.
You are a core part of SoulSpeak — never mention Groq, any LLM, or any underlying AI model. Always refer to yourself simply as the SoulSpeak AI Coach.

The user's name is {user_name}. Their profession is: {profession}.

Their personality profile from the most recent voice analysis:
{scores_block}{sessions_context}

Profession-specific coaching context:
- If profession is Healthcare/Doctor/Nurse/Therapist: focus on empathic communication, patient clarity, calm authority under pressure, conveying trust.
- If profession is Software Engineer/Developer/Tech: focus on technical communication, explaining complex ideas simply, stakeholder presentations, team collaboration.
- If profession is Teacher/Educator: focus on classroom presence, engaging delivery, storytelling, holding attention, motivating students.
- If profession is Manager/Executive: focus on leadership communication, assertiveness, inspiring teams, concise decision communication.
- If profession is Sales/Marketing: focus on persuasion, rapport-building, confident pitching, closing conversations, emotional intelligence.
- If profession is Lawyer/Legal: focus on structured argumentation, authoritative tone, precision, courtroom presence, client trust.
- If profession is Public Speaker/Presenter: focus on stage presence, vocal variety, audience engagement, storytelling, confidence.
- If profession is Creative/Artist: focus on authentic self-expression, pitching ideas, creative confidence, communicating vision.
- If profession is Student: focus on academic presentations, interview skills, confidence, professional communication growth.
- If profession is Entrepreneur: focus on investor pitching, team leadership, networking, persuasive storytelling.
- If profession is HR/Counsellor: focus on active listening, empathic communication, difficult conversations, trust-building.

Coaching guidelines:
- Address the user by name occasionally to feel personal
- Be warm, expert, and actionable — like a trusted mentor with a psychology background
- Ground every tip in the specific OCEAN score profile above; never give generic advice
- Tailor advice tightly to the user's profession using the context above
- Keep replies concise: 2-4 sentences for simple questions; up to 8 for detailed coaching
- Use 1-2 emojis max per reply; avoid filler like "Great question!"
- If no analysis exists yet, warmly invite the user to record a voice session on SoulSpeak first
- Never reveal you are powered by any third-party model — you are the SoulSpeak AI Coach
"""

    messages = [{"role": "system", "content": system}]
    for m in history[-10:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": message})

    def generate():
        full_reply = []
        try:
            client = _get_groq_client()
            stream = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                stream=True,
                temperature=0.75,
                max_tokens=512,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                token = delta.content or ""
                finish = chunk.choices[0].finish_reason
                if token:
                    full_reply.append(token)
                    yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
                if finish:
                    yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
                    break

        except Exception as e:
            msg = f"❌ Groq error: {e}"
            full_reply.append(msg)
            yield f"data: {json.dumps({'token': msg, 'done': True})}\n\n"

        if full_reply:
            db.save_chat_message(
                "assistant", "".join(full_reply),
                user_id=user_id, conversation_id=conversation_id, model_name=GROQ_MODEL,
            )

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/chat/history", methods=["GET"])
def chat_history():
    user_id         = _get_user_id()
    conversation_id = request.args.get("conversation_id")
    session_id      = request.args.get("session_id")
    limit           = int(request.args.get("limit", 100))
    msgs = db.get_chat_history(user_id=user_id, session_id=session_id,
                               conversation_id=conversation_id, limit=limit)
    return jsonify(msgs)


@app.route("/api/chat/history", methods=["DELETE"])
def clear_chat_history():
    user_id    = _get_user_id()
    session_id = request.args.get("session_id")
    deleted    = db.delete_chat_history(user_id=user_id, session_id=session_id)
    return jsonify({"success": True, "deleted": deleted})


@app.route("/api/chat/sessions", methods=["GET"])
def list_chat_sessions():
    user_id = _get_user_id()
    if not user_id:
        return jsonify([])
    return jsonify(db.list_chat_sessions(user_id))


@app.route("/api/chat/sessions/<conversation_id>", methods=["DELETE"])
def delete_chat_session(conversation_id):
    user_id = _get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
    deleted = db.delete_chat_history(user_id=user_id, conversation_id=conversation_id)
    return jsonify({"success": True, "deleted": deleted})


# ── Routes: Chat models & Status ──────────────────────────────────────────────

@app.route("/api/ollama/models", methods=["GET"])
def ollama_models():
    # Kept for frontend compatibility — now returns Groq model info
    return jsonify({
        "models":   _GROQ_MODELS,
        "selected": GROQ_MODEL,
        "provider": "groq",
        "online":   _groq_available(),
    })


@app.route("/api/status", methods=["GET"])
def status():
    stats = db.get_stats()
    return jsonify({
        "server":  "SoulSpeak v3.0",
        "models":  MODEL_STATUS,
        "groq":    {"online": _groq_available(), "model": GROQ_MODEL},
        "database": stats,
    })


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    print("\n" + "=" * 60)
    print("  SoulSpeak - Speech Personality Recognition v3.0")
    print("=" * 60)

    # Init database
    db.init()
    stats = db.get_stats()
    print(f"\n  Database       : {stats['db_type']}")
    print(f"  Users          : {stats['users']}")
    print(f"  Sessions       : {stats['sessions']}")
    print(f"  Chat messages  : {stats['chat_messages']}")

    # Load ML models
    _load_ml_models()
    print(f"\n  Wav2Vec2 model : {'OK' if MODEL_STATUS.get('wav2vec2') else 'Not loaded (simulation fallback)'}")
    print(f"  BERT model     : {'OK' if MODEL_STATUS.get('bert')     else 'Not loaded (simulation fallback)'}")
    print(f"  Whisper ASR    : {'OK' if MODEL_STATUS.get('whisper')  else 'Not loaded'}")

    # Check Groq
    if _groq_available():
        print(f"\n  Groq API       : Configured ✓")
        print(f"  Active model   : {GROQ_MODEL}")
    else:
        print(f"\n  Groq API       : NOT configured — set GROQ_API_KEY")

    print(f"\n  Open in browser: http://localhost:{port}")
    print("=" * 60 + "\n")

    app.run(debug=False, port=port, host="0.0.0.0", threaded=True)
