#!/usr/bin/env python3
"""
Patch script: adds profession_benchmarks Groq call to app.py
and updates _updateProfessionCard in index.html.
Run on the server:  python3 patch_profession_benchmarks.py
"""
import os, sys, re

SERVER_APPPY   = "/root/soulspeak/app.py"
SERVER_HTML    = "/root/soulspeak/templates/index.html"

# ── 1. Patch app.py ─────────────────────────────────────────────────────────

APPPY_OLD = """    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    now = datetime.now()"""

APPPY_NEW = """    finally:
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
                f"Based on psychological research, give typical scores top {profession} professionals exhibit.\\n"
                "Output ONLY JSON. Replace each INT with a realistic integer:\\n"
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
            _mb = _reb.search(r'\\{[^}]+\\}', _rb.choices[0].message.content.strip())
            if _mb:
                _pb = json.loads(_mb.group())
                _TL = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
                profession_benchmarks = {t: max(0, min(100, int(_pb.get(t, 50)))) for t in _TL}
        except Exception as _eb:
            log.warning("Groq profession benchmark failed: %s", _eb)
    if profession_benchmarks:
        result['profession_benchmarks'] = profession_benchmarks

    now = datetime.now()"""

def patch_apppy():
    if not os.path.exists(SERVER_APPPY):
        print(f"ERROR: {SERVER_APPPY} not found"); return False
    text = open(SERVER_APPPY).read()
    if 'profession_benchmarks' in text:
        print("app.py: profession_benchmarks already present — skipping"); return True
    if APPPY_OLD not in text:
        print("app.py: anchor not found — check file manually"); return False
    patched = text.replace(APPPY_OLD, APPPY_NEW, 1)
    open(SERVER_APPPY, 'w').write(patched)
    print("app.py: PATCHED OK")
    return True

# ── 2. Patch index.html — replace _updateProfessionCard function ─────────────

HTML_OLD_FUNC_START = "function _updateProfessionCard(profession, scores) {"
HTML_OLD_FUNC_DETECT = "function _updateProfessionCard(profession, scores, benchmarks)"

def patch_html():
    if not os.path.exists(SERVER_HTML):
        print(f"ERROR: {SERVER_HTML} not found"); return False
    text = open(SERVER_HTML).read()

    if HTML_OLD_FUNC_DETECT in text:
        print("index.html: _updateProfessionCard already has benchmarks param — skipping"); return True

    # Find the old function and replace it entirely
    # We'll use the start marker and scan for the matching closing brace
    start_marker = "function _updateProfessionCard("
    start_idx = text.find(start_marker)
    if start_idx == -1:
        print("index.html: _updateProfessionCard not found"); return False

    # Find the end of the function by counting braces
    depth = 0
    found_open = False
    end_idx = start_idx
    for i, ch in enumerate(text[start_idx:], start_idx):
        if ch == '{':
            depth += 1
            found_open = True
        elif ch == '}':
            depth -= 1
            if found_open and depth == 0:
                end_idx = i + 1
                break

    old_func = text[start_idx:end_idx]

    NEW_FUNC = r"""function _updateProfessionCard(profession, scores, benchmarks) {
  const tagEl   = document.getElementById('prof-tag');
  const titleEl = document.getElementById('prof-title');
  const bodyEl  = document.getElementById('prof-body');
  if (!tagEl && !titleEl && !bodyEl) return;

  const p = (profession || '').toLowerCase();
  let cat = 'general';
  if (p.match(/doctor|health|nurs|medic|therap|physician|clinical/))         cat = 'healthcare';
  else if (p.match(/teach|educat|professor|lectur|instruc/))                  cat = 'teacher';
  else if (p.match(/engineer|develop|software|tech|program|devop/))           cat = 'engineer';
  else if (p.match(/business|manag|execut|director|ceo|coo|cfo/))            cat = 'business';
  else if (p.match(/legal|lawyer|attorney|solicitor|barrister|law/))          cat = 'legal';
  else if (p.match(/sales|market/))                                           cat = 'sales';
  else if (p.match(/student|undergrad|postgrad|academic|researcher/))         cat = 'student';
  else if (p.match(/speaker|present|host|anchor|broadcast/))                  cat = 'speaker';
  else if (p.match(/counsel|social work|psychol|coach|mentor/))               cat = 'counsellor';
  else if (p.match(/entrepreneur|founder|startup|venture/))                   cat = 'entrepreneur';

  const META = {
    healthcare:   { icon:'🏥', title:`Ideal Big Five scores to become a top ${profession}` },
    teacher:      { icon:'📚', title:`Ideal Big Five scores to become a top ${profession}` },
    engineer:     { icon:'💻', title:`Ideal Big Five scores to become a top ${profession}` },
    business:     { icon:'💼', title:`Ideal Big Five scores to become a top ${profession}` },
    legal:        { icon:'⚖️', title:`Ideal Big Five scores to become a top ${profession}` },
    sales:        { icon:'📈', title:`Ideal Big Five scores to become a top ${profession}` },
    student:      { icon:'🎓', title:`Ideal Big Five scores to become a top ${profession}` },
    speaker:      { icon:'🎤', title:`Ideal Big Five scores to become a top ${profession}` },
    counsellor:   { icon:'🤝', title:`Ideal Big Five scores to become a top ${profession}` },
    entrepreneur: { icon:'🚀', title:`Ideal Big Five scores to become a top ${profession}` },
    general:      { icon:'📋', title:`Ideal Big Five scores for a top ${profession}` },
  };
  const m = META[cat] || META.general;

  const TCOLS = { Openness:'#8b5cf6', Conscientiousness:'#06b6d4', Extraversion:'#f59e0b', Agreeableness:'#10b981', Neuroticism:'#ef4444' };
  const TRAITS = ['Openness','Conscientiousness','Extraversion','Agreeableness','Neuroticism'];

  if (tagEl)   tagEl.textContent   = `${m.icon} Profession Insight · ${profession}`;
  if (titleEl) titleEl.textContent = m.title;

  if (!bodyEl) return;

  if (benchmarks && Object.keys(benchmarks).length) {
    const rows = TRAITS.map(t => {
      const user  = scores[t] || 0;
      const ideal = benchmarks[t] || 50;
      const gap   = user - ideal;
      const col   = TCOLS[t] || '#8b5cf6';
      const gapLabel = gap >= 0
        ? `<span style="color:#10b981">+${gap}% ↑ above ideal</span>`
        : `<span style="color:#f59e0b">${gap}% below ideal</span>`;
      return `
        <div style="margin-bottom:14px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;font-size:0.82rem">
            <strong style="color:${col}">${t}</strong>
            <span style="color:var(--text2);font-size:0.75rem">${gapLabel}</span>
          </div>
          <div style="position:relative;height:10px;background:var(--surface3);border-radius:5px;margin-bottom:3px">
            <div style="height:100%;width:${user}%;background:${col};border-radius:5px;opacity:0.9;transition:width 1s ease"></div>
            <div style="position:absolute;top:-3px;left:${ideal}%;width:2px;height:16px;background:#fff;opacity:0.7;border-radius:1px"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:var(--text3)">
            <span>You: <strong style="color:${col}">${user}%</strong></span>
            <span>Ideal: <strong style="color:#fff">${ideal}%</strong> <span style="opacity:0.5">(white line)</span></span>
          </div>
        </div>`;
    }).join('');

    const behind = TRAITS.filter(t => (scores[t]||0) < (benchmarks[t]||50) - 5);
    const advice = behind.length
      ? `Focus on: <strong>${behind.join(', ')}</strong> — these are furthest from the ideal profile for a ${profession}.`
      : `Your profile closely matches or exceeds the ideal for a ${profession}. Keep refining through practice.`;

    bodyEl.innerHTML =
      `<p style="font-size:0.82rem;color:var(--text2);margin-bottom:16px">
        The white line on each bar shows the <strong>ideal score</strong> for a professional ${profession}. Your score is the coloured fill.
      </p>` +
      rows +
      `<div style="margin-top:12px;padding:10px 14px;background:rgba(139,92,246,0.08);border-left:3px solid var(--accent);border-radius:6px;font-size:0.82rem;color:var(--text2)">
        ${advice}
      </div>`;
  } else {
    bodyEl.innerHTML = `
      <div style="text-align:center;padding:20px;color:var(--text3);font-size:0.85rem">
        ⏳ Generating ideal ${profession} benchmark scores via AI…
      </div>`;
  }
}"""

    patched = text[:start_idx] + NEW_FUNC + text[end_idx:]
    open(SERVER_HTML, 'w').write(patched)
    print("index.html: _updateProfessionCard PATCHED OK")
    return True

# ── Also fix the call site in _populateResultsView ──────────────────────────

def patch_html_callsite():
    if not os.path.exists(SERVER_HTML):
        return False
    text = open(SERVER_HTML).read()

    # Fix the call from _updateProfessionCard(_prof, session.scores || {}) to include benchmarks
    old_call = "_updateProfessionCard(_prof, session.scores || {})"
    new_call = "_updateProfessionCard(_prof, session.scores || {}, session.profession_benchmarks || null)"

    if new_call in text:
        print("index.html: call site already updated — skipping"); return True
    if old_call not in text:
        print("index.html: call site not found — check manually"); return False
    patched = text.replace(old_call, new_call, 1)
    open(SERVER_HTML, 'w').write(patched)
    print("index.html: call site PATCHED OK")
    return True

# ── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ok1 = patch_apppy()
    ok2 = patch_html()
    ok3 = patch_html_callsite()
    if ok1 and ok2 and ok3:
        print("\nAll patches applied. Restart gunicorn:")
        print("  systemctl restart soulspeak")
    else:
        print("\nSome patches failed — check output above.")
