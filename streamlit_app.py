"""
GlioblastomaGPT — Streamlit Cloud Entry Point
All UI code is here (not imported from another module) so Streamlit reruns work correctly.
Secrets from Streamlit Cloud dashboard are injected into env before any import.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# ── Inject Streamlit Cloud secrets into env BEFORE any gbm_copilot import ────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import streamlit as _st
    for _k, _v in _st.secrets.items():
        if isinstance(_v, str) and _k not in os.environ:
            os.environ[_k] = _v
except Exception:
    pass  # Running locally with .env — fine

# ── Now safe to import project code ──────────────────────────────────────────
import asyncio
import uuid
import streamlit as st

# ── Page Config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="GlioblastomaGPT",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
  --bg: #0a0e1a; --surface: #111827; --surface2: #1f2937;
  --accent: #6366f1; --accent2: #8b5cf6;
  --text: #e5e7eb; --muted: #9ca3af;
  --success: #10b981; --warning: #f59e0b; --danger: #ef4444;
  --border: #374151;
}
.stApp { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stRadio label { cursor: pointer; }
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
  color: var(--text) !important; font-size: 0.9rem;
}
.user-msg {
  background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 100%);
  border-radius: 16px 16px 4px 16px; padding: 14px 18px;
  margin: 8px 0; max-width: 80%; margin-left: auto;
  color: #fff; font-size: 0.95rem; line-height: 1.6;
  box-shadow: 0 4px 12px rgba(99,102,241,0.2);
}
.bot-msg {
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 4px 16px 16px 16px; padding: 16px 20px;
  margin: 8px 0; max-width: 90%; color: var(--text);
  font-size: 0.94rem; line-height: 1.7; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.bot-msg h1,.bot-msg h2,.bot-msg h3 { color: var(--text); }
.bot-msg strong { color: #c4b5fd; }
.bot-msg blockquote { border-left: 3px solid var(--warning); padding-left: 12px; color: var(--muted); }
.bot-msg a { color: var(--accent); }
.disclaimer {
  background: linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
  border: 1px solid #f59e0b55; border-left: 4px solid #f59e0b;
  border-radius: 8px; padding: 10px 16px; margin: 4px 0 16px 0;
  font-size: 0.8rem; color: #fcd34d;
}
.mode-badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; margin-left: 6px; }
.mode-patient { background: #065f46; border: 1px solid #10b981; color: #6ee7b7; }
.mode-caregiver { background: #1e3a8a; border: 1px solid #6366f1; color: #a5b4fc; }
.mode-clinician { background: #7c2d12; border: 1px solid #f97316; color: #fdba74; }
.groq-badge { display: inline-block; background: #0f2027; border: 1px solid #f97316; color: #fdba74; padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }
.citation-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; margin: 5px 0; font-size: 0.82rem; color: var(--muted); }
.citation-card a { color: var(--accent); text-decoration: none; }
.citation-card:hover { border-color: var(--accent); }
.trial-card { background: linear-gradient(135deg,var(--surface) 0%,#0f1a35 100%); border: 1px solid #374151; border-left: 3px solid var(--accent); border-radius: 10px; padding: 14px 18px; margin: 8px 0; font-size: 0.86rem; }
.phase-badge { display: inline-block; background: var(--accent); color: #fff; font-size: 0.72rem; font-weight: 600; padding: 2px 8px; border-radius: 20px; margin-left: 8px; }
.safety-flag { background: #450a0a; border: 1px solid var(--danger); border-radius: 6px; padding: 4px 10px; font-size: 0.76rem; color: #fca5a5; display: inline-block; margin: 2px; }
.conf-bar { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; display: inline-block; width: 120px; vertical-align: middle; margin-left: 6px; }
.conf-fill { height: 100%; border-radius: 2px; }
.stButton > button { background: linear-gradient(135deg,var(--accent) 0%,var(--accent2) 100%); color: white; border: none; border-radius: 10px; font-weight: 600; transition: all 0.2s; }
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(99,102,241,0.4); }
</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)


# ── In-process pipeline call ──────────────────────────────────────────────────
def run_pipeline(query: str, literacy_mode: str, session_id: str, conversation_history: list) -> dict:
    """
    Calls the LangGraph pipeline in a fresh thread with its own event loop.
    This avoids the uvloop/asyncio conflict on Streamlit Cloud (Python 3.14).
    """
    try:
        from gbm_copilot.agents.graph import run_query
        import concurrent.futures

        async def _run():
            return await run_query(
                query=query,
                literacy_mode=literacy_mode,
                session_id=session_id,
                conversation_history=conversation_history,
            )

        # Always run in a new thread with a fresh event loop.
        # This is the ONLY safe way on Streamlit Cloud (uvloop, Python 3.14).
        def _thread_target():
            return asyncio.run(_run())

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_thread_target)
            return future.result(timeout=120)

    except Exception as e:
        import traceback
        return {
            "final_answer": f"⚠️ **Error:** {e}\n\n```\n{traceback.format_exc()[-800:]}\n```",
            "error": str(e),
        }


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "literacy_mode" not in st.session_state:
    st.session_state.literacy_mode = "patient"
if "retrieval_mode" not in st.session_state:
    st.session_state.retrieval_mode = "numpy"

# ── Apply retrieval mode to config (mutate cfg so hybrid_retriever picks it up)
import gbm_copilot.config as _cfg
_cfg.RETRIEVAL_MODE = st.session_state.retrieval_mode  # type: ignore

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 GlioblastomaGPT")
    st.markdown(
        "<div style='color:#9ca3af;font-size:0.82rem;margin-bottom:4px;'>GBM Research &amp; Care Intelligence</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<span class='groq-badge'>⚡ Powered by Groq — Free &amp; Fast</span>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 📖 Literacy Mode")
    _mode_options = ["patient", "caregiver", "clinician"]
    _mode_labels = {
        "patient":   "👤 Patient — Plain English",
        "caregiver": "❤️  Caregiver — Detailed",
        "clinician": "⚕️  Clinician — Technical",
    }

    # key="literacy_mode" auto-syncs st.session_state.literacy_mode with the radio
    st.radio(
        label="literacy_mode_select",
        options=_mode_options,
        format_func=lambda x: _mode_labels[x],
        key="literacy_mode",         # ← canonical Streamlit pattern for persistence
        label_visibility="collapsed",
    )

    _chosen = st.session_state.literacy_mode
    _badge_text = {"patient": "Patient mode", "caregiver": "Caregiver mode", "clinician": "Clinician mode"}[_chosen]
    st.markdown(f"Active: <span class='mode-badge mode-{_chosen}'>{_badge_text}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🔬 Retrieval Version**")

    _ver_options = ["numpy", "pinecone"]
    _ver_labels = {
        "numpy":    "🅱️ Version B — NumPy (no DB)",
        "pinecone": "🅰️ Version A — Pinecone",
    }
    st.radio(
        label="retrieval_version",
        options=_ver_options,
        format_func=lambda x: _ver_labels[x],
        key="retrieval_mode",
        label_visibility="collapsed",
    )
    # Sync to config so hybrid_retriever uses the right backend
    import gbm_copilot.config as _cfg_sidebar
    _cfg_sidebar.RETRIEVAL_MODE = st.session_state.retrieval_mode  # type: ignore

    if st.session_state.retrieval_mode == "pinecone":
        st.markdown(
            "<span style='display:inline-block;background:#065f46;border:1px solid #10b981;"
            "color:#6ee7b7;padding:3px 10px;border-radius:20px;font-size:0.72rem;font-weight:600;'>"
            "🅰️ Version A Active — Pinecone</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<span style='display:inline-block;background:#1e3a8a;border:1px solid #6366f1;"
            "color:#a5b4fc;padding:3px 10px;border-radius:20px;font-size:0.72rem;font-weight:600;'>"
            "🅱️ Version B Active — NumPy</span>",
            unsafe_allow_html=True,
        )
    st.markdown("---")

    st.markdown("### 💬 Try These")
    for _q in [
        "What does IDH wild-type mean for treatment?",
        "What clinical trials are open for GBM?",
        "How does temozolomide work?",
        "What is the Stupp protocol?",
        "I'm scared — what do I do?",
        "Does MGMT methylation affect prognosis?",
        "What are the side effects of bevacizumab?",
    ]:
        if st.button(_q, use_container_width=True, key=f"ex_{_q[:20]}"):
            st.session_state["_prefill"] = _q

    st.markdown("---")
    if st.button("🗑️ New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("---")
    st.markdown("""
<div style="font-size:0.74rem;color:#6b7280;line-height:1.7;">
<strong style="color:#9ca3af;">Data sources</strong><br>
• PubMed abstracts<br>• clinicaltrials.gov (live API)<br>• FDA drug labels<br>• NCCN/ASCO guidelines<br><br>
<strong style="color:#9ca3af;">BM25 + MiniLM + RRF</strong><br>Medical synonym expansion<br>3-stage query rewriter
</div>""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:16px 0 8px;">
  <div style="font-size:2.2rem;font-weight:800;
    background:linear-gradient(135deg,#6366f1,#8b5cf6,#c084fc);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    🧠 GlioblastomaGPT
  </div>
  <div style="color:#9ca3af;font-size:0.88rem;margin-top:6px;">
    GBM Research &amp; Care Intelligence &nbsp;·&nbsp;
    <span class='mode-badge mode-{_chosen}'>{_badge_text}</span> &nbsp;·&nbsp;
    <span class='groq-badge'>⚡ Groq Free</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer">⚕️ <strong>This is not medical advice.</strong> '
    'Always consult your neuro-oncology team for medical decisions. For educational purposes only.</div>',
    unsafe_allow_html=True,
)


# ── Render message ─────────────────────────────────────────────────────────────
def render_message(msg: dict):
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        return

    data = msg.get("data", {})
    answer_html = msg["content"].replace("\n", "<br>")
    st.markdown(f'<div class="bot-msg">{answer_html}</div>', unsafe_allow_html=True)

    flags = data.get("safety_flags", [])
    if flags:
        flag_labels = {
            "dosage_advice": "🚫 Dosage advice blocked",
            "prognosis_stats": "📊 Prognosis context added",
            "distress_signal": "💙 Emotional support routed",
            "low_confidence": "⚠️ Limited evidence",
            "confidence_blocked": "🔒 Low confidence",
        }
        st.markdown(" ".join(f'<span class="safety-flag">{flag_labels.get(f, f)}</span>' for f in flags), unsafe_allow_html=True)

    conf = data.get("confidence_score", 0.0)
    if conf > 0:
        conf_color = "#10b981" if conf > 0.7 else "#f59e0b" if conf > 0.5 else "#ef4444"
        conf_label = "High" if conf > 0.7 else "Moderate" if conf > 0.5 else "Low"
        qtype = data.get("query_type", "")
        st.markdown(
            f'<div style="font-size:0.74rem;color:#6b7280;margin:6px 0 2px;">Confidence: <strong>{conf_label}</strong> ({conf:.0%})'
            f'<div class="conf-bar"><div class="conf-fill" style="width:{conf*100:.0f}%;background:{conf_color};"></div></div>'
            + (f'&nbsp;·&nbsp;<em>{qtype}</em>' if qtype else '') + '</div>',
            unsafe_allow_html=True,
        )

    citations = data.get("citations", [])
    if citations:
        with st.expander(f"📚 {len(citations)} Sources", expanded=False):
            for c in citations:
                pmid_html = f' <span style="color:#6366f1;font-size:0.72rem;">PMID:{c["pmid"]}</span>' if c.get("pmid") else ""
                st.markdown(
                    f'<div class="citation-card"><a href="{c.get("url","#")}" target="_blank">↗ {c.get("title","Source")[:100]}</a> '
                    f'<span style="color:#6b7280;">({c.get("source","")}, {c.get("year","")})</span>{pmid_html}</div>',
                    unsafe_allow_html=True,
                )

    trials = data.get("trial_results", [])
    if trials:
        with st.expander(f"🔬 {len(trials)} Clinical Trials Found", expanded=True):
            for t in trials[:8]:
                interventions = ", ".join(i["name"] for i in t.get("interventions", [])[:3])
                locs = t.get("locations", [])
                loc_str = ", ".join(f"{l.get('city','')}, {l.get('state','')}" for l in locs[:2] if l.get("city")) or "Multiple locations"
                st.markdown(f"""<div class="trial-card"><strong>{t.get("title","")[:80]}</strong>
<span class="phase-badge">{t.get("phase","")}</span>
<br><span style="color:#6b7280;font-size:0.78rem;">{t.get("nct_id","")} · {loc_str}</span>
{"<br><span style='color:#9ca3af;'>Intervention: " + interventions + "</span>" if interventions else ""}
<br><a href="{t.get('url','')}" target="_blank" style="color:#6366f1;font-size:0.78rem;">View on ClinicalTrials.gov ↗</a>
</div>""", unsafe_allow_html=True)

    if data.get("emotional_resources"):
        with st.expander("💙 Support Resources", expanded=False):
            for r in data["emotional_resources"]:
                phone = f" · {r['phone']}" if r.get("phone") else ""
                st.markdown(f'<div class="citation-card"><a href="{r["url"]}" target="_blank">↗ {r["name"]}</a><br><span style="color:#6b7280;">{r["description"]}{phone}</span></div>', unsafe_allow_html=True)


# ── Message history ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    render_message(msg)


# ── Chat input ─────────────────────────────────────────────────────────────────
query = st.chat_input(placeholder="Ask about treatments, trials, drugs, or anything about GBM...")

# Handle example question prefill
if st.session_state.get("_prefill"):
    query = st.session_state["_prefill"]
    del st.session_state["_prefill"]

if query:
    user_msg = {"role": "user", "content": query}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    st.markdown(
        f'<div style="font-size:0.72rem;color:#6b7280;text-align:right;margin:-4px 0 8px;">'
        f'Mode: <strong>{st.session_state.literacy_mode}</strong> &nbsp;·&nbsp; Retrieval: <strong>NUMPY</strong></div>',
        unsafe_allow_html=True,
    )

    with st.spinner("🧠 Thinking... (Groq llama-3.3-70b)"):
        result = run_pipeline(
            query=query,
            literacy_mode=st.session_state.literacy_mode,
            session_id=st.session_state.session_id,
            conversation_history=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-8:]
                if m["role"] in ("user", "assistant")
            ],
        )

    bot_msg = {
        "role": "assistant",
        "content": result.get("final_answer", "No response received."),
        "data": {
            "citations": result.get("citations", []),
            "confidence_score": result.get("confidence_score", 0.0),
            "safety_flags": result.get("safety_flags", []),
            "trial_results": result.get("trial_results", []),
            "emotional_resources": result.get("emotional_resources", []),
            "is_blocked": result.get("is_blocked", False),
            "query_type": result.get("query_type", ""),
        },
    }
    st.session_state.messages.append(bot_msg)
    render_message(bot_msg)
    st.rerun()
