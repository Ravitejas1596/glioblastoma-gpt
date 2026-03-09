"""
Streamlit Chat UI for GlioblastomaGPT
Dark, premium medical interface with:
- Patient / Caregiver / Clinician literacy modes (fixed for Streamlit 1.55+)
- Runtime Version A (Pinecone) / Version B (NumPy) switcher
- Streaming chat with expandable citations
- Clinical trial cards
- Persistent medical disclaimer banner
"""
from __future__ import annotations

import uuid
import httpx
import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────
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

/* Radio buttons — fix visibility in dark mode */
[data-testid="stSidebar"] .stRadio label { cursor: pointer; }
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
  color: var(--text) !important; font-size: 0.9rem;
}

/* Messages */
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

/* Disclaimer */
.disclaimer {
  background: linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
  border: 1px solid #f59e0b55; border-left: 4px solid #f59e0b;
  border-radius: 8px; padding: 10px 16px; margin: 4px 0 16px 0;
  font-size: 0.8rem; color: #fcd34d;
}

/* Version badge */
.version-badge-a {
  display: inline-block; background: #065f46; border: 1px solid #10b981;
  color: #6ee7b7; padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600;
}
.version-badge-b {
  display: inline-block; background: #1e3a8a; border: 1px solid #6366f1;
  color: #a5b4fc; padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600;
}

/* Literacy mode badge */
.mode-badge {
  display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; margin-left: 6px;
}
.mode-patient { background: #065f46; border: 1px solid #10b981; color: #6ee7b7; }
.mode-caregiver { background: #1e3a8a; border: 1px solid #6366f1; color: #a5b4fc; }
.mode-clinician { background: #7c2d12; border: 1px solid #f97316; color: #fdba74; }

/* Citation cards */
.citation-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 10px 14px; margin: 5px 0; font-size: 0.82rem; color: var(--muted);
}
.citation-card a { color: var(--accent); text-decoration: none; }
.citation-card:hover { border-color: var(--accent); }

/* Trial cards */
.trial-card {
  background: linear-gradient(135deg,var(--surface) 0%,#0f1a35 100%);
  border: 1px solid #374151; border-left: 3px solid var(--accent);
  border-radius: 10px; padding: 14px 18px; margin: 8px 0; font-size: 0.86rem;
}
.phase-badge {
  display: inline-block; background: var(--accent); color: #fff;
  font-size: 0.72rem; font-weight: 600; padding: 2px 8px; border-radius: 20px; margin-left: 8px;
}

/* Safety flag */
.safety-flag {
  background: #450a0a; border: 1px solid var(--danger); border-radius: 6px;
  padding: 4px 10px; font-size: 0.76rem; color: #fca5a5; display: inline-block; margin: 2px;
}

/* Confidence bar */
.conf-bar { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; display: inline-block; width: 120px; vertical-align: middle; margin-left: 6px; }
.conf-fill { height: 100%; border-radius: 2px; }

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg,var(--accent) 0%,var(--accent2) 100%);
  color: white; border: none; border-radius: 10px; font-weight: 600;
  transition: all 0.2s;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(99,102,241,0.4); }

/* Select / radio fix */
.stRadio > div { gap: 6px; }
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] { gap: 0; }

/* Index stats */
.stat-box {
  background: var(--surface2); border: 1px solid var(--border); border-radius: 8px;
  padding: 10px 14px; margin: 6px 0; font-size: 0.82rem;
}
</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

API_BASE = "http://localhost:8000"

# ── Session state initialisation (must happen BEFORE any widget) ──────────────
def _init_state():
    defaults = {
        "messages": [],
        "session_id": str(uuid.uuid4()),
        "literacy_mode": "patient",
        "retrieval_mode": "numpy",
        "_prefill": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helper: call API ──────────────────────────────────────────────────────────
def api_get(path: str) -> dict:
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_post(path: str, data: dict) -> dict:
    try:
        r = httpx.post(f"{API_BASE}{path}", json=data, timeout=120)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 GlioblastomaGPT")
    st.markdown(
        "<div style='color:#9ca3af;font-size:0.82rem;margin-bottom:16px;'>"
        "GBM Research &amp; Care Intelligence</div>",
        unsafe_allow_html=True,
    )

    # ── Literacy Mode ─────────────────────────────────────────────────────────
    st.markdown("### 📖 Literacy Mode")
    st.markdown(
        "<div style='color:#6b7280;font-size:0.78rem;margin-bottom:8px;'>"
        "Controls how answers are explained</div>",
        unsafe_allow_html=True,
    )

    # Use index-based approach with explicit on_change to avoid key conflicts
    _mode_options = ["patient", "caregiver", "clinician"]
    _mode_labels = {
        "patient":   "👤 Patient — Plain English",
        "caregiver": "❤️  Caregiver — Detailed",
        "clinician": "⚕️  Clinician — Technical",
    }
    _current_idx = _mode_options.index(st.session_state.literacy_mode)

    def _on_mode_change():
        # called before rerun — read the widget value from its key
        st.session_state.literacy_mode = st.session_state["_literacy_radio"]

    st.radio(
        label="literacy_mode_select",
        options=_mode_options,
        format_func=lambda x: _mode_labels[x],
        index=_current_idx,
        key="_literacy_radio",
        on_change=_on_mode_change,
        label_visibility="collapsed",
    )

    # Show active mode badge
    _badge_class = f"mode-{st.session_state.literacy_mode}"
    _badge_text = {"patient": "Patient mode", "caregiver": "Caregiver mode", "clinician": "Clinician mode"}[st.session_state.literacy_mode]
    st.markdown(
        f"Active: <span class='mode-badge {_badge_class}'>{_badge_text}</span>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Version A / B ─────────────────────────────────────────────────────────
    st.markdown("### 🔬 Retrieval Version")
    st.markdown(
        "<div style='color:#6b7280;font-size:0.78rem;margin-bottom:8px;'>"
        "Switch between Pinecone (A) and NumPy (B)</div>",
        unsafe_allow_html=True,
    )

    # Fetch current backend mode
    _health = api_get("/config")
    _backend_mode = _health.get("retrieval_mode", st.session_state.retrieval_mode)
    st.session_state.retrieval_mode = _backend_mode

    _version_options = ["numpy", "pinecone"]
    _version_labels = {
        "numpy":    "🅱️  Version B — NumPy (no DB)",
        "pinecone": "🅰️  Version A — Pinecone",
    }
    _version_idx = _version_options.index(st.session_state.retrieval_mode)

    def _on_version_change():
        new_mode = st.session_state["_version_radio"]
        result = api_post("/config", {"retrieval_mode": new_mode})
        if "error" not in result:
            st.session_state.retrieval_mode = new_mode
        else:
            st.error(f"Could not switch: {result['error']}")

    st.radio(
        label="version_select",
        options=_version_options,
        format_func=lambda x: _version_labels[x],
        index=_version_idx,
        key="_version_radio",
        on_change=_on_version_change,
        label_visibility="collapsed",
    )

    # Version badge + stats
    if st.session_state.retrieval_mode == "numpy":
        st.markdown("<span class='version-badge-b'>🅱️ Version B Active — NumPy hybrid RAG</span>", unsafe_allow_html=True)
        _stats = api_get("/health")
        _idx_stats = _stats.get("index_stats", {})
        if _idx_stats.get("total_vectors"):
            st.markdown(
                f"<div class='stat-box'>📊 <strong>{_idx_stats['total_vectors']:,}</strong> vectors "
                f"| dim: {_idx_stats.get('embedding_dim', 1024)}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<span class='version-badge-a'>🅰️ Version A Active — Pinecone</span>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Example Questions ─────────────────────────────────────────────────────
    st.markdown("### 💬 Try These")
    _examples = [
        "What does IDH wild-type mean for treatment?",
        "What clinical trials are open for GBM?",
        "How does temozolomide work?",
        "What is the Stupp protocol?",
        "I'm scared — what do I do?",
        "Does MGMT methylation affect prognosis?",
        "What are the side effects of bevacizumab?",
    ]
    for q in _examples:
        if st.button(q, use_container_width=True, key=f"ex_{q[:20]}"):
            st.session_state["_prefill"] = q

    st.markdown("---")
    if st.button("🗑️ New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("---")
    st.markdown("""
<div style="font-size:0.74rem;color:#6b7280;line-height:1.7;">
<strong style="color:#9ca3af;">Data sources</strong><br>
• PubMed (15,973 abstracts)<br>
• clinicaltrials.gov (live API)<br>
• FDA drug labels (openFDA)<br>
• NCCN/ASCO guidelines<br><br>
<strong style="color:#9ca3af;">BM25 + BGE-M3 + RRF</strong><br>
Medical synonym expansion<br>
3-stage query rewriter
</div>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
_mode_badge = f"<span class='mode-badge mode-{st.session_state.literacy_mode}'>{_badge_text}</span>"
_ver_badge = (
    "<span class='version-badge-b'>🅱️ Version B</span>"
    if st.session_state.retrieval_mode == "numpy"
    else "<span class='version-badge-a'>🅰️ Version A</span>"
)

st.markdown(f"""
<div style="text-align:center;padding:16px 0 8px;">
  <div style="font-size:2.2rem;font-weight:800;
    background:linear-gradient(135deg,#6366f1,#8b5cf6,#c084fc);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    🧠 GlioblastomaGPT
  </div>
  <div style="color:#9ca3af;font-size:0.88rem;margin-top:6px;">
    GBM Research &amp; Care Intelligence &nbsp;·&nbsp; {_mode_badge} &nbsp;·&nbsp; {_ver_badge}
  </div>
</div>
""", unsafe_allow_html=True)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="disclaimer">⚕️ <strong>This is not medical advice.</strong> '
    'Always consult your neuro-oncology team for medical decisions. '
    'For educational purposes only.</div>',
    unsafe_allow_html=True,
)


# ── Render a single message ───────────────────────────────────────────────────
def render_message(msg: dict):
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        return

    data = msg.get("data", {})

    # Main answer — render as markdown inside a styled div
    import re
    answer_html = msg["content"].replace("\n", "<br>")
    st.markdown(f'<div class="bot-msg">{answer_html}</div>', unsafe_allow_html=True)

    # Safety flags
    flags = data.get("safety_flags", [])
    if flags:
        flag_labels = {
            "dosage_advice": "🚫 Dosage advice blocked",
            "prognosis_stats": "📊 Prognosis context added",
            "distress_signal": "💙 Emotional support routed",
            "low_confidence": "⚠️ Limited evidence",
            "confidence_blocked": "🔒 Low confidence — answer blocked",
        }
        st.markdown(
            " ".join(f'<span class="safety-flag">{flag_labels.get(f, f)}</span>' for f in flags),
            unsafe_allow_html=True,
        )

    # Confidence
    conf = data.get("confidence_score", 0.0)
    if conf > 0:
        conf_color = "#10b981" if conf > 0.7 else "#f59e0b" if conf > 0.5 else "#ef4444"
        conf_label = "High" if conf > 0.7 else "Moderate" if conf > 0.5 else "Low"
        qtype = data.get("query_type", "")
        qtype_html = f"&nbsp;·&nbsp;routing: <em>{qtype}</em>" if qtype else ""
        st.markdown(
            f'<div style="font-size:0.74rem;color:#6b7280;margin:6px 0 2px;">'
            f'Confidence: <strong>{conf_label}</strong> ({conf:.0%})'
            f'<div class="conf-bar"><div class="conf-fill" style="width:{conf*100:.0f}%;background:{conf_color};"></div></div>'
            f'{qtype_html}</div>',
            unsafe_allow_html=True,
        )

    # Citations
    citations = data.get("citations", [])
    if citations:
        with st.expander(f"📚 {len(citations)} Sources", expanded=False):
            for c in citations:
                url = c.get("url", "#")
                title = c.get("title", "Source")[:100]
                year = c.get("year", "")
                source = c.get("source", "")
                pmid = c.get("pmid", "")
                pmid_html = f' <span style="color:#6366f1;font-size:0.72rem;">PMID:{pmid}</span>' if pmid else ""
                st.markdown(
                    f'<div class="citation-card">'
                    f'<a href="{url}" target="_blank">↗ {title}</a> '
                    f'<span style="color:#6b7280;">({source}, {year})</span>{pmid_html}</div>',
                    unsafe_allow_html=True,
                )

    # Trial cards
    trials = data.get("trial_results", [])
    if trials:
        with st.expander(f"🔬 {len(trials)} Clinical Trials Found", expanded=True):
            for t in trials[:8]:
                interventions = ", ".join(i["name"] for i in t.get("interventions", [])[:3])
                locs = t.get("locations", [])
                loc_str = ", ".join(
                    f"{l.get('city','')}, {l.get('state','')}" for l in locs[:2] if l.get("city")
                ) or "Multiple locations"
                st.markdown(f"""
<div class="trial-card">
<strong>{t.get("title","")[:80]}</strong>
<span class="phase-badge">{t.get("phase","")}</span>
<br><span style="color:#6b7280;font-size:0.78rem;">{t.get("nct_id","")} · {loc_str}</span>
{"<br><span style='color:#9ca3af;'>Intervention: " + interventions + "</span>" if interventions else ""}
<br><a href="{t.get('url','')}" target="_blank" style="color:#6366f1;font-size:0.78rem;">View on ClinicalTrials.gov ↗</a>
</div>""", unsafe_allow_html=True)

    # Emotional resources
    if data.get("emotional_resources"):
        with st.expander("💙 Support Resources", expanded=False):
            for r in data["emotional_resources"]:
                phone = f" · {r['phone']}" if r.get("phone") else ""
                st.markdown(
                    f'<div class="citation-card">'
                    f'<a href="{r["url"]}" target="_blank">↗ {r["name"]}</a>'
                    f'<br><span style="color:#6b7280;">{r["description"]}{phone}</span></div>',
                    unsafe_allow_html=True,
                )


# ── Message history ───────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    render_message(msg)


# ── Chat input & send ─────────────────────────────────────────────────────────
prefill = st.session_state.pop("_prefill", "")
query = st.chat_input(
    placeholder="Ask about treatments, trials, drugs, or anything about GBM..."
)
if prefill:
    query = prefill

if query:
    # Show user bubble immediately
    user_msg = {"role": "user", "content": query}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    # Show active settings confirmation
    st.markdown(
        f'<div style="font-size:0.72rem;color:#6b7280;text-align:right;margin:-4px 0 8px;">'
        f'Mode: <strong>{st.session_state.literacy_mode}</strong> &nbsp;·&nbsp; '
        f'Retrieval: <strong>{st.session_state.retrieval_mode.upper()}</strong></div>',
        unsafe_allow_html=True,
    )

    with st.spinner("🧠 Thinking..."):
        result = api_post("/chat", {
            "query": query,
            "literacy_mode": st.session_state.literacy_mode,
            "session_id": st.session_state.session_id,
            "conversation_history": [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-8:]
                if m["role"] in ("user", "assistant")
            ],
        })

    if "error" in result and "answer" not in result:
        bot_msg = {
            "role": "assistant",
            "content": (
                f"**Connection error:** {result['error']}\n\n"
                "Make sure the API is running:\n"
                "```\nPYTHONPATH=. uvicorn gbm_copilot.api.main:app --reload --port 8000\n```"
            ),
            "data": {},
        }
    else:
        bot_msg = {
            "role": "assistant",
            "content": result.get("answer", "No response received."),
            "data": {
                "citations": result.get("citations", []),
                "confidence_score": result.get("confidence_score", 0.0),
                "safety_flags": result.get("safety_flags", []),
                "trial_results": result.get("trial_results", []),
                "emotional_resources": [],
                "is_blocked": result.get("is_blocked", False),
                "query_type": result.get("query_type", ""),
            },
        }

    st.session_state.messages.append(bot_msg)
    render_message(bot_msg)
    # Rerun to update sidebar stats after the response
    st.rerun()
