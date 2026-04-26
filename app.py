import streamlit as st
import time
import json
import os
from datetime import datetime
from pipeline import RAGPipeline

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ACity RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=Inter:wght@300;400;500;600&display=swap');

:root {
  --bg-base:     #070B14;
  --bg-surface:  #0D1117;
  --bg-elevated: #161B22;
  --bg-hover:    #1C2128;
  --border:      rgba(255,255,255,0.07);
  --border-hi:   rgba(99,179,237,0.3);
  --accent:      #3B82F6;
  --accent-2:    #6366F1;
  --teal:        #14B8A6;
  --green:       #10B981;
  --amber:       #F59E0B;
  --red:         #EF4444;
  --text-1:      #F0F6FC;
  --text-2:      #8B949E;
  --text-3:      #484F58;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
  background-color: var(--bg-base) !important;
  color: var(--text-1) !important;
  font-family: 'Inter', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bg-hover); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-3); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

/* ── Inputs ── */
.stTextInput > div > div > input {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  color: var(--text-1) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 14px !important;
  padding: 14px 20px !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
  border-color: var(--border-hi) !important;
  box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
  outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: var(--text-3) !important; }

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, var(--accent) 0%, #2563EB 100%) !important;
  border: none !important;
  border-radius: 12px !important;
  color: #fff !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  letter-spacing: 0.03em !important;
  padding: 10px 20px !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
  width: 100% !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 20px rgba(59,130,246,0.3) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Expanders ── */
.streamlit-expanderHeader {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text-2) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 12px !important;
  padding: 10px 16px !important;
}
.streamlit-expanderContent {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 10px 10px !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] {
  color: var(--text-2) !important;
  font-size: 11px !important;
}
[data-testid="stMetricValue"] {
  color: var(--accent) !important;
  font-family: 'Syne', sans-serif !important;
  font-size: 22px !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text-1) !important;
}

/* ── Slider ── */
.stSlider > div > div > div { background: var(--accent) !important; }

/* ── Radio ── */
.stRadio > div { gap: 6px !important; }
.stRadio > div > label {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 8px 14px !important;
  cursor: pointer !important;
  transition: all 0.15s !important;
  color: var(--text-2) !important;
  font-size: 13px !important;
}
.stRadio > div > label:hover {
  border-color: var(--border-hi) !important;
  color: var(--text-1) !important;
}

/* ── Animations ── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-green {
  0%, 100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.4); }
  50%       { box-shadow: 0 0 0 4px rgba(16,185,129,0); }
}

.msg-animate { animation: fadeUp 0.25s ease forwards; }
.dot-pulse   { animation: pulse-green 2s infinite; }
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []
if "show_details" not in st.session_state:
    st.session_state.show_details = {}
if "last_submitted" not in st.session_state:
    st.session_state.last_submitted = ""
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# ── Sidebar ─────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Header / logo
        st.markdown("""
        <div style="padding:28px 20px 20px;
                    border-bottom:1px solid rgba(255,255,255,0.07);">
            <div style="display:flex; align-items:center; gap:14px;">
                <div style="width:44px; height:44px; flex-shrink:0;
                            background:linear-gradient(135deg,#3B82F6 0%,#6366F1 100%);
                            border-radius:12px; display:flex; align-items:center;
                            justify-content:center; font-size:22px;
                            box-shadow:0 8px 24px rgba(59,130,246,0.3);">🎓</div>
                <div>
                    <div style="font-family:'Syne',sans-serif; font-weight:800;
                                font-size:16px; color:#F0F6FC;
                                letter-spacing:-0.02em;">ACity RAG</div>
                    <div style="font-family:'DM Mono',monospace; font-size:10px;
                                color:rgba(99,179,237,0.55); letter-spacing:0.1em;
                                margin-top:2px;">ACADEMIC CITY · GHANA</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Status indicator
        if st.session_state.pipeline_ready:
            st.markdown("""
            <div style="margin:0 16px 20px; padding:10px 14px;
                        background:rgba(16,185,129,0.07);
                        border:1px solid rgba(16,185,129,0.18);
                        border-radius:10px; display:flex; align-items:center; gap:10px;">
                <div style="width:8px; height:8px; border-radius:50%;
                            background:#10B981; flex-shrink:0;" class="dot-pulse"></div>
                <div>
                    <div style="font-family:'DM Mono',monospace; font-size:11px;
                                color:#10B981; font-weight:500;">System Online</div>
                    <div style="font-family:'DM Mono',monospace; font-size:10px;
                                color:rgba(255,255,255,0.22); margin-top:1px;">
                        RAG Pipeline v1.0</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="margin:0 16px 20px; padding:10px 14px;
                        background:rgba(245,158,11,0.07);
                        border:1px solid rgba(245,158,11,0.18);
                        border-radius:10px; display:flex; align-items:center; gap:10px;">
                <div style="width:8px; height:8px; border-radius:50%;
                            background:#F59E0B; flex-shrink:0;"></div>
                <div>
                    <div style="font-family:'DM Mono',monospace; font-size:11px;
                                color:#F59E0B; font-weight:500;">Not Initialized</div>
                    <div style="font-family:'DM Mono',monospace; font-size:10px;
                                color:rgba(255,255,255,0.22); margin-top:1px;">
                        RAG Pipeline v1.0</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Init button
        if not st.session_state.pipeline_ready:
            st.markdown("<div style='padding:0 16px 8px;'>", unsafe_allow_html=True)
            if st.button("⚡ Initialize Pipeline", key="init_btn"):
                try:
                    with st.status("Initializing RAG Pipeline…", expanded=True) as init_status:
                        pipeline = RAGPipeline()
                        pipeline.initialize(on_progress=lambda msg: st.write(msg))
                        st.session_state.pipeline = pipeline
                        st.session_state.pipeline_ready = True
                        init_status.update(label="✅ Pipeline Ready!", state="complete")
                    st.rerun()
                except Exception as e:
                    st.error(f"Initialization failed: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Settings ──
        st.markdown("""
        <div style="padding:6px 20px 10px;">
            <div style="font-family:'DM Mono',monospace; font-size:10px; font-weight:500;
                        color:rgba(255,255,255,0.22); letter-spacing:0.13em;
                        text-transform:uppercase;">Settings</div>
        </div>
        """, unsafe_allow_html=True)

        top_k = st.slider("Top-K Chunks", 1, 10, 5, key="top_k",
                          help="Number of document chunks to retrieve")
        prompt_style = st.selectbox(
            "Prompt Template",
            ["Hallucination-Controlled", "Chain-of-Thought", "Basic"],
            key="prompt_style"
        )
        show_chunks = st.toggle("Show retrieved chunks",  value=True,  key="show_chunks")
        show_scores = st.toggle("Show similarity scores", value=True,  key="show_scores")
        show_prompt = st.toggle("Show full prompt",       value=False, key="show_prompt")

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # ── Data sources ──
        st.markdown("""
        <div style="padding:6px 20px 10px;">
            <div style="font-family:'DM Mono',monospace; font-size:10px; font-weight:500;
                        color:rgba(255,255,255,0.22); letter-spacing:0.13em;
                        text-transform:uppercase;">Data Sources</div>
        </div>
        """, unsafe_allow_html=True)

        for icon, name, desc, clr in [
            ("📊", "Ghana Elections CSV", "Election results dataset", "#14B8A6"),
            ("📄", "2025 Budget PDF",     "Budget Statement & Policy", "#3B82F6"),
        ]:
            st.markdown(f"""
            <div style="margin:0 16px 8px; padding:10px 14px;
                        background:rgba(255,255,255,0.025);
                        border:1px solid rgba(255,255,255,0.06);
                        border-radius:10px; display:flex; align-items:center; gap:10px;">
                <div style="width:32px; height:32px; border-radius:8px; flex-shrink:0;
                            background:{clr}18; display:flex; align-items:center;
                            justify-content:center; font-size:16px;">{icon}</div>
                <div>
                    <div style="font-family:'Inter',sans-serif; font-size:12px;
                                font-weight:500; color:#F0F6FC;">{name}</div>
                    <div style="font-family:'DM Mono',monospace; font-size:10px;
                                color:rgba(255,255,255,0.28); margin-top:2px;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # ── Session stats ──
        st.markdown("""
        <div style="padding:6px 20px 10px;">
            <div style="font-family:'DM Mono',monospace; font-size:10px; font-weight:500;
                        color:rgba(255,255,255,0.22); letter-spacing:0.13em;
                        text-transform:uppercase;">Session</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", st.session_state.query_count)
        with col2:
            fb  = st.session_state.feedback_log
            pos = sum(1 for f in fb if f.get("rating") == "👍") if fb else 0
            st.metric("👍 Helpful", pos)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        st.markdown("<div style='padding:0 16px;'>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat", key="clear_btn"):
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.session_state.last_submitted = ""
            st.session_state.input_key += 1
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # Footer
        st.markdown("""
        <div style="position:absolute; bottom:20px; left:0; right:0;
                    text-align:center; padding:0 16px;">
            <div style="font-family:'DM Mono',monospace; font-size:10px;
                        color:rgba(255,255,255,0.14); line-height:1.9;">
                Academic City · RAG Assignment<br>
                <span style="color:rgba(99,179,237,0.22);">
                    No LangChain · No LlamaIndex</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Render chat message ──────────────────────────────────────────────────────────
def render_message(msg, idx):
    role    = msg["role"]
    content = msg["content"]
    ts      = msg.get("timestamp", "")

    if role == "user":
        # User bubble — content is plain text, safe to use st.chat_message
        with st.chat_message("user"):
            st.write(content)

    else:
        answer      = content.get("answer", "")  if isinstance(content, dict) else str(content)
        chunks      = content.get("chunks",  []) if isinstance(content, dict) else []
        scores      = content.get("scores",  []) if isinstance(content, dict) else []
        prompt      = content.get("prompt",  "") if isinstance(content, dict) else ""
        source_tags = content.get("sources", []) if isinstance(content, dict) else []
        latency     = content.get("latency", "") if isinstance(content, dict) else ""

        with st.chat_message("assistant"):
            # Header row: label + latency
            if latency:
                st.caption(f"🎓 ACity Assistant · ⚡ {latency}")
            else:
                st.caption("🎓 ACity Assistant")

            # Source badges as plain text pills
            if source_tags:
                seen = []
                for s in source_tags:
                    short = s.split("(")[0].strip()   # "2025 Budget PDF (p.63)" → "2025 Budget PDF"
                    if short not in seen:
                        seen.append(short)
                st.markdown(" · ".join(f"`{s}`" for s in seen))

            # Answer — always use st.write so special chars never break rendering
            st.write(answer)

            # Timestamp
            st.caption(ts)

            # Retrieved chunks panel
            if st.session_state.get("show_chunks") and chunks:
                with st.expander(f"📄 Retrieved Chunks ({len(chunks)})", expanded=False):
                    for i, (chunk, score) in enumerate(zip(chunks, scores)):
                        rel_label = "🟢 High" if score > 0.75 else ("🟡 Medium" if score > 0.5 else "🔴 Low")
                        st.markdown(f"**Chunk {i+1}** — {rel_label} · score: `{score:.4f}`")
                        st.progress(min(score, 1.0))
                        st.caption(chunk[:400] + ("…" if len(chunk) > 400 else ""))
                        st.divider()

            if st.session_state.get("show_prompt") and prompt:
                with st.expander("📨 Full Prompt Sent to LLM", expanded=False):
                    st.code(prompt, language="markdown")

            # Feedback buttons
            fb_key = f"fb_{idx}"
            if fb_key not in st.session_state:
                st.session_state[fb_key] = None

            col1, col2, col3 = st.columns([1, 1, 10])
            with col1:
                if st.button("👍", key=f"up_{idx}", help="Helpful"):
                    st.session_state[fb_key] = "👍"
                    st.session_state.feedback_log.append({
                        "idx": idx, "rating": "👍",
                        "answer_preview": answer[:80],
                        "timestamp": datetime.now().isoformat()
                    })
                    _save_feedback()
            with col2:
                if st.button("👎", key=f"dn_{idx}", help="Not helpful"):
                    st.session_state[fb_key] = "👎"
                    st.session_state.feedback_log.append({
                        "idx": idx, "rating": "👎",
                        "answer_preview": answer[:80],
                        "timestamp": datetime.now().isoformat()
                    })
                    _save_feedback()

            if st.session_state[fb_key]:
                st.caption(f"Feedback recorded {st.session_state[fb_key]}")


def _save_feedback():
    try:
        os.makedirs("logs", exist_ok=True)
        with open("logs/feedback.json", "w") as f:
            json.dump(st.session_state.feedback_log, f, indent=2)
    except Exception:
        pass


# ── Welcome / hero screen ────────────────────────────────────────────────────────
def render_welcome():
    # Single self-contained HTML block — no open/close tags split across calls
    st.markdown("""
    <div style="display:flex; flex-direction:column; align-items:center;
                padding:64px 40px 8px; text-align:center;">
        <div style="width:88px; height:88px;
                    background:linear-gradient(135deg,#3B82F6 0%,#6366F1 100%);
                    border-radius:24px; display:flex; align-items:center;
                    justify-content:center; font-size:44px; margin-bottom:28px;
                    box-shadow:0 16px 48px rgba(59,130,246,0.35),
                               0 0 0 1px rgba(99,179,237,0.12);">🎓</div>
        <h1 style="font-family:'Syne',sans-serif; font-weight:800; font-size:34px;
                   color:#F0F6FC; letter-spacing:-0.03em; margin:0 0 10px;">
            ACity RAG Assistant
        </h1>
        <p style="font-family:'DM Mono',monospace; font-size:11px;
                  color:rgba(99,179,237,0.55); letter-spacing:0.14em; margin:0 0 22px;">
            RETRIEVAL-AUGMENTED GENERATION · ACADEMIC CITY GHANA
        </p>
        <p style="font-family:'Inter',sans-serif; font-size:14px;
                  color:rgba(240,246,252,0.48); max-width:460px;
                  line-height:1.8; margin:0 0 28px;">
            Ask anything about
            <strong style="color:#60A5FA; font-weight:600;">Ghana's 2025 Budget</strong>
            or
            <strong style="color:#2DD4BF; font-weight:600;">Ghana Election Results</strong>.
            Every answer is grounded in the source documents — no hallucinations.
        </p>
        <div style="display:flex; gap:10px; margin-bottom:36px;
                    flex-wrap:wrap; justify-content:center;">
            <span style="padding:6px 16px; background:rgba(59,130,246,0.1);
                        border:1px solid rgba(59,130,246,0.2); border-radius:20px;
                        font-family:'DM Mono',monospace; font-size:11px;
                        color:rgba(99,179,237,0.75);">📑 2 Data Sources</span>
            <span style="padding:6px 16px; background:rgba(99,102,241,0.1);
                        border:1px solid rgba(99,102,241,0.2); border-radius:20px;
                        font-family:'DM Mono',monospace; font-size:11px;
                        color:rgba(167,139,250,0.75);">🔍 Hybrid Vector Search</span>
            <span style="padding:6px 16px; background:rgba(20,184,166,0.1);
                        border:1px solid rgba(20,184,166,0.2); border-radius:20px;
                        font-family:'DM Mono',monospace; font-size:11px;
                        color:rgba(45,212,191,0.75);">✓ Citation Grounded</span>
        </div>
        <p style="font-family:'DM Mono',monospace; font-size:10px;
                  color:rgba(255,255,255,0.22); letter-spacing:0.14em;
                  text-transform:uppercase; margin-bottom:16px;">Try asking…</p>
    </div>
    """, unsafe_allow_html=True)

    suggestions = [
        ("💰", "What is Ghana's inflation target for 2025?"),
        ("🗳️", "Who won the presidential election in Ashanti Region?"),
        ("📈", "What is the GDP growth projection in the 2025 budget?"),
        ("🏆", "Which party won the most parliamentary seats?"),
        ("💵", "What are the key revenue measures in the 2025 budget?"),
        ("📊", "What was the voter turnout in the Greater Accra region?"),
    ]

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    for i, (icon, q) in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(f"{icon}  {q}", key=f"sug_{i}"):
                st.session_state["pending_query"] = q

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    # Top bar
    st.markdown("""
    <div style="background:rgba(7,11,20,0.92); backdrop-filter:blur(24px);
                -webkit-backdrop-filter:blur(24px);
                border-bottom:1px solid rgba(255,255,255,0.06);
                padding:14px 32px; display:flex; align-items:center;
                justify-content:space-between; position:sticky; top:0; z-index:100;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:7px; height:7px; border-radius:50%;
                        background:#3B82F6;
                        box-shadow:0 0 8px rgba(59,130,246,0.7);"></div>
            <span style="font-family:'Syne',sans-serif; font-weight:700;
                         font-size:15px; color:#F0F6FC;
                         letter-spacing:-0.01em;">Chat</span>
        </div>
        <div style="font-family:'DM Mono',monospace; font-size:10px;
                    color:rgba(255,255,255,0.22); letter-spacing:0.09em;">
            GHANA ELECTIONS · 2025 BUDGET · RAG-POWERED
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Message area
    if not st.session_state.messages:
        render_welcome()
    else:
        with st.container():
            st.markdown("<div style='padding:20px 32px;'>", unsafe_allow_html=True)
            for i, msg in enumerate(st.session_state.messages):
                render_message(msg, i)
            st.markdown("</div>", unsafe_allow_html=True)

    # Input bar
    st.markdown("""
    <div style="position:sticky; bottom:0;
                background:rgba(7,11,20,0.96); backdrop-filter:blur(24px);
                -webkit-backdrop-filter:blur(24px);
                border-top:1px solid rgba(255,255,255,0.06);
                padding:16px 32px 20px;">
    """, unsafe_allow_html=True)

    col_in, col_btn = st.columns([7, 1])
    with col_in:
        pending = st.session_state.pop("pending_query", "")
        query = st.text_input(
            "", placeholder="Ask about Ghana Elections or the 2025 Budget…",
            value=pending, key=f"query_input_{st.session_state.input_key}",
            label_visibility="collapsed"
        )
    with col_btn:
        send = st.button("Send ➤", key="send_btn")

    st.markdown("</div>", unsafe_allow_html=True)

    # Only fire when Send button is clicked — never on bare reruns
    if send and query.strip():
        if not st.session_state.pipeline_ready:
            st.warning("⚡ Please initialize the pipeline first using the sidebar button.")
            return

        user_query = query.strip()
        # Bump the key so the text input resets to empty after rerun
        st.session_state.input_key += 1
        st.session_state.last_submitted = user_query

        st.session_state.messages.append({
            "role": "user",
            "content": user_query,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        st.session_state.query_count += 1

        with st.spinner("🔍 Retrieving relevant context…"):
            try:
                t0 = time.time()
                result = st.session_state.pipeline.query(
                    query=user_query,
                    top_k=st.session_state.get("top_k", 5),
                    prompt_style=st.session_state.get(
                        "prompt_style", "Hallucination-Controlled")
                )
                latency = f"{time.time() - t0:.2f}s"

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {
                        "answer":  result["answer"],
                        "chunks":  result["chunks"],
                        "scores":  result["scores"],
                        "prompt":  result["prompt"],
                        "sources": result["sources"],
                        "latency": latency,
                        "logs":    result.get("logs", [])
                    },
                    "timestamp": datetime.now().strftime("%H:%M")
                })

            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {
                        "answer": f"⚠️ Error processing your query: {str(e)}",
                        "chunks": [], "scores": [], "prompt": "", "sources": []
                    },
                    "timestamp": datetime.now().strftime("%H:%M")
                })

        st.rerun()


if __name__ == "__main__":
    main()