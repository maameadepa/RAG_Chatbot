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
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Base reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background-color: #0A0E1A !important;
    color: #E8EAF0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1220 0%, #080C17 100%) !important;
    border-right: 1px solid rgba(99, 179, 237, 0.12) !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

/* ── Main content area ── */
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0A0E1A; }
::-webkit-scrollbar-thumb { background: #2D3A5C; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #4A6FA5; }

/* ── Input override ── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    border-radius: 12px !important;
    color: #E8EAF0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    padding: 14px 18px !important;
    transition: all 0.2s ease !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(99,179,237,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.08) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: rgba(232,234,240,0.3) !important; }

/* ── Button override ── */
.stButton > button {
    background: linear-gradient(135deg, #1E6FD9 0%, #0F4FA8 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.04em !important;
    padding: 10px 24px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2A85F5 0%, #1A5FC8 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(30,111,217,0.35) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(99,179,237,0.1) !important;
    border-radius: 10px !important;
    color: rgba(232,234,240,0.7) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    padding: 10px 16px !important;
}
.streamlit-expanderContent {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(99,179,237,0.08) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(99,179,237,0.1) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] { color: rgba(232,234,240,0.5) !important; font-size: 11px !important; }
[data-testid="stMetricValue"] { color: #63B3ED !important; font-family: 'Syne', sans-serif !important; font-size: 22px !important; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    border-radius: 10px !important;
    color: #E8EAF0 !important;
}

/* ── Slider ── */
.stSlider > div > div > div { background: #1E6FD9 !important; }

/* ── Radio ── */
.stRadio > div { gap: 8px !important; }
.stRadio > div > label {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(99,179,237,0.15) !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}
.stRadio > div > label:hover { border-color: rgba(99,179,237,0.4) !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ──────────────────────────────────────────────────────────
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

# ── Helper: render sidebar ──────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Logo / header
        st.markdown("""
        <div style="padding: 32px 24px 24px; border-bottom: 1px solid rgba(99,179,237,0.1);">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
                <div style="width:40px; height:40px; background:linear-gradient(135deg,#1E6FD9,#0F4FA8);
                            border-radius:10px; display:flex; align-items:center; justify-content:center;
                            font-size:20px; flex-shrink:0;">🎓</div>
                <div>
                    <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:17px;
                                color:#E8EAF0; letter-spacing:-0.02em;">ACity RAG</div>
                    <div style="font-family:'DM Mono',monospace; font-size:10px;
                                color:rgba(99,179,237,0.7); letter-spacing:0.08em;">ACADEMIC CITY · GHANA</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # Status indicator
        status_color = "#10B981" if st.session_state.pipeline_ready else "#F59E0B"
        status_text  = "System Online" if st.session_state.pipeline_ready else "Not Initialized"
        status_dot   = "●"
        st.markdown(f"""
        <div style="margin: 0 16px 24px; padding:12px 16px;
                    background:rgba(255,255,255,0.03);
                    border:1px solid rgba(99,179,237,0.1); border-radius:10px;
                    display:flex; align-items:center; gap:10px;">
            <span style="color:{status_color}; font-size:18px; line-height:1;">{status_dot}</span>
            <div>
                <div style="font-family:'DM Mono',monospace; font-size:11px;
                            color:{status_color};">{status_text}</div>
                <div style="font-family:'DM Mono',monospace; font-size:10px;
                            color:rgba(232,234,240,0.3);">RAG Pipeline v1.0</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Init button
        if not st.session_state.pipeline_ready:
            st.markdown("<div style='padding: 0 16px;'>", unsafe_allow_html=True)
            if st.button("⚡ Initialize Pipeline", key="init_btn"):
                with st.spinner("Loading documents & building index…"):
                    try:
                        pipeline = RAGPipeline()
                        pipeline.initialize()
                        st.session_state.pipeline = pipeline
                        st.session_state.pipeline_ready = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Settings
        st.markdown("""
        <div style="padding: 0 16px 8px;">
            <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:11px;
                        color:rgba(232,234,240,0.4); letter-spacing:0.1em;
                        text-transform:uppercase; margin-bottom:12px;">Settings</div>
        </div>
        """, unsafe_allow_html=True)

        top_k = st.slider("Top-K Chunks", 1, 10, 5, key="top_k",
                          help="Number of document chunks to retrieve")
        prompt_style = st.selectbox(
            "Prompt Template",
            ["Hallucination-Controlled", "Chain-of-Thought", "Basic"],
            key="prompt_style"
        )
        show_chunks   = st.toggle("Show retrieved chunks",   value=True,  key="show_chunks")
        show_scores   = st.toggle("Show similarity scores",  value=True,  key="show_scores")
        show_prompt   = st.toggle("Show full prompt",        value=False, key="show_prompt")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Data sources
        st.markdown("""
        <div style="padding: 0 16px 8px;">
            <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:11px;
                        color:rgba(232,234,240,0.4); letter-spacing:0.1em;
                        text-transform:uppercase; margin-bottom:12px;">Data Sources</div>
        </div>
        """, unsafe_allow_html=True)

        sources = [
            ("📊", "Ghana Elections CSV", "Election results dataset"),
            ("📄", "2025 Budget PDF",     "Budget Statement & Economic Policy"),
        ]
        for icon, name, desc in sources:
            st.markdown(f"""
            <div style="margin: 0 16px 8px; padding:10px 14px;
                        background:rgba(255,255,255,0.02);
                        border:1px solid rgba(99,179,237,0.08); border-radius:8px;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-size:16px;">{icon}</span>
                    <div>
                        <div style="font-family:'DM Sans',sans-serif; font-weight:500;
                                    font-size:12px; color:#E8EAF0;">{name}</div>
                        <div style="font-family:'DM Mono',monospace; font-size:10px;
                                    color:rgba(232,234,240,0.35);">{desc}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Stats
        st.markdown("""
        <div style="padding: 0 16px 8px;">
            <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:11px;
                        color:rgba(232,234,240,0.4); letter-spacing:0.1em;
                        text-transform:uppercase; margin-bottom:12px;">Session Stats</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", st.session_state.query_count)
        with col2:
            fb = st.session_state.feedback_log
            pos = sum(1 for f in fb if f.get("rating") == "👍") if fb else 0
            st.metric("👍 Helpful", pos)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Clear chat
        st.markdown("<div style='padding: 0 16px;'>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat", key="clear_btn"):
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # Footer
        st.markdown("""
        <div style="position:absolute; bottom:20px; left:0; right:0;
                    padding: 0 16px; text-align:center;">
            <div style="font-family:'DM Mono',monospace; font-size:10px;
                        color:rgba(232,234,240,0.2);">
                Built for Academic City · RAG Assignment<br/>
                <span style="color:rgba(99,179,237,0.3);">No LangChain · No LlamaIndex</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Helper: render chat message ─────────────────────────────────────────────────
def render_message(msg, idx):
    role    = msg["role"]
    content = msg["content"]
    ts      = msg.get("timestamp", "")

    if role == "user":
        st.markdown(f"""
        <div style="display:flex; justify-content:flex-end; margin:16px 0 8px;">
            <div style="max-width:70%; background:linear-gradient(135deg,#1E6FD9,#0F4FA8);
                        border-radius:18px 18px 4px 18px; padding:14px 18px;
                        box-shadow: 0 4px 16px rgba(30,111,217,0.25);">
                <div style="font-family:'DM Sans',sans-serif; font-size:14px;
                            color:#fff; line-height:1.6;">{content}</div>
                <div style="font-family:'DM Mono',monospace; font-size:10px;
                            color:rgba(255,255,255,0.4); margin-top:6px;
                            text-align:right;">{ts}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:  # assistant
        answer = content.get("answer", "")
        chunks = content.get("chunks", [])
        scores = content.get("scores", [])
        prompt = content.get("prompt", "")
        source_tags = content.get("sources", [])
        latency = content.get("latency", "")

        # Source badges
        badge_html = ""
        seen = set()
        for s in source_tags:
            if s not in seen:
                seen.add(s)
                color = "#1E6FD9" if "Budget" in s else "#0D9488"
                badge_html += f"""<span style="background:{color}22; border:1px solid {color}55;
                    color:{color}; font-family:'DM Mono',monospace; font-size:10px;
                    border-radius:6px; padding:2px 8px; margin-right:6px;">{s}</span>"""

        st.markdown(f"""
        <div style="margin:8px 0 4px;">
            <div style="display:flex; align-items:flex-start; gap:12px;">
                <div style="width:36px; height:36px; background:linear-gradient(135deg,#0D1A35,#1A2D5A);
                            border:1px solid rgba(99,179,237,0.2); border-radius:10px;
                            display:flex; align-items:center; justify-content:center;
                            font-size:18px; flex-shrink:0;">🎓</div>
                <div style="flex:1; background:rgba(255,255,255,0.03);
                            border:1px solid rgba(99,179,237,0.1); border-radius:4px 18px 18px 18px;
                            padding:16px 20px;">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                        <span style="font-family:'Syne',sans-serif; font-weight:700; font-size:12px;
                                     color:rgba(99,179,237,0.8); letter-spacing:0.06em;">ACity Assistant</span>
                        {f'<span style="font-family:DM Mono,monospace; font-size:10px; color:rgba(232,234,240,0.25);">⚡ {latency}</span>' if latency else ''}
                    </div>
                    {f'<div style="margin-bottom:10px;">{badge_html}</div>' if badge_html else ''}
                    <div style="font-family:'DM Sans',sans-serif; font-size:14px;
                                color:#E8EAF0; line-height:1.75; white-space:pre-wrap;">{answer}</div>
                    <div style="font-family:'DM Mono',monospace; font-size:10px;
                                color:rgba(232,234,240,0.25); margin-top:10px;">{ts}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Detail panels
        if st.session_state.get("show_chunks") and chunks:
            with st.expander(f"📄 Retrieved Chunks ({len(chunks)})", expanded=False):
                for i, (chunk, score) in enumerate(zip(chunks, scores)):
                    relevance = "🟢 High" if score > 0.75 else ("🟡 Medium" if score > 0.5 else "🔴 Low")
                    bar_w = int(score * 100)
                    bar_color = "#10B981" if score > 0.75 else ("#F59E0B" if score > 0.5 else "#EF4444")
                    st.markdown(f"""
                    <div style="margin-bottom:12px; padding:14px;
                                background:rgba(255,255,255,0.02);
                                border:1px solid rgba(99,179,237,0.08); border-radius:10px;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                            <span style="font-family:'DM Mono',monospace; font-size:11px;
                                         color:rgba(99,179,237,0.7);">Chunk {i+1}</span>
                            <span style="font-family:'DM Mono',monospace; font-size:11px;
                                         color:rgba(232,234,240,0.5);">{relevance} · {score:.4f}</span>
                        </div>
                        <div style="height:3px; background:rgba(255,255,255,0.05);
                                    border-radius:2px; margin-bottom:10px;">
                            <div style="height:100%; width:{bar_w}%;
                                        background:{bar_color}; border-radius:2px;
                                        transition:width 0.5s ease;"></div>
                        </div>
                        <div style="font-family:'DM Sans',sans-serif; font-size:12px;
                                    color:rgba(232,234,240,0.7); line-height:1.6;">{chunk[:400]}{'…' if len(chunk) > 400 else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)

        if st.session_state.get("show_prompt") and prompt:
            with st.expander("📨 Full Prompt Sent to LLM", expanded=False):
                st.code(prompt, language="markdown")

        # Feedback
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
            st.markdown(f"""
            <div style="font-family:'DM Mono',monospace; font-size:11px;
                        color:rgba(232,234,240,0.4); margin-top:4px; padding-left:4px;">
                Feedback recorded {st.session_state[fb_key]}
            </div>
            """, unsafe_allow_html=True)


def _save_feedback():
    try:
        os.makedirs("logs", exist_ok=True)
        with open("logs/feedback.json", "w") as f:
            json.dump(st.session_state.feedback_log, f, indent=2)
    except Exception:
        pass


# ── Helper: render hero / welcome screen ───────────────────────────────────────
def render_welcome():
    # Hero section
    st.markdown("""
    <div style="display:flex; flex-direction:column; align-items:center;
                justify-content:center; padding:60px 40px 32px; text-align:center;">
        <div style="width:80px; height:80px;
                    background:linear-gradient(135deg,#1E6FD9 0%,#0A4A9F 100%);
                    border-radius:20px; display:flex; align-items:center;
                    justify-content:center; font-size:40px; margin-bottom:28px;
                    box-shadow:0 20px 60px rgba(30,111,217,0.4);">🎓</div>
        <h1 style="font-family:'Syne',sans-serif; font-weight:800; font-size:36px;
                   color:#E8EAF0; letter-spacing:-0.03em; margin:0 0 8px;">
            ACity RAG Assistant
        </h1>
        <p style="font-family:'DM Mono',monospace; font-size:12px;
                  color:rgba(99,179,237,0.7); letter-spacing:0.12em; margin:0 0 24px;">
            RETRIEVAL-AUGMENTED GENERATION · ACADEMIC CITY GHANA
        </p>
        <p style="font-family:'DM Sans',sans-serif; font-size:15px;
                  color:rgba(232,234,240,0.55); max-width:480px; line-height:1.7; margin:0 0 32px;">
            Ask anything about <strong style="color:#63B3ED;">Ghana's 2025 Budget</strong>
            or <strong style="color:#0D9488;">Ghana Election Results</strong>.
            Every answer is grounded in the source documents no hallucinations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Suggestion label
    st.markdown("""
    <p style="text-align:center; font-family:'Syne',sans-serif; font-weight:700;
              font-size:11px; color:rgba(232,234,240,0.35); letter-spacing:0.12em;
              text-transform:uppercase; margin-bottom:16px;">
        Try asking…
    </p>
    """, unsafe_allow_html=True)

    # Suggestion buttons using native Streamlit columns
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
            if st.button(f"{icon} {q}", key=f"sug_{i}"):
                st.session_state["pending_query"] = q

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)


# ── Main layout ─────────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    # ── Top bar ──
    st.markdown("""
    <div style="background:rgba(10,14,26,0.95); backdrop-filter:blur(20px);
                border-bottom:1px solid rgba(99,179,237,0.08);
                padding:16px 32px; display:flex; align-items:center;
                justify-content:space-between; position:sticky; top:0; z-index:100;">
        <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:16px;
                    color:#E8EAF0; letter-spacing:-0.01em;">
            💬 Chat Interface
        </div>
        <div style="font-family:'DM Mono',monospace; font-size:11px;
                    color:rgba(99,179,237,0.6);">
            Ghana Elections · 2025 Budget · RAG-Powered
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Message area ──
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            render_welcome()
        else:
            st.markdown("<div style='padding:24px 32px;'>", unsafe_allow_html=True)
            for i, msg in enumerate(st.session_state.messages):
                render_message(msg, i)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Input bar ──
    st.markdown("""
    <div style="position:sticky; bottom:0; background:rgba(10,14,26,0.98);
                backdrop-filter:blur(20px); border-top:1px solid rgba(99,179,237,0.08);
                padding:20px 32px;">
    """, unsafe_allow_html=True)

    col_in, col_btn = st.columns([6, 1])
    with col_in:
        pending = st.session_state.pop("pending_query", "")
        query = st.text_input(
            "", placeholder="Ask about Ghana Elections or the 2025 Budget…",
            value=pending, key="query_input", label_visibility="collapsed"
        )
    with col_btn:
        send = st.button("Send ➤", key="send_btn")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Handle query ──
    if (send or query) and query.strip():
        if not st.session_state.pipeline_ready:
            st.warning("⚡ Please initialize the pipeline first using the sidebar button.")
            return

        user_msg = {
            "role": "user",
            "content": query.strip(),
            "timestamp": datetime.now().strftime("%H:%M")
        }
        st.session_state.messages.append(user_msg)
        st.session_state.query_count += 1

        with st.spinner("🔍 Retrieving relevant context…"):
            try:
                t0 = time.time()
                result = st.session_state.pipeline.query(
                    query=query.strip(),
                    top_k=st.session_state.get("top_k", 5),
                    prompt_style=st.session_state.get("prompt_style", "Hallucination-Controlled")
                )
                latency = f"{time.time() - t0:.2f}s"

                assistant_msg = {
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
                }
                st.session_state.messages.append(assistant_msg)

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
