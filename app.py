import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

# Load environmental configs directly from your root secure box
load_dotenv()

# Import our separated modular local pipeline engine
from rag_pipeline import LocalFAISSRAGEngine

# FORCE WIDESCREEN: Crucial for making the side-by-side dashboard look spacious and premium
st.set_page_config(page_title="FAISS Local CPU Storage Sandbox", layout="wide", page_icon="📚")

# Caching the engine ensures your app loads instantly instead of recalculating on every click!
@st.cache_resource
def load_cached_rag_engine():
    return LocalFAISSRAGEngine()

if "local_rag" not in st.session_state:
    st.session_state.local_rag = load_cached_rag_engine()

# Inject Premium Custom CSS to build the dark-mode hover cards exactly like NotebookLM
st.markdown("""
    <style>
    .project-card {
        background-color: #1E1F24;
        border: 1px solid #4A4B50;
        border-radius: 12px;
        padding: 20px;
        height: 180px;
        margin-bottom: 15px;
        transition: transform 0.2s, border-color 0.2s;
    }
    .project-card:hover {
        transform: translateY(-4px);
        border-color: #FF4B4B;
        box-shadow: 0px 4px 15px rgba(255, 75, 75, 0.15);
    }
    .card-title {
        font-size: 16px;
        font-weight: bold;
        color: #FFFFFF;
        margin-top: 10px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .card-meta {
        font-size: 11px;
        color: #A3A3A3;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Master Registry to dynamically track all files stored in your local index
all_chunks = st.session_state.local_rag.get_all_stored_chunks()
unique_files_in_db = sorted(list(set([chunk["source"] for chunk in all_chunks if "source" in chunk])))

# Synchronize your UI session state memory with the actual physical files found on your disk
if "notebooks" not in st.session_state:
    st.session_state.notebooks = []
    
# Rebuild visual workspace cards based on actual files detected in the database index
st.session_state.notebooks = []
for filename in unique_files_in_db:
    # Calculate how many chunks belong to this specific file layout
    file_chunks_count = sum(1 for chunk in all_chunks if chunk["source"] == filename)
    st.session_state.notebooks.append({
        "title": filename,
        "date": "Active Storage",
        "sources": f"{file_chunks_count} chunks cached",
        "icon": "📄"
    })

# Set fallback default if no notebook is active or chosen yet
if not st.session_state.notebooks:
    if "active_notebook" not in st.session_state:
        st.session_state.active_notebook = "No Active Document"
elif "active_notebook" not in st.session_state or st.session_state.active_notebook not in unique_files_in_db:
    st.session_state.active_notebook = st.session_state.notebooks[0]["title"]

# Secure fallbacks for text streaming pass
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_ENDPOINT = str(os.getenv("AZURE_OPENAI_ENDPOINT", "")).strip()
MODEL_DEPLOYMENT = str(os.getenv("CHAT_DEPLOYMENT_NAME", "")).strip()

# =========================================================================
# 🏛️ MASTER NOTEBOOKLM SIDE-BY-SIDE SPLIT LAYOUT (65% Workspace | 35% Chat Sidebar)
# =========================================================================
workspace_layout_col, chat_sidebar_col = st.columns([0.65, 0.35], gap="large")

# -------------------------------------------------------------------------
# LEFT REGION: THE WORKSPACE DASHBOARD (Notebook Cards & Uploaders)
# -------------------------------------------------------------------------
with workspace_layout_col:
    st.title("📚 NotebookLM Workspace Hub")
    st.write(f"🎯 **Active Workspace Shelf Memory:** `{st.session_state.active_notebook}`")
    st.markdown("---")
    
    dashboard_tab, database_inspector_tab = st.tabs(["🗂️ My Recent Notebooks", "🔍 Backstage Vector Store Inspector"])
    
    with dashboard_tab:
        if not st.session_state.notebooks:
            st.info("💡 Your library shelves are completely empty! Expand the window panel below to upload your first PDF book file.")
        else:
            cards_per_row = 3
            for i in range(0, len(st.session_state.notebooks), cards_per_row):
                row_items = st.session_state.notebooks[i:i+cards_per_row]
                grid_columns = st.columns(cards_per_row)
                
                for idx, item in enumerate(row_items):
                    with grid_columns[idx]:
                        st.markdown(f"""
                            <div class="project-card">
                                <div style="font-size: 26px;">{item['icon']}</div>
                                <div class="card-title">{item['title']}</div>
                                <div class="card-meta">📁 {item['sources']}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button("🔌 Open Notebook", key=f"select_{i+idx}", use_container_width=True):
                            st.session_state.active_notebook = item['title']
                            st.rerun()
                        
        # INGESTION HUB ACCORDION BAR
        st.markdown("---")
        with st.expander("📥 Add New Document / Source to Workspace", expanded=True if not st.session_state.notebooks else False):
            uploaded_pdf_asset = st.file_uploader("Drop your target manual or handbook PDF file here", type=["pdf"])
            
            if st.button("🚀 Ingest Document Locally (Build Index)", type="primary", use_container_width=True):
                if not uploaded_pdf_asset:
                    st.error("❌ Action Blocked: Drop a PDF document asset first!")
                else:
                    with st.spinner("PyMuPDF extracting text strings and compiling local FAISS matrix mapping..."):
                        try:
                            temp_local_path = os.path.join(".", uploaded_pdf_asset.name)
                            with open(temp_local_path, "wb") as f:
                                f.write(uploaded_pdf_asset.getbuffer())
                                
                            total_chunks = st.session_state.local_rag.ingest_local_document(temp_local_path)
                            
                            if os.path.exists(temp_local_path):
                                os.remove(temp_local_path)
                                
                            st.session_state.active_notebook = uploaded_pdf_asset.name
                            st.success(f"🎯 Local Ledger Sync Stable! Compiled {total_chunks} blocks inside local directory file matrix index.")
                            st.rerun()
                        except Exception as fatal_err:
                            st.error(f"🔴 Local Pipeline Crash: {str(fatal_err)}")

    with database_inspector_tab:
        st.markdown("### 🗄️ Master Database Inventory & Global Index Map")
        if not all_chunks:
            st.info("💡 Storage Ledger Clear: Upload a document file to audit its internal layout array.")
        else:
            st.metric(label="📊 Total Compiled Vector Chunks in Active Index (All Files)", value=len(all_chunks))
            st.markdown("---")
            for chunk_item in all_chunks:
                with st.expander(f"📦 Chunk #{chunk_item['chunk_index']} | 📄 {chunk_item['source']} | Page: {chunk_item['page']}"):
                    side_col1, side_col2 = st.columns([4, 3])
                    with side_col1:
                        st.markdown("**📄 Segment Text:**")
                        st.info(chunk_item['chunk_text'])
                    with side_col2:
                        st.markdown(f"**📐 Total Geometry Size:** `{len(chunk_item['full_vector'])} Dimensions`")
                        st.markdown("**📍 Absolute Vector Space Location Matrix Array (Sample):**")
                        st.json(chunk_item['full_vector'][:5] + ["... remaining axes truncated"])

# -------------------------------------------------------------------------
# RIGHT REGION: THE FIXED CONVERSATIONAL INTERACTIVE SIDEBAR PANEL
# -------------------------------------------------------------------------
with chat_sidebar_col:
    st.markdown("### 💬 Workspace Assistant Chat")
    st.caption(f"Sync connected strictly to local storage matrices...")
    
    if "audit_chat_history" not in st.session_state:
        st.session_state.audit_chat_history = [
            {"role": "assistant", "content": "The cloud layers are gone. Ask me any question. If an index file is loaded, I will query it in real-time locally!"}
        ]
        
    if st.button("🔄 Clear Timeline Logs", use_container_width=True):
        st.session_state.audit_chat_history = [
            {"role": "assistant", "content": "The cloud layers are gone. Ask me any question. If an index file is loaded, I will query it in real-time locally!"}
        ]
        st.rerun()

    chat_box_viewport = st.container(height=500, border=True)
    
    with chat_box_viewport:
        for message in st.session_state.audit_chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if user_query := st.chat_input("Ask any question about your documents freely..."):
        if st.session_state.active_notebook == "No Active Document":
            st.error("❌ Action Blocked: You must upload and open a notebook document card before querying the assistant!")
        else:
            st.session_state.audit_chat_history.append({"role": "user", "content": user_query})
            
            with chat_box_viewport:
                with st.chat_message("user"):
                    st.markdown(user_query)

                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    accumulated_text = ""
                    
                    clean_query = user_query.strip().lower().rstrip(".?!")
                    raw_data_dump = []
                    
                    if clean_query in ["hi", "hello", "hey", "greetings", "sup"]:
                        pdf_source_context = "No relevant document excerpts found (User sent a standard conversational greeting)."
                    else:
                        # 🎯 THE MASTER UPGRADE FIX: Call our new partitioned workspace filter search!
                        raw_data_dump = st.session_state.local_rag.search_workspace_vector_space(
                            query=user_query,
                            workspace_name=st.session_state.active_notebook,
                            top_k=5
                        )
                        context_blocks = [f"[Page {item['page']}] {item['chunk_text']}" for item in raw_data_dump]
                        pdf_source_context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant document excerpts found."

                    unbounded_system_prompt = f"""You are a senior executive AI consultant and an elite domain expert.
Below are some highly relevant paragraph blocks extracted via real-time local FAISS spatial matrix lookups from the user's uploaded manual files.

--- RETRIEVED LOCAL EXCERPTS CONTEXT CONTAINER ---
{pdf_source_context}

--- EXECUTION INSTRUCTIONS ---
1. Deliver a thorough, high-value, deep answer addressing the user's core intent ONLY if it directly relates to the uploaded document context.
2. Incorporate and explicitly cite the document excerpts above to tie your advice directly to their uploaded text.
3. CRITICAL MANDATE: If the context container reads "No relevant document excerpts found," or if the topic is completely absent from the retrieved chunks, you must politely but firmly refuse to answer. State clearly that the requested topic cannot be found within the uploaded document.
"""

                    foundry_headers = {"api-key": AZURE_OPENAI_KEY, "Content-Type": "application/json"}
                    foundry_payload = {
                        "messages": [
                            {"role": "system", "content": unbounded_system_prompt},
                            {"role": "user", "content": str(user_query).strip()}
                        ],
                        "max_completion_tokens": 2048,
                        "temperature": 0.4,
                        "stream": True
                    }

                    gateway_url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{MODEL_DEPLOYMENT}/chat/completions?api-version=2024-06-01"

                    try:
                        with requests.post(gateway_url, headers=foundry_headers, stream=True, json=foundry_payload) as stream_pipe:
                            if stream_pipe.status_code == 200:
                                for binary_line in stream_pipe.iter_lines():
                                    if binary_line:
                                        decoded_line = binary_line.decode("utf-8").strip()
                                        if decoded_line.startswith("data: "):
                                            decoded_line = decoded_line[6:]
                                        if decoded_line == "[DONE]":
                                            break
                                        try:
                                            json_chunk = json.loads(decoded_line)
                                            delta = json_chunk["choices"][0].get("delta", {})
                                            if "content" in delta:
                                                accumulated_text += delta["content"]
                                                response_placeholder.markdown(accumulated_text + "▌")
                                        except Exception:
                                            pass
                                response_placeholder.markdown(accumulated_text)
                                st.session_state.audit_chat_history.append({"role": "assistant", "content": accumulated_text})
                            else:
                                st.error(f"🔴 AI Gateway Error ({stream_pipe.status_code}): {stream_pipe.text}")
                    except Exception as network_fail:
                        st.error(f"🔴 Route failure: Lost connectivity. ({str(network_fail)})")

            st.rerun()

# =========================================================================
# BOTTOM ROW SCREEN MARGIN: ACCURACY BACKSTAGE MATRIX PANEL
# =========================================================================
if "audit_chat_history" in st.session_state and len(st.session_state.audit_chat_history) > 1:
    try:
        last_user_msg = [m["content"] for m in st.session_state.audit_chat_history if m["role"] == "user"][-1]
        # Make sure our backstage metrics window matches the filtered workspace constraint as well!
        raw_data_dump = st.session_state.local_rag.search_workspace_vector_space(
            query=last_user_msg, 
            workspace_name=st.session_state.active_notebook, 
            top_k=3
        )
        
        if raw_data_dump:
            st.markdown("---")
            st.markdown("### 🧮 Backstage RAG Math Inspection Matrix")
            metric_cols = st.columns(len(raw_data_dump))
            
            for idx, item in enumerate(raw_data_dump):
                with metric_cols[idx]:
                    with st.container(border=True):
                        st.markdown(f"**🧩 Chunk Match #{idx + 1} (Page {item['page']})**")
                        st.caption(f"Source File: `{item['source']}`")
                        st.caption(f"Accuracy: **{item['accuracy_percentage'] * 100:.2f}%**")
                        st.text_area("Chunk Content Preview:", value=item['chunk_text'][:200] + "...", height=90, disabled=True, key=f"prev_{idx}")
                        st.markdown(f"FAISS Distance Metric: `{item['raw_distance']:.4f}`")
    except Exception:
        pass