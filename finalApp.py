
import os
import time
import tempfile
from datetime import datetime

import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_mistralai import ChatMistralAI

from langchain_core.prompts import ChatPromptTemplate
api_key = st.secrets.get("MISTRAL_API_KEY", os.getenv("MISTRAL_API_KEY"))

st.set_page_config(
    page_title="DocMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

st.markdown("""
<style>

.block-container{
    padding-top:1.5rem;
}

.hero-title{
    text-align:center;
    font-size:4rem;
    font-weight:800;
    margin-top:30px;
    margin-bottom:10px;
}

.hero-subtitle{
    text-align:center;
    color:#9ca3af;
    font-size:1.1rem;
    margin-bottom:40px;
}

.metric-card{
    background:#111827;
    padding:15px;
    border-radius:15px;
    border:1px solid #374151;
    text-align:center;
}

.feature-card{
    background:#111827;
    padding:25px;
    border-radius:18px;
    border:1px solid #374151;
    text-align:center;
    height:170px;
}

.source-box{
    background:#1f2937;
    padding:10px;
    border-radius:10px;
    margin-top:10px;
}

.chat-header{
    text-align:center;
    margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)

DEFAULT_STATE = {
    "vector_store": None,
    "messages": [],
    "full_text": "",
    "chunks": [],
    "docs": [],
    "mode": "chat",
    "pdf_name": "",
    "page_count": 0,
    "word_count": 0,
    "processing_time": 0
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_llm():

    api_key = st.secrets.get(
        "MISTRAL_API_KEY",
        os.getenv("MISTRAL_API_KEY")
    )

    return ChatMistralAI(
        model="mistral-small-2603",
        temperature=0.2,
        api_key=api_key
    )
def build_vector_store(chunks):

    embeddings = load_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vector_store


def process_pdf(uploaded_file):

    start_time = time.time()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.read())
        pdf_path = temp_file.name

    loader = PyPDFLoader(pdf_path)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    full_text = "\n".join(
        doc.page_content
        for doc in docs
    )

    vector_store = build_vector_store(chunks)

    processing_time = round(
        time.time() - start_time,
        2
    )

    return {
        "docs": docs,
        "chunks": chunks,
        "full_text": full_text,
        "vector_store": vector_store,
        "page_count": len(docs),
        "word_count": len(full_text.split()),
        "processing_time": processing_time
    }


def show_document_stats():

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Pages",
            st.session_state.page_count
        )

    with col2:
        st.metric(
            "Chunks",
            len(st.session_state.chunks)
        )

    with col3:
        st.metric(
            "Words",
            f"{st.session_state.word_count:,}"
        )

    with col4:
        st.metric(
            "Processing",
            f"{st.session_state.processing_time}s"
        )


def get_chat_history():

    if len(st.session_state.messages) == 0:
        return ""

    history = []

    for msg in st.session_state.messages[-6:]:

        history.append(
            f"{msg['role']}: {msg['content']}"
        )

    return "\n".join(history)


def get_retriever():

    return st.session_state.vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are DocMind AI.

Answer ONLY from the provided context.

Rules:

1. Use only document information.
2. Never hallucinate.
3. Be concise.
4. Cite page numbers if available.
5. If answer not found say:

"I couldn't find the answer in the document."
"""
        ),
        (
            "human",
            """
Chat History:
{history}

Context:
{context}

Question:
{question}
"""
        )
    ]
)

llm = load_llm()

if st.session_state.vector_store is None:

    st.markdown("""
    <div class="hero-title">
        📚 DocMind AI
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero-subtitle">
        Chat with PDFs using RAG, Mistral AI and ChromaDB
        <br>
        Upload • Summarize • Analyze • Explore
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
        <h3>💬 Ask Questions</h3>
        <p>Chat naturally with your document using Retrieval Augmented Generation.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
        <h3>📄 Summarize</h3>
        <p>Generate concise document summaries instantly.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
        <h3>🔍 Insights</h3>
        <p>Discover topics, entities, concepts and takeaways.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file:

        try:

            with st.spinner(
                "📖 Processing PDF..."
            ):

                data = process_pdf(
                    uploaded_file
                )

                st.session_state.docs = data["docs"]
                st.session_state.chunks = data["chunks"]
                st.session_state.full_text = data["full_text"]
                st.session_state.vector_store = data["vector_store"]

                st.session_state.page_count = data["page_count"]
                st.session_state.word_count = data["word_count"]

                st.session_state.processing_time = data["processing_time"]

                st.session_state.pdf_name = uploaded_file.name

            st.success(
                "✅ Document processed successfully!"
            )

            st.rerun()

        except Exception as e:

            st.error(
                f"Error processing PDF:\n{e}"
            )
else:
    with st.sidebar:

        st.title("📚 DocMind AI")

        st.success("Document Loaded")

        st.markdown(
            f"**File:** {st.session_state.pdf_name}"
        )

        st.divider()

        if st.button(
            "💬 Ask Questions",
            use_container_width=True
        ):
            st.session_state.mode = "chat"

        if st.button(
            "📄 Summarize",
            use_container_width=True
        ):
            st.session_state.mode = "summary"

        if st.button(
            "🔍 Explore Insights",
            use_container_width=True
        ):
            st.session_state.mode = "insights"

        st.divider()

        st.subheader("📊 Document Stats")

        st.write(
            f"📄 Pages: {st.session_state.page_count}"
        )

        st.write(
            f"🧩 Chunks: {len(st.session_state.chunks)}"
        )

        st.write(
            f"📝 Words: {st.session_state.word_count:,}"
        )

        st.write(
            f"⚡ Processing: {st.session_state.processing_time}s"
        )

        st.divider()

        if st.button(
            "🗑 Clear Chat",
            use_container_width=True
        ):

            st.session_state.messages = []

            st.rerun()

        if st.button(
            "🔄 Upload New PDF",
            use_container_width=True
        ):

            st.session_state.vector_store = None
            st.session_state.messages = []
            st.session_state.full_text = ""
            st.session_state.docs = []
            st.session_state.chunks = []

            st.rerun()

        if st.button(
            "🚪 Exit",
            use_container_width=True
        ):

            st.session_state.clear()

            st.success(
                "Session Ended"
            )

            st.stop()

    st.markdown("""
    <div class="chat-header">
    <h1>📚 DocMind AI</h1>
    <p>AI Powered PDF Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    show_document_stats()

    st.divider()

    if st.session_state.mode == "summary":

        st.subheader(
            "📄 Document Summary"
        )

        if st.button(
            "Generate Summary",
            use_container_width=True
        ):

            with st.spinner(
                "Generating summary..."
            ):

                try:

                    summary_prompt = f"""
                    Generate a professional summary.

                    Include:

                    - Overview
                    - Main Topics
                    - Key Findings
                    - Important Conclusions

                    Document:

                    {st.session_state.full_text[:15000]}
                    """

                    response = llm.invoke(
                        summary_prompt
                    )

                    st.markdown(
                        response.content
                    )

                except Exception as e:

                    st.error(
                        f"Summary Error:\n{e}"
                    )

        st.stop()

    if st.session_state.mode == "insights":

        st.subheader(
            "🔍 Document Insights"
        )

        if st.button(
            "Generate Insights",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing document..."
            ):

                try:

                    insights_prompt = f"""
                    Analyze this document and provide:

                    1. Main Topics
                    2. Key Concepts
                    3. Important Takeaways
                    4. Technical Terms
                    5. Named Entities
                    6. Action Items
                    7. Executive Insights

                    Document:

                    {st.session_state.full_text[:15000]}
                    """

                    response = llm.invoke(
                        insights_prompt
                    )

                    st.markdown(
                        response.content
                    )

                except Exception as e:

                    st.error(
                        f"Insights Error:\n{e}"
                    )

        st.stop()

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

user_query = st.chat_input(
    "Ask a question about your PDF..."
)

if user_query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    with st.chat_message("user"):

        st.markdown(user_query)

    with st.chat_message("assistant"):

        try:

            retriever = get_retriever()

            with st.spinner(
                "Searching document..."
            ):

                retrieved_docs = retriever.invoke(
                    user_query
                )
            context = "\n\n".join(
                doc.page_content
                for doc in retrieved_docs
            )

            history = get_chat_history()

            final_prompt = RAG_PROMPT.invoke(
                {
                    "history": history,
                    "context": context,
                    "question": user_query
                }
            )
            response = llm.invoke(
                final_prompt
            )

            answer = response.content

            placeholder = st.empty()

            streamed_text = ""

            for word in answer.split():

                streamed_text += word + " "

                placeholder.markdown(
                    streamed_text + "▌"
                )

                time.sleep(0.015)

            placeholder.markdown(
                streamed_text
            )
            source_pages = []

            for doc in retrieved_docs:

                if "page" in doc.metadata:

                    source_pages.append(
                        doc.metadata["page"] + 1
                    )

            source_pages = sorted(
                list(set(source_pages))
            )

            if source_pages:

                st.markdown("---")

                st.markdown(
                    "### 📚 Sources"
                )

                source_text = ", ".join(
                    [
                        f"Page {page}"
                        for page in source_pages
                    ]
                )

                st.markdown(
                    f"""
                    <div class="source-box">
                    {source_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            final_answer = (
                answer +
                "\n\nSources: " +
                source_text
                if source_pages
                else answer
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_answer
                }
            )

        except Exception as e:

            error_message = (
                f"❌ Error: {str(e)}"
            )

            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message
                }
            )

st.markdown("---")

st.caption(
    "📚 DocMind AI | Powered by Mistral AI, LangChain, ChromaDB & HuggingFace"
)
