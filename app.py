"""
🤖 AI Knowledge Base Q&A Bot — Streamlit Interface
RAG-powered chatbot using HuggingFace + FAISS vector search.
Runs 100% locally — no API keys required.
"""

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI Q&A Bot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── No API Key Needed ─────────────────────────────────────────


@st.cache_resource(show_spinner="Loading AI model...")
def load_qa_chain():
    """Load FAISS index and build the QA chain (cached)."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    prompt_template = """You are an expert AI assistant with deep knowledge of artificial intelligence, 
machine learning, and data science. Use the following context to answer the question. 
If the answer is not in the context, say "I don't have enough information to answer that."

Provide clear, well-structured answers. Use bullet points when listing items.
Keep answers concise but comprehensive.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 300},
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa_chain


# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/artificial-intelligence.png",
        width=80,
    )
    st.title("AI Q&A Bot")
    st.markdown("---")
    st.markdown(
        """
    **Tech Stack:**
    - 🧠 GPT-2 (Local, Free)
    - 📐 all-MiniLM-L6-v2 Embeddings
    - ⚡ FAISS Vector Store
    - 🔗 LangChain
    - 🎈 Streamlit
    - 🆓 No API Key Required
    """
    )
    st.markdown("---")
    st.markdown(
        """
    **Sample Questions:**
    - What is RAG?
    - Compare CNNs and RNNs
    - How do vector databases work?
    - What are the types of ML?
    - Explain transformer architecture
    """
    )
    st.markdown("---")
    st.caption("Built with ❤️ using LangChain + HuggingFace — 100% Free")

# ─── Main Chat Interface ──────────────────────────────────────

st.title("🤖 AI Knowledge Base Q&A")
st.markdown("Ask me anything about **AI, Machine Learning, Deep Learning, NLP, LLMs**, and more!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Ask a question about AI..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            qa_chain = load_qa_chain()
            result = qa_chain.invoke({"query": user_input})

            answer = result["result"]
            sources = result["source_documents"]

            st.markdown(answer)

            # Show source chunks in expander
            with st.expander(f"📚 View {len(sources)} source chunks"):
                for i, doc in enumerate(sources, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.caption(doc.page_content[:300] + "...")
                    st.markdown("---")

    st.session_state.messages.append({"role": "assistant", "content": answer})
