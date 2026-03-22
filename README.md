# 🤖 AI Knowledge Base Q&A Bot

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about AI, Machine Learning, Deep Learning, NLP, and LLMs using a custom knowledge base.

Built with **HuggingFace** (GPT-2 + all-MiniLM-L6-v2), **FAISS vector search**, **LangChain**, and deployed with **Streamlit**.

**100% free — no API keys required. Runs entirely on your machine.**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Free_LLM-yellow?logo=huggingface)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Chat_UI-red?logo=streamlit)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-purple)

---

## 🎯 Features

- **RAG Pipeline** — Retrieves relevant context from a vector store before generating answers
- **100% Free & Local** — No API keys, no costs, runs entirely offline
- **HuggingFace Models** — GPT-2 for generation + all-MiniLM-L6-v2 for embeddings
- **FAISS Vector Search** — Sub-millisecond semantic similarity search across document chunks
- **Custom Prompt Engineering** — Tailored system prompt for professional, structured answers
- **Streamlit Chat UI** — Beautiful, interactive chat interface with message history
- **Source Transparency** — View the retrieved source chunks used to generate each answer
- **Persistent Index** — FAISS index saved to disk for instant loading without re-embedding

## 🏗️ Architecture

```
User Question
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Streamlit  │────▶│   Retriever  │────▶│    FAISS     │
│   Chat UI    │     │   (top-k=4)  │     │ Vector Store │
└─────────────┘     └──────────────┘     └─────────────┘
      │                                         │
      │              ┌──────────────┐           │
      └─────────────▶│  GPT-2 (Local)│◀──────────┘
                     │  + Context   │   Retrieved Chunks
                     └──────────────┘
                           │
                           ▼
                     Generated Answer
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | GPT-2 (HuggingFace, local) |
| **Embeddings** | all-MiniLM-L6-v2 (HuggingFace) |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **Framework** | LangChain |
| **Frontend** | Streamlit |
| **Language** | Python 3.11 |
| **Cost** | 🆓 100% Free |

## 📁 Project Structure

```
Q&A Bot/
├── app.py                  # Streamlit chatbot interface
├── main.ipynb              # Jupyter notebook (full RAG pipeline) 
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md
├── data/
│   └── knowledge_base.txt  # AI/ML knowledge base (10 sections)
└── faiss_index/            # Saved FAISS vector index (generated)
```

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/AI-QA-Bot-RAG.git
cd AI-QA-Bot-RAG
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Build the vector index
Run all cells in `main.ipynb` to:
- Load and chunk the knowledge base
- Create embeddings and build the FAISS index (downloads model automatically)
- Test the QA pipeline

**No API key setup needed!**

### 4. Launch the chatbot
```bash
streamlit run app.py
```

## 💬 Sample Questions

| Question | What It Tests |
|----------|---------------|
| "What is RAG and how does it work?" | Core RAG understanding |
| "Compare supervised vs unsupervised learning" | ML fundamentals |
| "What are the different types of neural networks?" | Deep learning |
| "How do vector databases work?" | Infrastructure knowledge |
| "What is GPT-4?" | LLM knowledge |
| "How is AI used in healthcare?" | Industry applications |

## 📊 Knowledge Base

The knowledge base covers 10 comprehensive sections:

1. **Introduction to AI** — History, types, and subfields
2. **Machine Learning Fundamentals** — Supervised, unsupervised, semi-supervised, reinforcement learning
3. **Deep Learning & Neural Networks** — CNNs, RNNs, Transformers, GANs
4. **Natural Language Processing** — Key tasks and breakthroughs
5. **Large Language Models** — GPT, BERT, LLaMA, Claude, Gemini
6. **RAG (Retrieval-Augmented Generation)** — Pipeline, components, use cases
7. **Vector Databases & Embeddings** — FAISS, Pinecone, ChromaDB, similarity metrics
8. **AI Frameworks & Tools** — LangChain, Streamlit, TensorFlow, PyTorch
9. **AI Ethics** — Bias, privacy, safety, explainability
10. **AI in Industry** — Healthcare, finance, retail, manufacturing, and more

## 🔑 Key Concepts Demonstrated

- **Retrieval-Augmented Generation (RAG)** — Combining retrieval with generation
- **Vector Embeddings** — Semantic text representation
- **Prompt Engineering** — Custom prompts for structured answers
- **Document Chunking** — Intelligent text splitting with overlap
- **FAISS Indexing** — Efficient approximate nearest neighbor search
- **Streamlit Deployment** — Interactive web UI with chat history
- **Source Attribution** — Transparent retrieval with expandable source chunks

## 📝 License

MIT License — feel free to use, modify, and share.
