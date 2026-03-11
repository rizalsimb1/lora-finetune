# lora-finetune

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Stars](https://img.shields.io/github/stars/rizalsimb1/lora-finetune?style=social)
![Issues](https://img.shields.io/github/issues/rizalsimb1/lora-finetune)

> A production-grade Retrieval-Augmented Generation (RAG) system built with LangChain, ChromaDB, and OpenAI. Index documents, retrieve context, and generate grounded answers.

## ✨ Features

- ✅ Document ingestion pipeline supporting PDF, DOCX, Markdown, and web pages
- ✅ Semantic chunking with configurable overlap and size
- ✅ Vector embeddings with OpenAI or HuggingFace models
- ✅ ChromaDB and FAISS vector store backends
- ✅ Hybrid search (semantic + BM25 keyword)
- ✅ Multi-query retrieval for improved recall
- ✅ Streaming answers with source citations
- ✅ REST API via FastAPI for production deployment

## 🛠️ Tech Stack

`Python 3.11+` • `LangChain` • `ChromaDB` • `OpenAI` • `FastAPI` • `FAISS` • `PyPDF2`

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/rizalsimb1/lora-finetune.git
cd lora-finetune

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

```python
from rag_pipeline import RAGSystem

rag = RAGSystem(
    vector_store="chroma",
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",
)

# Index documents
rag.ingest("./docs/")
rag.ingest_url("https://docs.example.com/guide")

# Query with streaming
for token in rag.query_stream("How does authentication work?"):
    print(token, end="", flush=True)

# Query with sources
result = rag.query("What are the rate limits?")
print(result.answer)
print(result.sources)  # [{'doc': 'api-guide.pdf', 'page': 3, 'score': 0.92}]

```

## 📁 Project Structure

```
lora-finetune/
├── src/
│   └── main files
├── tests/
│   └── test files
├── requirements.txt
├── README.md
└── LICENSE
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'feat: add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ❤️ by <a href="https://github.com/rizalsimb1">rizalsimb1</a></p>

