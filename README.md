# 🧠 DocMind AI

An intelligent PDF Question-Answering system powered by Retrieval-Augmented Generation (RAG). DocMind AI enables users to upload PDF documents, extract relevant information, and ask natural language questions to obtain context-aware answers from the document content.

## 🚀 Features

* 📄 Upload and process PDF documents
* 🔍 Semantic document search using vector embeddings
* 🤖 Context-aware question answering with Mistral AI
* 🧠 Retrieval-Augmented Generation (RAG) pipeline
* ⚡ Fast document retrieval using ChromaDB
* 🎨 Interactive and user-friendly Streamlit interface
* 📚 Multi-page document understanding

## 🛠️ Tech Stack

### Frontend

* Streamlit

### AI & Machine Learning

* Mistral AI
* Hugging Face Embeddings
* LangChain

### Vector Database

* ChromaDB

### Backend

* Python

## 🏗️ System Architecture

1. User uploads PDF documents.
2. Documents are processed and split into chunks.
3. Hugging Face Embeddings convert text into vector representations.
4. ChromaDB stores the embeddings.
5. User submits a query.
6. Relevant document chunks are retrieved.
7. Mistral AI generates an answer using retrieved context.
8. Response is displayed through the Streamlit interface.

## 📂 Project Structure

```text
DocMind-AI/
├── document_loaders/
├── retrievers/
├── vectorStore/
├── create_database.py
├── finalApp.py
├── finalMain.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation

```bash
git clone https://github.com/your-username/DocMind-AI.git
cd DocMind-AI
pip install -r requirements.txt
```

## 🔑 Environment Variables

Create a `.env` file:

```env
MISTRAL_API_KEY=your_api_key
```

## ▶️ Run Locally

```bash
streamlit run finalApp.py
```

## 🎯 Future Enhancements

* Multi-document comparison
* Chat history support
* Citation and source highlighting
* OCR support for scanned PDFs
* Support for DOCX and TXT files
* User authentication

## 👩‍💻 Author

Gauri

B.Tech Computer Science Engineering Student

Passionate about Artificial Intelligence, Machine Learning, and Generative AI.

```
```
