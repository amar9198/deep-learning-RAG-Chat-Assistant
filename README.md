# 🤖 Deep Learning RAG Chat Assistant

An AI-powered **Retrieval-Augmented Generation (RAG)** application that allows users to ask questions about a deep learning document and receive answers based only on the relevant information retrieved from the document.

The application combines **Mistral AI embeddings, ChromaDB, LangChain, and Groq LLMs** to provide a document-based question-answering system.

## 🚀 Live Demo

👉 **[Try the Deep Learning RAG Chat Assistant](https://deep-learning-rag-chat-assistant-peerwcw6feuz3uy4d7zkyb.streamlit.app/)**

---

## 📌 Project Overview

Large Language Models can generate useful answers, but they may sometimes provide information that is not present in a specific document.

This project uses **Retrieval-Augmented Generation (RAG)** to solve this problem.

Instead of asking the LLM to answer directly, the application:

1. Loads the document.
2. Splits the document into smaller chunks.
3. Converts the chunks into vector embeddings.
4. Stores the embeddings in ChromaDB.
5. Retrieves the most relevant chunks for a user's question.
6. Sends the retrieved context to the LLM.
7. Generates an answer based on the retrieved document content.

### RAG Pipeline

```text
                Deep Learning PDF
                       │
                       ▼
                Document Loader
                       │
                       ▼
                 Text Splitting
                       │
                       ▼
             Mistral AI Embeddings
                       │
                       ▼
                   ChromaDB
                       │
                       ▼
                 User Question
                       │
                       ▼
              MMR Retriever
                       │
                       ▼
              Relevant Context
                       │
                       ▼
                Groq LLM
                       │
                       ▼
                Final Answer
```

---

## ✨ Features

* 📚 Question answering from a deep learning document
* 🔎 Semantic document search
* 🧠 Retrieval-Augmented Generation
* 🔢 Mistral AI embeddings
* 🗄️ ChromaDB vector database
* 🎯 Maximum Marginal Relevance (MMR) retrieval
* 🤖 Groq-powered LLM responses
* 🌐 Streamlit web interface
* 📄 PDF document processing
* 🔐 Environment-variable based API key management
* 📖 Retrieved context can be viewed in the application

---

## 🛠️ Technologies Used

| Technology          | Purpose                         |
| ------------------- | ------------------------------- |
| Python              | Main programming language       |
| LangChain           | RAG application framework       |
| LangChain Community | Document loading and utilities  |
| LangChain Chroma    | ChromaDB integration            |
| Mistral AI          | Text embeddings                 |
| Groq                | Large Language Model            |
| ChromaDB            | Vector database                 |
| PyPDF               | PDF document loading            |
| Streamlit           | Web application interface       |
| python-dotenv       | Environment variable management |

---

## 🧠 Models Used

### Embedding Model

```text
mistral-embed
```

Mistral AI's embedding model is used to convert document chunks and user queries into numerical vector representations.

### Language Model

```text
openai/gpt-oss-120b
```

The model is accessed through the Groq API and is used to generate the final answer from the retrieved document context.

---

## 🔍 Retrieval Configuration

The project uses **Maximum Marginal Relevance (MMR)** retrieval.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)
```

### Parameters

* `k = 4`
  Retrieves the 4 most relevant document chunks.

* `fetch_k = 10`
  Initially considers 10 candidate chunks.

* `lambda_mult = 0.5`
  Balances relevance and diversity between retrieved chunks.

MMR helps reduce the retrieval of multiple highly similar chunks and can provide more diverse context to the language model.

---

## 📄 Document Processing

The project uses:

```python
PyPDFLoader
```

to load the deep learning PDF.

The document is divided into chunks using:

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

### Chunk Configuration

| Parameter        | Value |
| ---------------- | ----: |
| Chunk Size       |  1000 |
| Chunk Overlap    |   200 |
| Retrieved Chunks |     4 |
| Candidate Chunks |    10 |

The source PDF contains approximately **534 pages** and is divided into approximately **1,146 chunks** during database creation.

---

## 📁 Project Structure

```text
deep-learning-RAG-Chat-Assistant/
│
├── app.py
├── main.py
├── page.py
├── create_database.py
│
├── retrievers/
│   ├── arxiv_retriever.py
│   ├── mmr.py
│   └── multiquery.py
│
├── vector store/
│   └── db.py
│
├── documents loaders/
│   ├── deep-learning-material-dept-ece-ase-blr-1.pdf
│   ├── notes.txt
│   ├── pdf.py
│   ├── sample.pdf
│   └── test.py
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ How It Works

### 1. Load the document

The PDF is loaded using `PyPDFLoader`.

```python
data = PyPDFLoader(
    "documents loaders/deep-learning-material-dept-ece-ase-blr-1.pdf"
)

docs = data.load()
```

### 2. Split the document

The document is divided into smaller chunks:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)
```

### 3. Generate embeddings

Mistral AI converts the chunks into vector representations:

```python
embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)
```

### 4. Store vectors

The embeddings are stored using ChromaDB:

```python
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)
```

### 5. Retrieve relevant information

When the user asks a question, the retriever searches for relevant chunks.

```python
docs = retriever.invoke(query)
```

### 6. Generate the answer

The retrieved context is provided to the LLM.

The system prompt instructs the model to answer only using the retrieved context.

If the answer is not available in the retrieved document, the application returns:

```text
I could not find the answer in the document.
```

---

## 💻 Installation

### Clone the repository

```bash
git clone https://github.com/amar9198/deep-learning-RAG-Chat-Assistant.git
```

Move into the project directory:

```bash
cd deep-learning-RAG-Chat-Assistant
```

### Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key
GROQ_API_KEY=your_groq_api_key
```

Never upload your real `.env` file or API keys to GitHub.

The repository uses `.env.example` as a template.

---

## ▶️ Run Locally

Start the Streamlit application:

```powershell
streamlit run app.py
```

The application will open in your browser.

Usually the local address is:

```text
http://localhost:8501
```

---

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

Deployment workflow:

```text
Local Project
     │
     ▼
   Git
     │
     ▼
  GitHub
     │
     ▼
Streamlit Community Cloud
     │
     ▼
 Public Web Application
```

### Live Application

👉 **[Deep Learning RAG Chat Assistant](https://deep-learning-rag-chat-assistant-peerwcw6feuz3uy4d7zkyb.streamlit.app/)**

---

## 🔐 API Key Configuration on Streamlit

For deployment, API keys should be configured through Streamlit Secrets rather than committing them to GitHub.

Required secrets:

```toml
MISTRAL_API_KEY = "your_mistral_api_key"
GROQ_API_KEY = "your_groq_api_key"
```

---

## 🎯 Example Questions

You can ask questions related to the information contained in the deep learning document, such as:

```text
What is deep learning?

What is gradient descent?

What is a neural network?

What is backpropagation?

What is CNN?

What is RNN?

What is Word2Vec?
```

The system retrieves relevant sections from the document before generating the response.

---

## 📊 Advantages of the Project

### Traditional LLM

```text
Question
   ↓
LLM
   ↓
Answer
```

The model may not have access to the specific information contained in the user's document.

### RAG System

```text
Question
   ↓
Retriever
   ↓
Relevant Document Chunks
   ↓
LLM + Retrieved Context
   ↓
Answer
```

This allows the application to ground its responses in the supplied document.

---

## 🔮 Future Improvements

Possible improvements include:

* 📚 Support multiple PDF documents
* 📤 Allow users to upload their own documents
* 💬 Add conversational chat history
* 🧠 Add multi-document retrieval
* 🔎 Improve retrieval with hybrid search
* 📈 Add evaluation metrics for RAG performance
* ⚡ Optimize embedding and retrieval speed
* 🗂️ Add document metadata filtering
* 🧪 Add automated RAG evaluation
* 🎨 Improve the Streamlit UI
* 🔐 Add user authentication
* 📊 Display retrieval scores and document sources

---

## 📚 Learning Outcomes

This project demonstrates practical understanding of:

* Large Language Models
* Retrieval-Augmented Generation
* Vector embeddings
* Vector databases
* Semantic search
* Document chunking
* Information retrieval
* Prompt engineering
* LangChain
* Mistral AI
* Groq
* ChromaDB
* Streamlit
* Git and GitHub
* Cloud deployment

---

## 👨‍💻 Author

**Amar**

BTech — Computer Science / Engineering

GitHub:
https://github.com/amar9198

---

## ⭐ Project

If you find this project useful for learning about RAG and LLM applications, consider giving the repository a ⭐ on GitHub.

**Live Demo:**
https://deep-learning-rag-chat-assistant-peerwcw6feuz3uy4d7zkyb.streamlit.app/

**Repository:**
https://github.com/amar9198/deep-learning-RAG-Chat-Assistant
