from langchain_core.documents import Document

from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from langchain_groq import ChatGroq

from dotenv import load_dotenv


load_dotenv()


# Documents
docs = [
    Document(
        page_content="Gradient descent is an optimization algorithm used in machine learning."
    ),
    Document(
        page_content="Gradient descent minimizes the loss function."
    ),
    Document(
        page_content="Gradient descent is an optimization that minimizes the loss function."
    ),
    Document(
        page_content="Neural networks use gradient descent for training."
    ),
    Document(
        page_content="Support Vector Machines are supervised learning algorithms."
    )
]


# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)


# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# Multi Query Retriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)


# User query
query = "What is gradient descent?"


# Retrieve documents
docs = multi_query_retriever.invoke(query)


# Display results
print("\nRetrieved Documents:\n")

for i, doc in enumerate(docs, start=1):
    print("=" * 60)
    print(f"Document {i}")
    print(doc.page_content)