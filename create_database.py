#load pdf 
#split into chunks 
#create the embeddings 
#store into chroma 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma 
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("documents loaders/deep-learning-material-dept-ece-ase-blr-1.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

print("Number of pages:", len(docs))
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks[:3]):
    print("\n--- Chunk", i, "---")
    print(chunk.page_content[:500])

embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

vectorstore = Chroma.from_documents(
    documents= chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)
