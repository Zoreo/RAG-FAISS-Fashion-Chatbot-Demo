from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

import os
from dotenv import load_dotenv

load_dotenv()

docs = []
for filename in os.listdir("data"):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join("data", filename))
        docs.extend(loader.load())

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small", 
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

vectorstore = FAISS.from_documents(split_docs, embedding_model)

vectorstore.save_local("faiss_index")
