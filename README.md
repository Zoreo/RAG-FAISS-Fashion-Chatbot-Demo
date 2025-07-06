# Retrieval-Augmented Generation (RAG) Demo
This project demonstrates a simple yet functional Retrieval-Augmented Generation (RAG) pipeline using:
- **FAISS** for fast vector-based document retrieval
- **OpenAI** embeddings (text-embedding-3-small) + LLM (GPT-3.5 Turbo)
- **LangChain** to connect retrieval with generation
- **CLI** simple interface for asking questions based on the document dataset. Great for testing and automation

## Project Structure
```
.
├── indexer.py           # Script to embed and index your documents
├── app_cli.py           # Command-line app to query the indexed documents
├── data/                # Text documents (fashion knowledge base)
├── faiss_index/         # Saved FAISS vector index (embeddings + metadata)
├── requirements.txt     # Python dependencies
├── .gitignore           # We don't want to accidentally upload our API keys
└── README.md            # What you're looking at right now
```
## Setup Instructions
Clone the repo
```
git clone https://github.com/Zoreo/RAG-FAISS-Fashion-Chatbot-Demo.git
cd RAG-FAISS-Fashion-Chatbot-Demo
```
## Install dependencies
```
pip install -r requirements.txt
```

## Set your OpenAI API key
Create a .env file in the root:
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Index your documents
This step converts your text files into vector embeddings and builds the FAISS index.
```
python indexer.py
```
The outputs are saved in ./faiss_index/.

## Run the CLI app
```
python app_cli.py
```

Now you can ask basic fashion-related questions:
```
Ask a question (or type 'exit'): How can I elevate my style?
```

## How It Works
The documents are split into chunks and embedded into 1536-dimensional vectors (text-embedding-3-small does this).
A FAISS index enables fast retrieval based on cosine similarity.
When a question is asked:
1. It's embedded and compared to document vectors.
2. The top k=3 relevant chunks are passed into the prompt for GPT.
3. GPT synthesizes a grounded answer using only that context.

## Security
- API keys are not included in the repository.
- .env is excluded via .gitignore.
- FAISS index is safe to include because it's locally generated.

## Possible Improvements
- Add a Streamlit UI for web-based use
- Containerizing the app with Docker would ensure environment consistency across machines, simplify setup for reviewers, and make the project easier to deploy or scale in production.
- Switch to Gemini embeddings or local models for cost efficiency
- Add support for document uploads and re-indexing

## Scalability Considerations
- Data scale: Switch from in-memory FAISS to disk-backed or distributed stores.
- Embedding batching: Chunk large datasets and batch embeddings to reduce cost and latency.
- Streaming responses: Stream OpenAI completions to the user instead of waiting for the full response.
- Frontend: Deploy as a full-fledged web app with login, document upload, and conversation history.
