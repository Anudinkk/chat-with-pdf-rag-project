### Chat with PDF (RAG) - FastAPI & React

This project is a full-stack "Chat with PDF" application that uses Retrieval-Augmented Generation (RAG) to answer questions based on an uploaded document.

The frontend is a modern React application, and the backend is a robust API built with FastAPI. The entire application is containerized with Docker Compose for easy setup and deployment.

A key feature is its resilience: it's designed to use OpenAI's powerful models by default, but will automatically fall back to local, open-source models if the OpenAI API key is missing or quota is exceeded.

# Features

PDF Upload: Users can upload a PDF document.

RAG Pipeline: The backend processes the PDF, splits it into chunks, and generates vector embeddings.

Chat Interface: A clean, React-based chat UI allows users to ask questions about the document.

Dual-Model Fallback System:

OpenAI (Default): Uses gpt-3.5-turbo for chat and text-embedding-3-small for embeddings.

Local (Fallback): Automatically switches to ctransformers (e.g., TinyLlama) for chat and sentence-transformers for embeddings if OpenAI fails or is not configured.

Dockerized: Fully containerized with docker-compose for one-command setup.

# Tech Stack

Backend: FastAPI, LangChain, OpenAI, CTransformers, ChromaDB, Sentence-Transformers

Frontend: React, Tailwind CSS, Lucide Icons, Axios

Deployment: Docker, Docker Compose

# Setup and Running

Follow these steps to get the application running on your local machine.

1. Prerequisites

Docker and Docker Compose

Git

2. Clone the Repository

git clone [https://github.com/your-username/pdf-rag-chat.git](https://github.com/your-username/pdf-rag-chat.git)
cd pdf-rag-chat


3. Backend Configuration

Add Local Model

The local fallback model is not included in this repository.

Download a GGUF-compatible model (e.g., TinyLlama-1.1B-Chat Q4_K_M).

Create a models folder inside the backend directory: backend/models.

Place your downloaded .gguf file into this new folder.

Update backend/main.py to point to your model file:

# Inside backend/main.py
FALLBACK_CHAT_MODEL_PATH = "/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" # <-- Make sure this filename matches!


Create .env File

Create a .env file inside the backend directory (backend/.env). This will be used by the Docker container.

# backend/.env
OPENAI_API_KEY="sk-YOUR_OPENAI_API_KEY"


To test the fallback: You can leave the OPENAI_API_KEY blank or use an invalid key. The app will log a warning and switch to local models.

4. Build and Run with Docker Compose

From the root directory of the project, run:

docker-compose build


Once the build is complete, start the services:

docker-compose up


5. Access the Application

Frontend (React): http://localhost:3000

Backend (FastAPI Docs): http://localhost:8002/docs

API Endpoints

The FastAPI backend exposes the following main endpoints:

POST /upload: Handles PDF file uploads. Requires a session_id and the file.

POST /chat: Receives a chat message (session_id and message) and returns a RAG-generated answer.

POST /summarize: (Optional) Generates a summary of the document.

Feel free to customize this README.md with your repository name and any additional details!
