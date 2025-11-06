import os
import shutil
import uvicorn
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # For fallback
from langchain_community.llms import CTransformers # For fallback chat
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import logging
import openai

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- API Key Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set in .env file. Fallback models will be used primarily.")

try:
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
        logger.info("OpenAI API configured successfully.")
except Exception as e:
   logger.error(f"Failed to configure OpenAI API, potentially invalid key. Fallback models will be used: {e}")
   OPENAI_API_KEY = None # Ensure OpenAI isn't used if config fails

# --- FastAPI App Initialization ---
app = FastAPI(title="Chat with PDF API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Allow React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Dictionaries & Constants ---
session_stores: Dict[str, Dict[str, Any]] = {}
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Fallback Models ---
FALLBACK_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Efficient local model
fallback_embeddings = None # Initialize lazily

# !!! IMPORTANT: Update this path to your downloaded GGUF model file
FALLBACK_CHAT_MODEL_PATH = "./model/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" 
fallback_llm = None # Initialize lazily

# OpenAI Model Names
OPENAI_CHAT_MODEL = "gpt-3.5-turbo"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_SUMMARIZE_MODEL = "gpt-3.5-turbo"

class ChatRequest(BaseModel):
    session_id: str
    message: str

class SummarizeRequest(BaseModel):
    session_id: str

# --- Embedding Model Initialization ---
def get_fallback_embeddings():
    """Initializes the fallback HuggingFace embedding model."""
    global fallback_embeddings
    if fallback_embeddings is None:
        try:
            logger.info(f"Initializing fallback embedding model: {FALLBACK_EMBEDDING_MODEL_NAME}")
            fallback_embeddings = HuggingFaceEmbeddings(
                model_name=FALLBACK_EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'} # Force CPU usage
            )
            logger.info("Fallback embedding model initialized successfully.")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to initialize fallback embedding model: {e}", exc_info=True)
            fallback_embeddings = "init_failed" # Mark as failed
    return fallback_embeddings if fallback_embeddings != "init_failed" else None

# --- NEW: Fallback Chat Model Initialization ---
def get_fallback_llm():
    """Initializes the fallback CTransformers chat model."""
    global fallback_llm
    if fallback_llm is None:
        try:
            if not os.path.exists(FALLBACK_CHAT_MODEL_PATH):
                logger.error(f"Fallback chat model file not found: {FALLBACK_CHAT_MODEL_PATH}")
                logger.error("Please download a GGUF model and update FALLBACK_CHAT_MODEL_PATH in main.py")
                fallback_llm = "init_failed"
                return None

            logger.info(f"Initializing fallback chat model from: {FALLBACK_CHAT_MODEL_PATH}")
            fallback_llm = CTransformers(
                model=FALLBACK_CHAT_MODEL_PATH,
                model_type="llama", # Adjust if using a different model type
                config={'context_length': 2048, 'max_new_tokens': 512, 'temperature': 0.1}
            )
            logger.info("Fallback chat model initialized successfully.")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to initialize fallback chat model: {e}", exc_info=True)
            fallback_llm = "init_failed" # Mark as failed
    return fallback_llm if fallback_llm != "init_failed" else None


# --- Helper Functions ---
def get_embedding_model(session_id: str):
    """Tries to get OpenAI embeddings, falls back to local model on error."""
    if OPENAI_API_KEY:
        try:
            logger.info(f"[{session_id}] Attempting to use OpenAI Embeddings ({OPENAI_EMBEDDING_MODEL})...")
            openai_embeddings = OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY
            )
            test_result = openai_embeddings.embed_query("test") # Test embedding to catch errors
            if not test_result or len(test_result) == 0:
                raise ValueError("OpenAI embeddings returned empty result")
            logger.info(f"[{session_id}] OpenAI Embeddings OK.")
            return openai_embeddings, "openai"
        except Exception as e:
            logger.warning(f"[{session_id}] OpenAI Embeddings failed: {type(e).__name__} - {str(e)}. Falling back...")
            fallback_model = get_fallback_embeddings()
            if fallback_model:
                logger.info(f"[{session_id}] Using fallback embedding model.")
                return fallback_model, "fallback"
            else:
                logger.error(f"[{session_id}] Fallback model is unavailable!")
                raise HTTPException(status_code=503, detail="Primary (OpenAI) and fallback embedding models unavailable.")
    else:
        logger.info(f"[{session_id}] No OpenAI API key provided. Using fallback embedding model.")
        fallback_model = get_fallback_embeddings()
        if fallback_model:
            return fallback_model, "fallback"
        else:
            logger.error(f"[{session_id}] No OpenAI key AND Fallback model failed to initialize!")
            raise HTTPException(status_code=503, detail="Embedding models unavailable (No OpenAI key and fallback failed).")


def load_and_process_pdf(file_path: str, session_id: str) -> Chroma:
    """Loads, splits, and embeds a PDF document using the best available model."""
    try:
        logger.info(f"[{session_id}] Loading PDF: {os.path.basename(file_path)}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        if not documents:
            logger.warning(f"[{session_id}] No text extracted from PDF.")
            raise ValueError("Could not extract text from the provided PDF.")
        logger.info(f"[{session_id}] Loaded {len(documents)} pages from PDF.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        logger.info(f"[{session_id}] Split PDF into {len(texts)} text chunks.")

        if not texts or all(not doc.page_content.strip() for doc in texts):
            logger.warning(f"[{session_id}] No meaningful text content found in PDF.")
            raise ValueError("The PDF appears to be empty or contains no extractable text.")

        full_text = " ".join([doc.page_content for doc in texts])
        embeddings, model_type = get_embedding_model(session_id)

        collection_name = f"pdf_chat_{session_id.replace('-', '_')}"
        logger.info(f"[{session_id}] Creating Chroma vector store '{collection_name}' using '{model_type}' embeddings...")

        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=None # Explicitly None for in-memory
        )
        logger.info(f"[{session_id}] Chroma vector store created successfully.")

        session_stores[session_id] = {
            "vector_store": vector_store,
            "full_text": full_text
        }
        return vector_store

    except HTTPException as http_exc:
        raise http_exc
    except ValueError as ve:
        logger.error(f"[{session_id}] Value error during PDF processing: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"[{session_id}] Unexpected error during PDF processing/vector store creation: {type(e).__name__} - {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF content due to a server error.")


# --- API Endpoints ---

@app.post("/upload")
async def upload_pdf(session_id: str = Form(...), file: UploadFile = File(...)):
    """Handles PDF file upload, processing, and vector store creation."""
    logger.info(f"Upload request received for session: {session_id}, filename: {file.filename}")
    if file.content_type != 'application/pdf':
        logger.warning(f"[{session_id}] Invalid file type received: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")

    file_path = os.path.join(TEMP_DIR, f"{session_id}_{file.filename}")

    try:
        logger.info(f"[{session_id}] Saving uploaded file temporarily to: {file_path}")
        with open(file_path, "wb") as buffer:
             shutil.copyfileobj(file.file, buffer)
        logger.info(f"[{session_id}] File saved successfully.")

        logger.info(f"[{session_id}] Starting PDF processing and embedding...")
        load_and_process_pdf(file_path, session_id)
        logger.info(f"[{session_id}] PDF processed and embedded successfully.")

        return {"filename": file.filename, "session_id": session_id, "message": "File processed successfully."}

    except HTTPException as http_exc:
        logger.error(f"[{session_id}] Upload failed (HTTPException): Status={http_exc.status_code}, Detail='{http_exc.detail}'")
        raise http_exc
    except ValueError as ve:
         logger.error(f"[{session_id}] Upload failed (ValueError): {ve}")
         raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"[{session_id}] Upload failed (Unexpected Server Error): {type(e).__name__} - {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected server error occurred during file processing. Please try again later.")

    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"[{session_id}] Removed temporary file: {file_path}")
            except Exception as e:
                logger.error(f"[{session_id}] Could not remove temporary file {file_path}: {e}")

@app.post("/chat")
async def chat_with_pdf(request: ChatRequest):
    """Handles chat queries using the RAG pipeline with OpenAI, falling back to local model."""
    logger.info(f"[{request.session_id}] Chat request received: '{request.message[:50]}...'")
    if request.session_id not in session_stores:
        logger.warning(f"[{request.session_id}] Session not found for chat request.")
        raise HTTPException(status_code=404, detail="Chat session not found or expired. Please upload the PDF again.")

    vector_store = session_stores[request.session_id]["vector_store"]
    llm = None
    model_type = "unknown"

    # 1. Try to use OpenAI
    if OPENAI_API_KEY:
        try:
            logger.info(f"[{request.session_id}] Initializing OpenAI Chat model ({OPENAI_CHAT_MODEL}) for QA...")
            llm = ChatOpenAI(
                model=OPENAI_CHAT_MODEL, 
                openai_api_key=OPENAI_API_KEY, 
                temperature=0.2
            )
            # Test with a simple invoke to catch quota/auth errors before RetrievalQA
            llm.invoke("Hello") 
            logger.info(f"[{request.session_id}] OpenAI Chat model initialized and tested.")
            model_type = "openai"
        except Exception as e:
            logger.warning(f"[{request.session_id}] OpenAI Chat model failed: {type(e).__name__} - {e}. Falling back...")
            llm = None # Ensure llm is None if it failed

    # 2. If OpenAI failed or wasn't available, use fallback
    if llm is None:
        logger.info(f"[{request.session_id}] Attempting to use fallback chat model.")
        llm = get_fallback_llm() # This function initializes the local model
        if llm:
            model_type = "fallback"
            logger.info(f"[{request.session_id}] Using fallback chat model.")
        else:
            logger.error(f"[{request.session_id}] Fallback chat model is unavailable!")
            # If OpenAI also failed, this is the final error
            if OPENAI_API_KEY: 
                raise HTTPException(status_code=503, detail="Chat functionality unavailable: OpenAI failed and fallback model is not available.")
            else:
                raise HTTPException(status_code=503, detail="Chat functionality unavailable: No OpenAI key and fallback model failed.")

    # 3. Proceed with the QA chain using whichever LLM was selected
    try:
        logger.info(f"[{request.session_id}] Setting up RetrievalQA chain with '{model_type}' model...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=False
        )
        logger.info(f"[{request.session_id}] Invoking QA chain with user query...")
        response = qa_chain.invoke(request.message)
        answer = response.get("result", "Sorry, I encountered an issue finding an answer in the document.")
        logger.info(f"[{request.session_id}] QA chain completed. Answer starts: '{answer[:50]}...'")
        return {"answer": answer}

    except Exception as e:
        # This will catch errors from the QA chain itself
        logger.error(f"[{request.session_id}] Error during chat processing (with {model_type} model): {type(e).__name__} - {e}", exc_info=True)
        detail = "An error occurred while processing your chat request."
        status_code = 500
        
        error_str = str(e).lower()
        if "rate_limit" in error_str or "quota" in error_str or "429" in error_str:
            detail = f"AI service rate limit or quota exceeded. Please try again later."
            status_code = 429
        elif "api key" in error_str or "unauthorized" in error_str:
            detail = "AI service API key is invalid or unauthorized."
            status_code = 401
            
        raise HTTPException(status_code=status_code, detail=detail)


@app.post("/summarize")
async def summarize_pdf(request: SummarizeRequest):
    """Generates a summary and suggested questions using OpenAI."""
    # This endpoint will still fail if OpenAI is out of quota.
    # A fallback LLM could be added here, but the prompt may need to be
    # changed to one that the local model understands well.
    logger.info(f"[{request.session_id}] Summarize request received.")
    if request.session_id not in session_stores or "full_text" not in session_stores[request.session_id]:
        logger.warning(f"[{request.session_id}] Session data or full text not found for summarization.")
        raise HTTPException(status_code=404, detail="Session expired or text not processed. Please upload the PDF again.")

    if not OPENAI_API_KEY:
        logger.error(f"[{request.session_id}] Cannot process summarization: OpenAI API key is not configured.")
        raise HTTPException(status_code=503, detail="Summarization functionality unavailable: AI service key not configured.")

    try:
        full_text = session_stores[request.session_id]["full_text"]
        max_summary_length = 15000 
        truncated_text = full_text[:max_summary_length]
        logger.info(f"[{request.session_id}] Using text length for summary: {len(truncated_text)} characters.")

        logger.info(f"[{request.session_id}] Initializing OpenAI for summarization...")
        llm = ChatOpenAI(
            model=OPENAI_SUMMARIZE_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.3
        )
        logger.info(f"[{request.session_id}] OpenAI initialized.")

        prompt = f"""Analyze the following document text and perform two tasks:
1. Write a concise, single-paragraph summary capturing the main points.
2. Generate exactly three distinct and insightful questions that a user might ask based on this text.

Return your response as a JSON object with this exact format:
{{
  "summary": "Your single-paragraph summary here.",
  "suggested_questions": [
    "Generated Question 1?",
    "Generated Question 2?",
    "Generated Question 3?"
  ]
}}

Document Text (potentially truncated):
---
{truncated_text}
---

Remember to return ONLY the JSON object, no other text."""

        logger.info(f"[{request.session_id}] Sending prompt to OpenAI for summary generation...")
        
        from langchain.schema import HumanMessage
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        response_text = response.content.strip()

        json_response_text = response_text
        if json_response_text.startswith("```json"):
            json_response_text = json_response_text[7:]
        elif json_response_text.startswith("```"):
            json_response_text = json_response_text[3:]
        if json_response_text.endswith("```"):
            json_response_text = json_response_text[:-3]
        json_response_text = json_response_text.strip()

        logger.info(f"[{request.session_id}] Attempting to parse JSON response from OpenAI...")
        json_response = json.loads(json_response_text)
        logger.info(f"[{request.session_id}] Summary and questions generated successfully.")

        if "summary" not in json_response or "suggested_questions" not in json_response or not isinstance(json_response["suggested_questions"], list):
             logger.error(f"[{request.session_id}] OpenAI response JSON did not match expected format. Received: {json_response_text}")
             raise ValueError("Parsed response did not contain expected 'summary' and 'suggested_questions' fields.")

        return json_response

    except json.JSONDecodeError as json_err:
        logger.error(f"[{request.session_id}] Failed to decode JSON response from OpenAI: {json_err}. Raw response: '{json_response_text}'")
        raise HTTPException(status_code=500, detail="Failed to parse the summary response from the AI service.")
    except Exception as e:
        logger.error(f"[{request.session_id}] Error during summarization: {type(e).__name__} - {e}", exc_info=True)
        detail = "Failed to generate summary."
        status_code = 500
        
        error_str = str(e).lower()
        if "api key" in error_str or "unauthorized" in error_str:
            detail = "Invalid or unauthorized OpenAI API Key."
            status_code = 401
        elif "429" in error_str or "rate_limit" in error_str or "quota" in error_str:
            detail = "OpenAI API rate limit or quota exceeded."
            status_code = 429
        elif isinstance(e, ValueError):
            detail = str(e)
            status_code = 500

        raise HTTPException(status_code=status_code, detail=detail)

# --- Root Endpoint ---
@app.get("/")
async def read_root():
    return {"message": "Chat with PDF Backend is running"}


# --- Run App ---
if __name__ == "__main__":
    logger.info("Starting Uvicorn server for local development...")
    # Initialize fallback models at startup for local dev
    get_fallback_embeddings()
    get_fallback_llm()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)