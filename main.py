from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
import asyncio
import subprocess
from concurrent_log_handler import ConcurrentRotatingFileHandler
from scripts.utils import get_gcs_client, gcs_list_files, parse_gcs_path, setup_logging, load_config
from scripts.run_pipeline import process_domain
import tempfile
import uuid
from utils.input_validator import validate_input, sanitize_input
from utils.retriever import hybrid_search, IndexNotFoundError
from utils.llm_interface import create_prompt, generate_response, handle_llm_errors
from utils.response_formatter import format_response
from utils.config_manager import get_system_prompt, update_system_prompt
from utils.history_manager import (
    get_buffer_session_history, 
    update_buffer_session_history, 
    generate_session_id, 
    get_all_sessions, 
    get_session_conversation_by_title,
    save_buffer_to_session_history,
    update_session_title,
    delete_session_by_title,
    check_title_exists,
    get_last_3_session_history_by_title,
    update_session_history_by_title
)
from carbon.carbon import  router as carbon_router

# Load environment variables
load_dotenv()
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
# Initialize FastAPI app (ONLY ONE - keep your original one)
app = FastAPI(
    title="RAG Vector Database API",
    description="API for managing a RAG-based vector database with FAISS and Sentence Transformer embeddings with Carbon Calculator.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - modify for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the carbon calculator router
app.include_router(carbon_router)

# ALL YOUR EXISTING ENDPOINTS FROM ORIGINAL main.py GO HERE
# Copy them exactly as they were - for example:

@app.get("/")
async def root():
    return {
        "message": "RAG Vector Database API with Carbon Calculator", 
        "version": "1.0.0",
        "services": ["rag-database", "carbon-calculator"]
    }
# Configure logging
def configure_logging():
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    
    log_file = "api.log"
    file_handler = ConcurrentRotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    return logger

logger = configure_logging()

# Pydantic model for delete request
class DeleteFilesRequest(BaseModel):
    filenames: List[str]

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    num_results: Optional[int] = 5
    similarity_threshold: Optional[float] = 0.5
    session_id: Optional[str] = None
    use_session_history: Optional[bool] = False
    session_history_title: Optional[str] = None

class SystemPromptRequest(BaseModel):
    prompt: str

class SaveSessionRequest(BaseModel):
    session_id: str
    title: str

class UpdateTitleRequest(BaseModel):
    old_title: str
    new_title: str

# Helper functions
def validate_file_extension(filename: str) -> bool:
    supported_extensions = {'.txt', '.docx', '.pdf', '.jpg', '.png', '.xlsx'}
    ext = '.' + filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in supported_extensions

async def upload_file_to_gcs(file: UploadFile, domain: str, client: storage.Client) -> str:
    try:
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob_path = f"docs/{domain}/{file.filename}"
        blob = bucket.blob(blob_path)
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            blob.upload_from_filename(temp_file.name)
        
        os.unlink(temp_file.name)
        logger.info(f"Uploaded file {file.filename} to gs://{GCS_BUCKET_NAME}/{blob_path}")
        return blob_path
    except Exception as e:
        logger.error(f"Error uploading {file.filename} to GCS: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload {file.filename}: {str(e)}")

async def run_pipeline_async(domain: str, base_dir: str, global_config: Dict[str, Any], force: bool = False):
    try:
        logger.info(f"Starting pipeline for domain {domain}")
        process_domain(domain, base_dir, global_config, stages=['ingest', 'chunk', 'embed', 'index'], skip_existing=False, force=force)
        logger.info(f"Completed pipeline for domain {domain}")
    except Exception as e:
        logger.error(f"Error running pipeline for domain {domain}: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline failed for domain {domain}: {str(e)}")

def get_file_metadata(blob: storage.Blob) -> Dict[str, Any]:
    return {
        "filename": blob.name.rsplit('/', 1)[-1],
        "size_bytes": blob.size,
        "upload_date": blob.updated.isoformat(),
        "path": f"gs://{GCS_BUCKET_NAME}/{blob.name}"
    }

def create_comprehensive_prompt(query: str, context: List[str], system_prompt: str, domain: str, has_knowledge_base: bool, history_string: str = "") -> str:
    """
    Create a comprehensive prompt that includes system prompt, conversation history, knowledge base, and user query.
    
    Args:
        query: The user query
        context: List of relevant document contents from knowledge base
        system_prompt: The domain-specific system prompt
        domain: The domain name
        has_knowledge_base: Whether relevant documents were found
        history_string: Previous conversation history (default: empty string)
    
    Returns:
        Formatted prompt string for the LLM
    """
    knowledge_base_section = ""
    if has_knowledge_base and context:
        knowledge_base_section = f"""
KNOWLEDGE BASE CONTEXT from {domain} domain:
"""
        for i, content in enumerate(context, 1):
            knowledge_base_section += f"\nDocument {i}:\n{content}\n"
    else:
        knowledge_base_section = f"\nNo specific documents found in the {domain} knowledge base for this query."
    
    comprehensive_prompt = f"""{system_prompt}

{history_string}

{knowledge_base_section}

USER QUERY: {query}

RESPONSE:"""
    
    return comprehensive_prompt

# API Endpoints
@app.post("/domain/{domain_name}/upload")
async def upload_files(domain_name: str, files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload multiple files to a specified domain and trigger pipeline processing.
    """
    if not domain_name:
        raise HTTPException(status_code=400, detail="Domain name cannot be empty")
    
    client = get_gcs_client()
    
    for file in files:
        if not validate_file_extension(file.filename):
            raise HTTPException(status_code=400, detail=f"Unsupported file type for {file.filename}")
    
    global_config_path = f"gs://{GCS_BUCKET_NAME}/scripts/config/global.yaml"
    try:
        global_config = load_config(global_config_path)
    except Exception as e:
        logger.error(f"Failed to load global config: {e}")
        raise HTTPException(status_code=500, detail="Failed to load global configuration")
    
    uploaded_paths = []
    for file in files:
        path = await upload_file_to_gcs(file, domain_name, client)
        uploaded_paths.append(path)
    
    background_tasks.add_task(run_pipeline_async, domain_name, f"gs://{GCS_BUCKET_NAME}", global_config)
    
    return JSONResponse(
        status_code=202,
        content={
            "message": f"Files uploaded successfully to domain {domain_name}. Pipeline processing started in the background.",
            "uploaded_files": uploaded_paths
        }
    )

@app.get("/domain/{domain_name}/files")
async def list_files(domain_name: str, page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100)):
    """
    List files in a specified domain with pagination.
    """
    if not domain_name:
        raise HTTPException(status_code=400, detail="Domain name cannot be empty")
    
    client = get_gcs_client()
    files = gcs_list_files(GCS_BUCKET_NAME, f"docs/{domain_name}/")
    
    bucket = client.bucket(GCS_BUCKET_NAME)
    file_metadata = []
    for file_path in files:
        blob_name = file_path.replace(f"gs://{GCS_BUCKET_NAME}/", '')
        blob = bucket.get_blob(blob_name)
        if blob:
            file_metadata.append(get_file_metadata(blob))
    
    total_files = len(file_metadata)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_files = file_metadata[start_idx:end_idx]
    
    return {
        "domain": domain_name,
        "files": paginated_files,
        "total_files": total_files,
        "page": page,
        "page_size": page_size,
        "total_pages": (total_files + page_size - 1) // page_size
    }

@app.delete("/domain/{domain_name}/files")
async def delete_files(domain_name: str, request: DeleteFilesRequest, background_tasks: BackgroundTasks = None):
    """
    Delete specified files from a domain and rebuild the index.
    """
    if not domain_name:
        raise HTTPException(status_code=400, detail="Domain name cannot be empty")
    if not request.filenames:
        raise HTTPException(status_code=400, detail="No filenames provided for deletion")
    
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    
    global_config_path = f"gs://{GCS_BUCKET_NAME}/scripts/config/global.yaml"
    try:
        global_config = load_config(global_config_path)
    except Exception as e:
        logger.error(f"Failed to load global config: {e}")
        raise HTTPException(status_code=500, detail="Failed to load global configuration")
    
    deleted_files = []
    not_found_files = []
    for filename in request.filenames:
        blob_path = f"docs/{domain_name}/{filename}"
        blob = bucket.blob(blob_path)
        if blob.exists():
            blob.delete()
            deleted_files.append(filename)
            logger.info(f"Deleted file gs://{GCS_BUCKET_NAME}/{blob_path}")
        else:
            not_found_files.append(filename)
            logger.warning(f"File not found: gs://{GCS_BUCKET_NAME}/{blob_path}")
    
    if deleted_files:
        background_tasks.add_task(run_pipeline_async, domain_name, f"gs://{GCS_BUCKET_NAME}", global_config, force=True)
    
    response = {
        "message": f"Deletion request processed for domain {domain_name}",
        "deleted_files": deleted_files,
        "not_found_files": not_found_files
    }
    
    if not_found_files and not deleted_files:
        raise HTTPException(status_code=404, detail="No specified files found for deletion")
    
    return response

@app.post("/query/{domain}")
async def query_domain(domain: str, request: QueryRequest):
    """Query the domain and return a context-aware response generated by the LLM."""
    if not domain:
        raise HTTPException(status_code=400, detail="Domain name cannot be empty")
    
    # Validate and sanitize input
    try:
        sanitized_query = sanitize_input(request.query)
        if not validate_input(sanitized_query, domain):
            raise HTTPException(status_code=400, detail="Invalid query format")
    except Exception:
        sanitized_query = request.query.strip()
        if not sanitized_query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Use provided session_id or generate a new one
    session_id = request.session_id if request.session_id else generate_session_id()
    
    # Get history based on the request
    history = []
    history_string = ""
    history_source = "buffer"
    use_existing_session = False
    
    if request.use_session_history and request.session_history_title:
        # Check if the session title exists in session_histories collection
        if check_title_exists(request.session_history_title, domain):
            # Use existing session history - get last 3 interactions for context
            history = get_last_3_session_history_by_title(request.session_history_title, domain)
            history_source = "existing_session"
            use_existing_session = True
            if history:
                history_string = "Previous conversation:\n"
                for interaction in history:
                    history_string += f"User: {interaction['query']}\nAssistant: {interaction['response']}\n"
        else:
            # Session title doesn't exist, fall back to buffer history
            history = get_buffer_session_history(session_id, domain)
            history_source = "buffer"
            if history:
                history_string = "Previous conversation:\n"
                for interaction in history:
                    history_string += f"User: {interaction['query']}\nAssistant: {interaction['response']}\n"
    else:
        # Use buffer session history (last 3 interactions)
        history = get_buffer_session_history(session_id, domain)
        history_source = "buffer"
        if history:
            history_string = "Previous conversation:\n"
            for interaction in history:
                history_string += f"User: {interaction['query']}\nAssistant: {interaction['response']}\n"
    
    # Get system prompt
    try:
        system_prompt = get_system_prompt(domain)
    except Exception as e:
        logger.error(f"Error getting system prompt for domain {domain}: {e}")
        system_prompt = f"You are a helpful AI assistant for the {domain} domain."
    
    # Retrieve documents
    documents = []
    try:
        documents = hybrid_search(sanitized_query, domain, k=request.num_results, similarity_threshold=request.similarity_threshold)
        logger.info(f"Retrieved {len(documents)} documents for query in domain {domain}")
    except IndexNotFoundError:
        logger.warning(f"No index found for domain {domain}")
    except Exception as e:
        logger.error(f"Error retrieving documents for domain {domain}: {e}")
    
    # Prepare context and sources
    context = [doc.page_content for doc in documents]
    sources = [
        {
            "document_id": doc.metadata.get("chunk_id", f"doc_{i}"),
            "source_file": doc.metadata.get("source_file", "unknown"),
            "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "similarity_score": doc.metadata.get("similarity_score", 0.0)
        }
        for i, doc in enumerate(documents)
    ]
    
    # Create prompt with history
    prompt = create_comprehensive_prompt(
        query=sanitized_query,
        context=context,
        system_prompt=system_prompt,
        domain=domain,
        has_knowledge_base=len(documents) > 0,
        history_string=history_string
    )
    
    # Generate response
    try:
        llm_response = generate_response(prompt)
        logger.info(f"Generated response for query in domain {domain}")
    except Exception as e:
        logger.error(f"Error generating LLM response: {e}")
        llm_response = handle_llm_errors(e)
    
    # Update history based on the scenario
    if use_existing_session and request.session_history_title:
        # Update the existing session history in session_histories collection
        update_session_history_by_title(request.session_history_title, domain, sanitized_query, llm_response)
        history_used = "existing_session"
    else:
        # Update buffer session history
        update_buffer_session_history(session_id, domain, sanitized_query, llm_response)
        history_used = "buffer"
    
    # Determine response type
    response_type = "knowledge_based" if documents else "conversational"
    
    return {
        "query": request.query,
        "answer": llm_response,
        "sources": sources,
        "response_type": response_type,
        "domain": domain,
        "documents_found": len(documents),
        "session_id": session_id,
        "history_used": history_used,
        "session_title": request.session_history_title if use_existing_session else None
    }

@app.get("/domain/{domain}/sessions")
async def list_sessions(domain: str):
    """
    List all session titles for a specific domain.
    
    Args:
        domain: The domain name.
    
    Returns:
        A list of session titles for the domain.
    """
    if not domain:
        raise HTTPException(status_code=400, detail="Domain name cannot be empty")
    
    try:
        sessions = get_all_sessions(domain)
        return {
            "message": f"Sessions retrieved successfully for domain {domain}",
            "domain": domain,
            "sessions": sessions,
            "total_sessions": len(sessions)
        }
    except Exception as e:
        logger.error(f"Error retrieving sessions for domain {domain}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve sessions: {str(e)}")

@app.get("/domain/{domain}/session/{title}")
async def get_conversation(domain: str, title: str):
    """
    Retrieve the full conversation for a specific title in a domain.
    
    Args:
        domain: The domain name.
        title: The unique title for the session.
    
    Returns:
        The title, domain, and conversation history.
    """
    if not domain:
        raise HTTPException(status_code=400, detail="Domain name cannot be empty")
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    
    try:
        conversation = get_session_conversation_by_title(title, domain)
        if not conversation["history"]:
            raise HTTPException(status_code=404, detail=f"No conversation found for title '{title}' in domain {domain}")
        return {
            "message": f"Conversation retrieved for title '{title}' in domain {domain}",
            "conversation": conversation
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation for title '{title}' in domain {domain}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve conversation: {str(e)}")

@app.put("/domain/{domain}/session/title")
async def update_title(domain: str, request: UpdateTitleRequest):
    """
    Update the title of an existing session in a domain.
    
    Args:
        domain: The domain name.
        request: Contains old_title and new_title.
    
    Returns:
        Confirmation of the title update.
    """
    if not domain:
        raise HTTPException(status_code=400, detail="Domain name cannot be empty")
    if not request.old_title:
        raise HTTPException(status_code=400, detail="Old title cannot be empty")
    if not request.new_title:
        raise HTTPException(status_code=400, detail="New title cannot be empty")
    
    try:
        # Check if new title already exists
        if check_title_exists(request.new_title, domain):
            raise HTTPException(status_code=400, detail=f"New title '{request.new_title}' already exists in domain {domain}")
        
        success = update_session_title(request.old_title, request.new_title, domain)
        if success:
            return {
                "message": f"Title updated successfully from '{request.old_title}' to '{request.new_title}' in domain {domain}",
                "old_title": request.old_title,
                "new_title": request.new_title,
                "domain": domain
            }
        else:
            raise HTTPException(status_code=404, detail=f"Session with title '{request.old_title}' not found in domain {domain}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating title from '{request.old_title}' to '{request.new_title}' in domain {domain}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update title: {str(e)}")

@app.delete("/domain/{domain}/session/{title}")
async def delete_session(domain: str, title: str):
    """
    Delete a session from session_histories by title in a domain.
    
    Args:
        domain: The domain name.
        title: The title of the session to delete.
    
    Returns:
        Confirmation of the deletion.
    """
    if not domain:
        raise HTTPException(status_code=400, detail="Domain name cannot be empty")
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    
    try:
        success = delete_session_by_title(title, domain)
        if success:
            return {
                "message": f"Session with title '{title}' deleted successfully from domain {domain}",
                "title": title,
                "domain": domain
            }
        else:
            raise HTTPException(status_code=404, detail=f"Session with title '{title}' not found in domain {domain}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session with title '{title}' in domain {domain}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

# Startup event to ensure project structure
@app.on_event("startup")
async def startup_event():
    from scripts.setup_structure import create_project_structure
    try:
        create_project_structure()
        logger.info("Verified/created project structure in GCS")
    except Exception as e:
        logger.error(f"Error setting up project structure: {e}")
        raise HTTPException(status_code=500, detail="Failed to set up project structure")

@app.get("/domain/{domain}/system-prompt")
async def get_system_prompt_endpoint(domain: str):
    """
    Retrieve the system prompt for a specific domain.
    
    Args:
        domain: The domain name.
    
    Returns:
        The system prompt text.
    """
    if not domain:
        raise HTTPException(status_code=400, detail="Domain name cannot be empty")
    
    try:
        prompt = get_system_prompt(domain)
        return {"domain": domain, "system_prompt": prompt}
    except Exception as e:
        logger.error(f"Error retrieving system prompt for domain {domain}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve system prompt: {str(e)}")

@app.post("/domain/{domain}/system-prompt")
async def update_system_prompt_endpoint(domain: str, request: SystemPromptRequest):
    """
    Update the system prompt for a specific domain.
    
    Args:
        domain: The domain name.
        request: The new system prompt.
    
    Returns:
        Confirmation of the update.
    """
    if not domain:
        raise HTTPException(status_code=400, detail="Domain name cannot be empty")
    
    try:
        update_system_prompt(domain, request.prompt)
        logger.info(f"Updated system prompt for domain {domain}")
        return {"message": f"System prompt updated for domain {domain}"}
    except Exception as e:
        logger.error(f"Error updating system prompt for domain {domain}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update system prompt: {str(e)}")

@app.post("/domain/{domain}/save-session")
async def save_session(domain: str, request: SaveSessionRequest):
    """
    Save a buffer session to session_histories with a unique title.
    
    Args:
        domain: The domain name.
        request: Contains session_id and title.
    
    Returns:
        Confirmation of the save operation.
    """
    if not domain:
        raise HTTPException(status_code=400, detail="Domain name cannot be empty")
    if not request.session_id:
        raise HTTPException(status_code=400, detail="Session ID cannot be empty")
    if not request.title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    
    try:
        # Check if title already exists
        if check_title_exists(request.title, domain):
            raise HTTPException(status_code=400, detail=f"Title '{request.title}' already exists in domain {domain}")
        
        success = save_buffer_to_session_history(request.session_id, domain, request.title)
        if success:
            return {
                "message": f"Session saved successfully as '{request.title}' in domain {domain}",
                "title": request.title,
                "domain": domain
            }
        else:
            raise HTTPException(status_code=404, detail=f"Buffer session {request.session_id} not found in domain {domain}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving session: {str(e)}")
    
