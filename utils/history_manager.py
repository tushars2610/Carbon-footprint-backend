from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os
import uuid
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta
import ssl

# MongoDB connection configuration - Fixed for SSL issues
MONGODB_URL = os.getenv("MONGODB_URI") or os.getenv("MONGODB_URL")
DATABASE_NAME = "rag_db"
SESSION_HISTORIES_COLLECTION = "session_histories"
BUFFER_SESSION_COLLECTION = "buffer_session"

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Global variables for MongoDB client and database
client = None
db = None
session_histories_collection = None
buffer_session_collection = None

def debug_environment():
    """Debug function to check MongoDB environment variables"""
    logger.info("=== MongoDB Environment Debug ===")
    logger.info(f"MONGODB_URI: {'Set' if os.getenv('MONGODB_URI') else 'Not set'}")
    logger.info(f"MONGODB_URL: {'Set' if os.getenv('MONGODB_URL') else 'Not set'}")
    
    if os.getenv('MONGODB_URI'):
        uri = os.getenv('MONGODB_URI')
        logger.info(f"MONGODB_URI length: {len(uri)}")
        logger.info(f"MONGODB_URI starts with: {uri[:30]}...")
    
    if os.getenv('MONGODB_URL'):
        url = os.getenv('MONGODB_URL')
        logger.info(f"MONGODB_URL length: {len(url)}")
        logger.info(f"MONGODB_URL starts with: {url[:30]}...")
    
    logger.info(f"All env vars containing 'MONGO': {[k for k in os.environ.keys() if 'MONGO' in k.upper()]}")
    logger.info("================================")

def get_mongodb_client():
    """Initialize MongoDB client with proper SSL configuration"""
    global client, db, session_histories_collection, buffer_session_collection
    
    if client is not None:
        return client, db, session_histories_collection, buffer_session_collection
    
    try:
        if not MONGODB_URL:
            logger.warning("Neither MONGODB_URI nor MONGODB_URL found in environment variables")
            logger.warning(f"Available env vars: {[k for k in os.environ.keys() if 'MONGO' in k.upper()]}")
            return None, None, None, None
            
        logger.info("Attempting to connect to MongoDB...")
        logger.info(f"Using connection string: {MONGODB_URL[:50]}...") # Only show first 50 chars for security
        
        # Method 1: Try with explicit SSL context (most compatible)
        try:
            logger.info("Trying connection with SSL context...")
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            client = MongoClient(
                MONGODB_URL,
                ssl_context=ssl_context,
                serverSelectionTimeoutMS=30000,
                connectTimeoutMS=30000,
                socketTimeoutMS=30000,
                maxPoolSize=10,
                minPoolSize=1,
                maxIdleTimeMS=30000,
                waitQueueTimeoutMS=5000
            )
            
            # Test the connection
            client.admin.command('ping')
            logger.info("MongoDB connected successfully with SSL context!")
            
        except Exception as e:
            logger.warning(f"SSL context method failed: {e}")
            
            # Method 2: Try with TLS options in connection string
            logger.info("Trying connection with TLS options...")
            
            # Clean the connection string and add proper TLS options
            clean_url = MONGODB_URL.split('?')[0]  # Remove existing query params
            
            # Build new connection string with explicit TLS options
            tls_options = [
                "tls=true",
                "tlsAllowInvalidCertificates=true",
                "tlsAllowInvalidHostnames=true",
                "retryWrites=true",
                "w=majority",
                "serverSelectionTimeoutMS=30000",
                "connectTimeoutMS=30000",
                "socketTimeoutMS=30000"
            ]
            
            new_url = f"{clean_url}?{'&'.join(tls_options)}"
            
            client = MongoClient(new_url)
            client.admin.command('ping')
            logger.info("MongoDB connected successfully with TLS options!")
        
        # Initialize database and collections
        db = client[DATABASE_NAME]
        session_histories_collection = db[SESSION_HISTORIES_COLLECTION]
        buffer_session_collection = db[BUFFER_SESSION_COLLECTION]
        
        logger.info(f"Database: {DATABASE_NAME}")
        
        # Try to list collections to verify connection
        try:
            collections = db.list_collection_names()
            logger.info(f"Collections available: {collections}")
        except Exception as e:
            logger.warning(f"Could not list collections: {e}")
        
        return client, db, session_histories_collection, buffer_session_collection
        
    except Exception as e:
        logger.error(f"All MongoDB connection methods failed: {e}")
        
        # Method 3: Try with basic pymongo options (fallback)
        try:
            logger.info("Trying basic connection as fallback...")
            
            # Use the original URL but with different client options
            client = MongoClient(
                MONGODB_URL,
                tls=True,
                tlsAllowInvalidCertificates=True,
                tlsAllowInvalidHostnames=True,
                serverSelectionTimeoutMS=30000,
                connectTimeoutMS=30000,
                socketTimeoutMS=30000
            )
            
            client.admin.command('ping')
            db = client[DATABASE_NAME]
            session_histories_collection = db[SESSION_HISTORIES_COLLECTION]
            buffer_session_collection = db[BUFFER_SESSION_COLLECTION]
            logger.info("MongoDB connected successfully with basic fallback method!")
            return client, db, session_histories_collection, buffer_session_collection
            
        except Exception as e2:
            logger.error(f"Fallback connection also failed: {e2}")
            logger.error("Application will continue without database functionality")
            client = None
            db = None
            session_histories_collection = None
            buffer_session_collection = None
            return None, None, None, None

def ensure_db_connection():
    """Ensure database connection is available"""
    global client, db, session_histories_collection, buffer_session_collection
    
    if db is None or session_histories_collection is None or buffer_session_collection is None:
        client, db, session_histories_collection, buffer_session_collection = get_mongodb_client()
    
    if db is None:
        logger.error("Database connection unavailable")
        raise ConnectionFailure("Database connection unavailable. Please try again later.")
    
    return db, session_histories_collection, buffer_session_collection

def generate_session_id() -> str:
    """
    Generate a new unique session ID.
    
    Returns:
        A string representation of a UUID.
    """
    return str(uuid.uuid4())

def get_buffer_session_history(session_id: str, domain: str) -> List[Dict[str, str]]:
    """
    Retrieve the last 3 conversation interactions for a given session ID from buffer collection.
    
    Args:
        session_id: The unique identifier for the session.
        domain: The domain name.
    
    Returns:
        A list of dictionaries containing the last 3 query-response pairs.
    """
    try:
        _, _, buffer_collection = ensure_db_connection()
        session = buffer_collection.find_one({"session_id": session_id, "domain": domain})
        if session and "history" in session:
            # Return only the last 3 interactions for context
            last_3_history = session["history"][-3:]
            logger.info(f"Retrieved last 3 interactions from buffer history for session {session_id} in domain {domain}")
            return last_3_history
        logger.info(f"No buffer history found for session {session_id} in domain {domain}, initializing empty history")
        return []
    except ConnectionFailure:
        logger.error(f"Database connection error while retrieving buffer session history for {session_id} in domain {domain}")
        return []
    except Exception as e:
        logger.error(f"Error retrieving buffer session history for {session_id} in domain {domain}: {e}")
        return []

def get_full_buffer_session_history(session_id: str, domain: str) -> List[Dict[str, str]]:
    """
    Retrieve the complete conversation history for a given session ID from buffer collection.
    
    Args:
        session_id: The unique identifier for the session.
        domain: The domain name.
    
    Returns:
        A list of dictionaries containing all query-response pairs.
    """
    try:
        _, _, buffer_collection = ensure_db_connection()
        session = buffer_collection.find_one({"session_id": session_id, "domain": domain})
        if session and "history" in session:
            logger.info(f"Retrieved full buffer history for session {session_id} in domain {domain} ({len(session['history'])} interactions)")
            return session["history"]
        logger.info(f"No buffer history found for session {session_id} in domain {domain}")
        return []
    except ConnectionFailure:
        logger.error(f"Database connection error while retrieving full buffer session history for {session_id} in domain {domain}")
        return []
    except Exception as e:
        logger.error(f"Error retrieving full buffer session history for {session_id} in domain {domain}: {e}")
        return []

def update_buffer_session_history(session_id: str, domain: str, query: str, response: str, max_stored: int = 50):
    """
    Update the buffer session history with a new query-response pair.
    Stores up to max_stored interactions but only uses last 3 for context.
    
    Args:
        session_id: The unique identifier for the session.
        domain: The domain name.
        query: The user's query.
        response: The LLM's response.
        max_stored: Maximum number of interactions to store (default: 50).
    """
    try:
        _, _, buffer_collection = ensure_db_connection()
        session = buffer_collection.find_one({"session_id": session_id, "domain": domain})
        history = session["history"] if session and "history" in session else []
        
        # Add new interaction
        history.append({"query": query, "response": response})
        
        # Keep only the last max_stored interactions to prevent unlimited growth
        if len(history) > max_stored:
            history = history[-max_stored:]
            logger.info(f"Trimmed buffer history to last {max_stored} interactions for session {session_id}")
        
        # Update or insert the session document with last_updated timestamp
        buffer_collection.update_one(
            {"session_id": session_id, "domain": domain},
            {"$set": {
                "history": history,
                "last_updated": datetime.utcnow()
            }},
            upsert=True
        )
        logger.info(f"Updated buffer session history for {session_id} in domain {domain} (total: {len(history)} interactions)")
    except ConnectionFailure:
        logger.error(f"Database connection error while updating buffer session history for {session_id} in domain {domain}")
    except Exception as e:
        logger.error(f"Error updating buffer session history for {session_id} in domain {domain}: {e}")

def get_buffer_session_stats(session_id: str, domain: str) -> Dict[str, int]:
    """
    Get statistics about the buffer session history.
    
    Args:
        session_id: The unique identifier for the session.
        domain: The domain name.
    
    Returns:
        Dictionary containing total interactions and interactions used for context.
    """
    try:
        _, _, buffer_collection = ensure_db_connection()
        session = buffer_collection.find_one({"session_id": session_id, "domain": domain})
        if session and "history" in session:
            total_interactions = len(session["history"])
            context_interactions = min(3, total_interactions)
            return {
                "total_interactions": total_interactions,
                "context_interactions": context_interactions
            }
        return {"total_interactions": 0, "context_interactions": 0}
    except ConnectionFailure:
        logger.error(f"Database connection error while getting buffer session stats for {session_id} in domain {domain}")
        return {"total_interactions": 0, "context_interactions": 0}
    except Exception as e:
        logger.error(f"Error getting buffer session stats for {session_id} in domain {domain}: {e}")
        return {"total_interactions": 0, "context_interactions": 0}

def get_session_history_by_title(title: str, domain: str) -> List[Dict[str, str]]:
    """
    Retrieve the conversation history for a given title from session_histories collection.
    
    Args:
        title: The unique title for the session.
        domain: The domain name.
    
    Returns:
        A list of dictionaries containing query-response pairs.
    """
    try:
        _, histories_collection, _ = ensure_db_connection()
        session = histories_collection.find_one({"title": title, "domain": domain})
        if session and "history" in session:
            logger.info(f"Retrieved session history for title '{title}' in domain {domain}")
            return session["history"]
        logger.info(f"No session history found for title '{title}' in domain {domain}")
        return []
    except ConnectionFailure:
        logger.error(f"Database connection error while retrieving session history for title '{title}' in domain {domain}")
        return []
    except Exception as e:
        logger.error(f"Error retrieving session history for title '{title}' in domain {domain}: {e}")
        return []

def get_last_3_session_history_by_title(title: str, domain: str) -> List[Dict[str, str]]:
    """
    Retrieve the last 3 conversation interactions for a given title from session_histories collection.
    
    Args:
        title: The unique title for the session.
        domain: The domain name.
    
    Returns:
        A list of dictionaries containing the last 3 query-response pairs.
    """
    try:
        _, histories_collection, _ = ensure_db_connection()
        session = histories_collection.find_one({"title": title, "domain": domain})
        if session and "history" in session:
            # Return only the last 3 interactions for context
            last_3_history = session["history"][-3:]
            logger.info(f"Retrieved last 3 interactions from session history for title '{title}' in domain {domain}")
            return last_3_history
        logger.info(f"No session history found for title '{title}' in domain {domain}")
        return []
    except ConnectionFailure:
        logger.error(f"Database connection error while retrieving last 3 session history for title '{title}' in domain {domain}")
        return []
    except Exception as e:
        logger.error(f"Error retrieving last 3 session history for title '{title}' in domain {domain}: {e}")
        return []

def update_session_history_by_title(title: str, domain: str, query: str, response: str):
    """
    Update the session history with a new query-response pair for a given title.
    
    Args:
        title: The unique title for the session.
        domain: The domain name.
        query: The user's query.
        response: The LLM's response.
    """
    try:
        _, histories_collection, _ = ensure_db_connection()
        session = histories_collection.find_one({"title": title, "domain": domain})
        if session:
            history = session.get("history", [])
            # Add new interaction
            history.append({"query": query, "response": response})
            
            # Update the session document
            histories_collection.update_one(
                {"title": title, "domain": domain},
                {"$set": {"history": history}}
            )
            logger.info(f"Updated session history for title '{title}' in domain {domain} (total: {len(history)} interactions)")
        else:
            logger.warning(f"No session found with title '{title}' in domain {domain} to update history")
    except ConnectionFailure:
        logger.error(f"Database connection error while updating session history for title '{title}' in domain {domain}")
    except Exception as e:
        logger.error(f"Error updating session history for title '{title}' in domain {domain}: {e}")

def get_all_sessions(domain: str) -> List[Dict[str, str]]:
    """
    Retrieve all session titles for a specific domain from session_histories collection.
    
    Args:
        domain: The domain name.
    
    Returns:
        A list of dictionaries containing title and domain.
    """
    try:
        _, histories_collection, _ = ensure_db_connection()
        sessions = histories_collection.find({"domain": domain}, {"title": 1, "domain": 1, "_id": 0})
        result = [{"title": session["title"], "domain": session["domain"]} for session in sessions]
        logger.info(f"Retrieved {len(result)} sessions for domain {domain}")
        return result
    except ConnectionFailure:
        logger.error(f"Database connection error while retrieving all sessions for domain {domain}")
        return []
    except Exception as e:
        logger.error(f"Error retrieving all sessions for domain {domain}: {e}")
        return []

def get_session_conversation_by_title(title: str, domain: str) -> Dict[str, any]:
    """
    Retrieve the full conversation for a specific title from session_histories collection.
    
    Args:
        title: The unique title for the session.
        domain: The domain name.
    
    Returns:
        A dictionary containing title, domain, and history.
    """
    try:
        _, histories_collection, _ = ensure_db_connection()
        session = histories_collection.find_one({"title": title, "domain": domain}, {"_id": 0})
        if session:
            logger.info(f"Retrieved conversation for title '{title}' in domain {domain}")
            return {
                "title": session["title"],
                "domain": session["domain"],
                "history": session.get("history", [])
            }
        logger.warning(f"No conversation found for title '{title}' in domain {domain}")
        return {"title": title, "domain": domain, "history": []}
    except ConnectionFailure:
        logger.error(f"Database connection error while retrieving conversation for title '{title}' in domain {domain}")
        return {"title": title, "domain": domain, "history": []}
    except Exception as e:
        logger.error(f"Error retrieving conversation for title '{title}' in domain {domain}: {e}")
        return {"title": title, "domain": domain, "history": []}

def save_buffer_to_session_history(session_id: str, domain: str, title: str) -> bool:
    """
    Save buffer session to session_histories collection with a unique title.
    Uses the full buffer history, not just the last 3 interactions.
    
    Args:
        session_id: The session ID from buffer collection.
        domain: The domain name.
        title: The unique title for the session.
    
    Returns:
        Boolean indicating success.
    """
    try:
        _, histories_collection, buffer_collection = ensure_db_connection()
        
        # Check if title already exists in the domain
        existing_session = histories_collection.find_one({"title": title, "domain": domain})
        if existing_session:
            logger.warning(f"Title '{title}' already exists in domain {domain}")
            return False
        
        # Get buffer session with full history
        buffer_session = buffer_collection.find_one({"session_id": session_id, "domain": domain})
        if not buffer_session:
            logger.warning(f"No buffer session found for {session_id} in domain {domain}")
            return False
        
        full_history = buffer_session.get("history", [])
        
        # Save to session_histories with complete history
        histories_collection.insert_one({
            "title": title,
            "domain": domain,
            "history": full_history
        })
        
        # Remove from buffer
        buffer_collection.delete_one({"session_id": session_id, "domain": domain})
        
        logger.info(f"Saved buffer session {session_id} as '{title}' in domain {domain} with {len(full_history)} interactions")
        return True
    except ConnectionFailure:
        logger.error(f"Database connection error while saving buffer session {session_id} as '{title}' in domain {domain}")
        return False
    except Exception as e:
        logger.error(f"Error saving buffer session {session_id} as '{title}' in domain {domain}: {e}")
        return False

def update_session_title(old_title: str, new_title: str, domain: str) -> bool:
    """
    Update the title of an existing session in session_histories collection.
    
    Args:
        old_title: The current title.
        new_title: The new title.
        domain: The domain name.
    
    Returns:
        Boolean indicating success.
    """
    try:
        _, histories_collection, _ = ensure_db_connection()
        
        # Check if new title already exists
        existing_session = histories_collection.find_one({"title": new_title, "domain": domain})
        if existing_session:
            logger.warning(f"New title '{new_title}' already exists in domain {domain}")
            return False
        
        # Update the title
        result = histories_collection.update_one(
            {"title": old_title, "domain": domain},
            {"$set": {"title": new_title}}
        )
        
        if result.matched_count > 0:
            logger.info(f"Updated title from '{old_title}' to '{new_title}' in domain {domain}")
            return True
        else:
            logger.warning(f"No session found with title '{old_title}' in domain {domain}")
            return False
    except ConnectionFailure:
        logger.error(f"Database connection error while updating title from '{old_title}' to '{new_title}' in domain {domain}")
        return False
    except Exception as e:
        logger.error(f"Error updating title from '{old_title}' to '{new_title}' in domain {domain}: {e}")
        return False

def delete_session_by_title(title: str, domain: str) -> bool:
    """
    Delete a session from session_histories collection by title.
    
    Args:
        title: The title of the session to delete.
        domain: The domain name.
    
    Returns:
        Boolean indicating success.
    """
    try:
        _, histories_collection, _ = ensure_db_connection()
        result = histories_collection.delete_one({"title": title, "domain": domain})
        if result.deleted_count > 0:
            logger.info(f"Deleted session with title '{title}' from domain {domain}")
            return True
        else:
            logger.warning(f"No session found with title '{title}' in domain {domain}")
            return False
    except ConnectionFailure:
        logger.error(f"Database connection error while deleting session with title '{title}' from domain {domain}")
        return False
    except Exception as e:
        logger.error(f"Error deleting session with title '{title}' in domain {domain}: {e}")
        return False

def check_title_exists(title: str, domain: str) -> bool:
    """
    Check if a title already exists in a domain.
    
    Args:
        title: The title to check.
        domain: The domain name.
    
    Returns:
        Boolean indicating if title exists.
    """
    try:
        _, histories_collection, _ = ensure_db_connection()
        session = histories_collection.find_one({"title": title, "domain": domain})
        return session is not None
    except ConnectionFailure:
        logger.error(f"Database connection error while checking if title '{title}' exists in domain {domain}")
        return False
    except Exception as e:
        logger.error(f"Error checking if title '{title}' exists in domain {domain}: {e}")
        return False

def clear_old_buffer_sessions(domain: str, days_old: int = 30) -> int:
    """
    Clear buffer sessions older than specified days to prevent database bloat.
    
    Args:
        domain: The domain name.
        days_old: Number of days after which to consider sessions old (default: 30).
    
    Returns:
        Number of deleted sessions.
    """
    try:
        _, _, buffer_collection = ensure_db_connection()
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        # Delete sessions older than cutoff_date using the last_updated field
        result = buffer_collection.delete_many({
            "domain": domain,
            "last_updated": {"$lt": cutoff_date}
        })
        
        deleted_count = result.deleted_count
        logger.info(f"Cleared {deleted_count} old buffer sessions from domain {domain}")
        return deleted_count
    except ConnectionFailure:
        logger.error(f"Database connection error while clearing old buffer sessions for domain {domain}")
        return 0
    except Exception as e:
        logger.error(f"Error clearing old buffer sessions for domain {domain}: {e}")
        return 0

def get_database_health() -> Dict[str, any]:
    """
    Check database connection health and return status information.
    
    Returns:
        Dictionary containing database status information.
    """
    try:
        db_conn, histories_collection, buffer_collection = ensure_db_connection()
        
        # Test connection with ping
        client.admin.command('ping')
        
        # Get collection counts
        histories_count = histories_collection.count_documents({})
        buffer_count = buffer_collection.count_documents({})
        
        return {
            "status": "healthy",
            "database": "connected",
            "database_name": DATABASE_NAME,
            "collections": {
                "session_histories": histories_count,
                "buffer_session": buffer_count
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except ConnectionFailure:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": "Database connection unavailable",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Initialize MongoDB connection when module loads
logger.info("Initializing MongoDB connection for history manager...")
debug_environment()  # Debug environment variables
get_mongodb_client()