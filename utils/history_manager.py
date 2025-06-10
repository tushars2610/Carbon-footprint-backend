from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
import os
import uuid
from typing import List, Dict
import logging

# Load environment variables
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
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

# Initialize MongoDB client
try:
    client = MongoClient(MONGODB_URI)
    db = client[DATABASE_NAME]
    session_histories_collection = db[SESSION_HISTORIES_COLLECTION]
    buffer_session_collection = db[BUFFER_SESSION_COLLECTION]
    # Test connection
    client.admin.command('ping')
    logger.info("Successfully connected to MongoDB Atlas")
except ConnectionFailure as e:
    logger.error(f"Failed to connect to MongoDB Atlas: {e}")
    raise

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
        session = buffer_session_collection.find_one({"session_id": session_id, "domain": domain})
        if session and "history" in session:
            # Return only the last 3 interactions for context
            last_3_history = session["history"][-3:]
            logger.info(f"Retrieved last 3 interactions from buffer history for session {session_id} in domain {domain}")
            return last_3_history
        logger.info(f"No buffer history found for session {session_id} in domain {domain}, initializing empty history")
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
        session = buffer_session_collection.find_one({"session_id": session_id, "domain": domain})
        if session and "history" in session:
            logger.info(f"Retrieved full buffer history for session {session_id} in domain {domain} ({len(session['history'])} interactions)")
            return session["history"]
        logger.info(f"No buffer history found for session {session_id} in domain {domain}")
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
        session = buffer_session_collection.find_one({"session_id": session_id, "domain": domain})
        history = session["history"] if session and "history" in session else []
        
        # Add new interaction
        history.append({"query": query, "response": response})
        
        # Keep only the last max_stored interactions to prevent unlimited growth
        if len(history) > max_stored:
            history = history[-max_stored:]
            logger.info(f"Trimmed buffer history to last {max_stored} interactions for session {session_id}")
        
        # Update or insert the session document
        buffer_session_collection.update_one(
            {"session_id": session_id, "domain": domain},
            {"$set": {"history": history}},
            upsert=True
        )
        logger.info(f"Updated buffer session history for {session_id} in domain {domain} (total: {len(history)} interactions)")
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
        session = buffer_session_collection.find_one({"session_id": session_id, "domain": domain})
        if session and "history" in session:
            total_interactions = len(session["history"])
            context_interactions = min(3, total_interactions)
            return {
                "total_interactions": total_interactions,
                "context_interactions": context_interactions
            }
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
        session = session_histories_collection.find_one({"title": title, "domain": domain})
        if session and "history" in session:
            logger.info(f"Retrieved session history for title '{title}' in domain {domain}")
            return session["history"]
        logger.info(f"No session history found for title '{title}' in domain {domain}")
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
        session = session_histories_collection.find_one({"title": title, "domain": domain})
        if session and "history" in session:
            # Return only the last 3 interactions for context
            last_3_history = session["history"][-3:]
            logger.info(f"Retrieved last 3 interactions from session history for title '{title}' in domain {domain}")
            return last_3_history
        logger.info(f"No session history found for title '{title}' in domain {domain}")
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
        session = session_histories_collection.find_one({"title": title, "domain": domain})
        if session:
            history = session.get("history", [])
            # Add new interaction
            history.append({"query": query, "response": response})
            
            # Update the session document
            session_histories_collection.update_one(
                {"title": title, "domain": domain},
                {"$set": {"history": history}}
            )
            logger.info(f"Updated session history for title '{title}' in domain {domain} (total: {len(history)} interactions)")
        else:
            logger.warning(f"No session found with title '{title}' in domain {domain} to update history")
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
        sessions = session_histories_collection.find({"domain": domain}, {"title": 1, "domain": 1, "_id": 0})
        result = [{"title": session["title"], "domain": session["domain"]} for session in sessions]
        logger.info(f"Retrieved {len(result)} sessions for domain {domain}")
        return result
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
        session = session_histories_collection.find_one({"title": title, "domain": domain}, {"_id": 0})
        if session:
            logger.info(f"Retrieved conversation for title '{title}' in domain {domain}")
            return {
                "title": session["title"],
                "domain": session["domain"],
                "history": session.get("history", [])
            }
        logger.warning(f"No conversation found for title '{title}' in domain {domain}")
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
        # Check if title already exists in the domain
        existing_session = session_histories_collection.find_one({"title": title, "domain": domain})
        if existing_session:
            logger.warning(f"Title '{title}' already exists in domain {domain}")
            return False
        
        # Get buffer session with full history
        buffer_session = buffer_session_collection.find_one({"session_id": session_id, "domain": domain})
        if not buffer_session:
            logger.warning(f"No buffer session found for {session_id} in domain {domain}")
            return False
        
        full_history = buffer_session.get("history", [])
        
        # Save to session_histories with complete history
        session_histories_collection.insert_one({
            "title": title,
            "domain": domain,
            "history": full_history
        })
        
        # Remove from buffer
        buffer_session_collection.delete_one({"session_id": session_id, "domain": domain})
        
        logger.info(f"Saved buffer session {session_id} as '{title}' in domain {domain} with {len(full_history)} interactions")
        return True
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
        # Check if new title already exists
        existing_session = session_histories_collection.find_one({"title": new_title, "domain": domain})
        if existing_session:
            logger.warning(f"New title '{new_title}' already exists in domain {domain}")
            return False
        
        # Update the title
        result = session_histories_collection.update_one(
            {"title": old_title, "domain": domain},
            {"$set": {"title": new_title}}
        )
        
        if result.matched_count > 0:
            logger.info(f"Updated title from '{old_title}' to '{new_title}' in domain {domain}")
            return True
        else:
            logger.warning(f"No session found with title '{old_title}' in domain {domain}")
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
        result = session_histories_collection.delete_one({"title": title, "domain": domain})
        if result.deleted_count > 0:
            logger.info(f"Deleted session with title '{title}' from domain {domain}")
            return True
        else:
            logger.warning(f"No session found with title '{title}' in domain {domain}")
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
        session = session_histories_collection.find_one({"title": title, "domain": domain})
        return session is not None
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
        from datetime import datetime, timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        # This assumes you add a 'last_updated' field to buffer sessions
        # You might want to modify update_buffer_session_history to include this
        result = buffer_session_collection.delete_many({
            "domain": domain,
            "last_updated": {"$lt": cutoff_date}
        })
        
        deleted_count = result.deleted_count
        logger.info(f"Cleared {deleted_count} old buffer sessions from domain {domain}")
        return deleted_count
    except Exception as e:
        logger.error(f"Error clearing old buffer sessions for domain {domain}: {e}")
        return 0