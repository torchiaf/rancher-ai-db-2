import json
import logging
import msgpack
import pickle

logger = logging.getLogger(__name__)


def deserialize_msgpack_blob(blob):
    """Deserialize a msgpack blob, handling errors gracefully."""
    if not blob:
        logger.debug("[deserialize_msgpack_blob] blob is None/empty")
        return None
    try:
        data = msgpack.unpackb(blob, raw=False)
        logger.debug(f"[deserialize_msgpack_blob] Successfully unpacked: {type(data)}")
        return data
    except Exception as e:
        logger.error(f"[deserialize_msgpack_blob] Failed to deserialize: {e}")
        return None


def extract_messages(messages_blob):
    """
    Extract and deserialize messages from checkpoint_writes blob.
    Handles multiple formats:
    - New format: Plain dicts with role/content
    - Old format: ExtType wrapping with type/human
    - Pickled bytes
    
    Returns: (user_msg, llm_msg) tuple
    """
    user_msg = ""
    llm_msg = ""
    
    if not messages_blob:
        logger.debug("[extract_messages] No messages_blob provided")
        return user_msg, llm_msg
    
    try:
        msgpack_data = msgpack.unpackb(messages_blob, raw=False)
        logger.debug(f"[extract_messages] Unpacked msgpack: {type(msgpack_data)}")
        
        messages = []
        
        # Handle different message formats from msgpack
        if isinstance(msgpack_data, list):
            for item in msgpack_data:
                if isinstance(item, dict):
                    # New format: already a dict with role/content
                    messages.append(item)
                    logger.debug(f"[extract_messages] Found dict message: {item.get('role', 'unknown')}")
                elif hasattr(item, 'data'):
                    # Old format: ExtType wrapping pickled data
                    ext_data = item.data
                    logger.debug(f"[extract_messages] Found ExtType, unpacking {len(ext_data)} bytes")
                    try:
                        inner_data = msgpack.unpackb(ext_data, raw=False)
                        if isinstance(inner_data, list):
                            messages.extend(inner_data)
                        elif isinstance(inner_data, bytes):
                            unpickled = pickle.loads(inner_data)
                            if isinstance(unpickled, list):
                                messages.extend(unpickled)
                            else:
                                messages.append(unpickled)
                    except Exception as e:
                        logger.error(f"[extract_messages] Failed to unpack ExtType: {e}")
                elif isinstance(item, (str, bytes)):
                    # Legacy: Pickled bytes
                    try:
                        if isinstance(item, str):
                            item_bytes = item.encode('latin-1')
                        else:
                            item_bytes = item
                        unpickled = pickle.loads(item_bytes)
                        if isinstance(unpickled, list):
                            messages.extend(unpickled)
                        else:
                            messages.append(unpickled)
                    except Exception as e:
                        logger.error(f"[extract_messages] Failed to unpickle: {e}")
        
        logger.debug(f"[extract_messages] Processed {len(messages)} messages")
        
        # Extract user and AI messages - support both old and new formats
        for msg in messages:
            if isinstance(msg, dict):
                # Support both formats: type/human and role/user
                msg_type = msg.get("type") or msg.get("role")
                content = msg.get("content", "")
                
                if msg_type in ("human", "user") and not user_msg:
                    user_msg = content
                    logger.info(f"[extract_messages] Found user message: {len(content)} chars")
                elif msg_type in ("ai", "assistant") and not llm_msg:
                    llm_msg = content
                    logger.info(f"[extract_messages] Found AI message: {len(content)} chars")
    
    except Exception as e:
        logger.error(f"[extract_messages] Failed to deserialize messages: {e}", exc_info=True)
    
    return user_msg, llm_msg


def extract_tags(tags_blob):
    """Extract and deserialize tags from blob."""
    if not tags_blob:
        return []
    
    tags_data = deserialize_msgpack_blob(tags_blob)
    if isinstance(tags_data, list):
        logger.info(f"[extract_tags] Extracted tags: {tags_data}")
        return tags_data
    else:
        logger.warning(f"[extract_tags] tags_data is not a list: {type(tags_data)}")
        return []


def extract_context(context_blob):
    """Extract and deserialize context from blob."""
    if not context_blob:
        return {}
    
    context_data = deserialize_msgpack_blob(context_blob)
    if isinstance(context_data, dict):
        logger.info(f"[extract_context] Extracted context keys: {list(context_data.keys())}")
        return context_data
    else:
        logger.warning(f"[extract_context] context_data is not a dict: {type(context_data)}")
        return {}


def extract_mcp_responses(mcp_blob):
    """Extract and deserialize mcp_responses from blob."""
    if not mcp_blob:
        return []
    
    mcp_data = deserialize_msgpack_blob(mcp_blob)
    if isinstance(mcp_data, list):
        logger.info(f"[extract_mcp_responses] Extracted {len(mcp_data)} items")
        return mcp_data
    else:
        logger.warning(f"[extract_mcp_responses] mcp_data is not a list: {type(mcp_data)}")
        return []


def convert_data_to_strings(user_msg, llm_msg, mcp_responses_list, context, tags):
    """
    Convert extracted data to final string formats for database storage.
    
    Returns: (context_str, tags_list, mcp_res) tuple
    """
    # Convert mcp_responses to string
    if isinstance(mcp_responses_list, list):
        mcp_res = " ".join(str(m) for m in mcp_responses_list) if mcp_responses_list else ""
    else:
        mcp_res = str(mcp_responses_list) if mcp_responses_list else ""
    
    # Convert context to JSON string
    if isinstance(context, dict):
        context_str = json.dumps(context) if context else ""
    else:
        context_str = str(context) if context else ""
    
    # tags should be a list for the PostgreSQL array column
    if not isinstance(tags, list):
        tags_list = [str(tags)] if tags else None
    else:
        tags_list = tags if tags else None
    
    return context_str, tags_list, mcp_res
