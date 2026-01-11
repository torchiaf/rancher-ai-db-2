import asyncio
import json
import logging

from psycopg import AsyncConnection
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from src.agent_client import get_chat_name
from src.config import get_db_url, LOG_LEVEL, LOOP_R_CHATS_INTERVAL, LOOP_R_MESSAGES_INTERVAL

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

async def setup_database() -> AsyncConnection:
    """
    Connect to database and create tables/triggers.
    """
    conn_info = get_db_url()
    conn = await AsyncConnection.connect(conn_info, autocommit=True)
    
    logger.info(f"Connected to database at {conn_info}")
    
    async with conn.cursor() as cur:
        # Create normalization thread queue table
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS r_normalization_thread_queue (
                thread_id UUID NOT NULL,
                user_id VARCHAR(10) NOT NULL,
                active BOOLEAN DEFAULT TRUE,
                processed BOOLEAN DEFAULT FALSE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (thread_id, user_id)
            );
        """)
        logger.info("Created/verified r_normalization_thread_queue table")
        
        # Create normalization request queue table
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS r_normalization_request_queue (
                thread_id UUID NOT NULL,
                request_id UUID NOT NULL,
                processed BOOLEAN DEFAULT FALSE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (thread_id, request_id)
            );
        """)
        logger.info("Created/verified r_normalization_request_queue table")
        
        # Create chats table
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS r_chats (
                chat_id VARCHAR(255) NOT NULL,
                user_id VARCHAR(10) NOT NULL,
                active BOOLEAN DEFAULT TRUE,
                name VARCHAR(255) DEFAULT '',
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                PRIMARY KEY (chat_id, user_id)
            )
        """)
        logger.info("Created/verified r_chats table")
        
        # Create messages table
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS r_messages (
                chat_id VARCHAR(255) NOT NULL,
                request_id VARCHAR(255) NOT NULL,
                user_message TEXT,
                mcp_responses TEXT,
                llm_response TEXT,
                context TEXT,
                tags TEXT[],
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                PRIMARY KEY (chat_id, request_id)
            )
        """)
        logger.info("Created/verified r_messages table")
    
    return conn

async def get_chat_messages(chat_id: str, max_count: int = 5) -> list[str]:
    """
    Retrieve the latest max_messages messages for a given chat_id.
    """
    db_url = get_db_url()
    conn = await AsyncConnection.connect(db_url)
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT user_message
                FROM r_messages
                WHERE chat_id = %s
                AND (tags IS NULL OR tags::text NOT LIKE %s)
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (chat_id, '%welcome%', max_count)
            )
            rows = await cur.fetchall()
            return [row[0] for row in rows if row[0]]
    finally:
        await conn.close()

async def sync_r_chats(
    cur,
    thread_id: str,
    user_id: str,
    active: bool,
) -> None:
    """
    Fetch chat info and sync to r_chats table.
    If chat is being set to inactive and has no name, generate one from ws/summary.
    """

    name = ""

    # If chat is being deactivated and has no name, generate one
    if not active:
        await cur.execute(
            "SELECT active, name, created_at FROM r_chats WHERE chat_id = %s AND user_id = %s",
            (thread_id, user_id)
        )
        existing = await cur.fetchone()
        
        if existing and existing[0] == True and existing[1] == "":
            # The chat is about to be deactivated and has no name
            logger.info(f"Assigning name to chat {thread_id}")
            
            messages = await get_chat_messages(thread_id)
            if len(messages) > 0:
                try:
                    name = await get_chat_name(thread_id, messages) or ""
                except Exception as e:
                    logger.debug(f"Failed to obtain name from websocket summary service: {e}")
                    name = existing[2].strftime("Chat %Y-%m-%d %H:%M:%S")
            else:
                logger.debug(f"No messages found for chat_id: {thread_id}, using empty name")

            logger.info(f"Generated name: '{name}'")
        elif existing:
            name = existing[1]
    
    await cur.execute(
        """
        INSERT INTO r_chats (chat_id, user_id, active, name, created_at, updated_at)
        VALUES (%s, %s, %s, %s, NOW(), NOW())
        ON CONFLICT (chat_id, user_id) DO UPDATE SET
        active = EXCLUDED.active,
        name = CASE WHEN EXCLUDED.name != '' THEN EXCLUDED.name ELSE r_chats.name END,
        updated_at = NOW()
        """,
        (thread_id, user_id, active, name))
    logger.info(f"Synced chat for thread_id: {thread_id}, user_id: {user_id}, name: '{name}'")

async def sync_r_messages(
    cur,
    saver: AsyncPostgresSaver,
    thread_id: str,
    request_id: str,
) -> None:
    """
    Fetch messages from LangGraph checkpoints and sync to r_messages table.
    """
    
    # Fetch checkpoint data using saver which handles deserialization
    config = {"configurable": {"thread_id": thread_id, "request_id": request_id}}
    checkpoint_tuple = await saver.aget_tuple(config)
    
    if not checkpoint_tuple:
        logger.warning(f"No checkpoint found for thread: {thread_id}")
        return
    
    checkpoint_data = checkpoint_tuple.checkpoint
    
    # Walk through checkpoint chain to find all messages
    all_messages = []
    current_config = {"configurable": {"thread_id": thread_id, "request_id": request_id}}
    visited = set()
    
    while True:
        if str(current_config) in visited:
            break
        visited.add(str(current_config))
        
        current_tuple = await saver.aget_tuple(current_config)
        if not current_tuple:
            break
        
        current_data = current_tuple.checkpoint
        msgs = current_data.get("channel_values", {}).get("messages", [])
        all_messages = msgs + all_messages
        
        if not current_tuple.parent_config:
            break
        current_config = current_tuple.parent_config
    
    messages = all_messages
    
    # Extract fields from channel_values
    channel_values = checkpoint_data.get("channel_values", {})
    context = channel_values.get("context", "")
    prompt = channel_values.get("prompt", "")
    mcp_responses_list = channel_values.get("mcp_responses", [])
    
    # Convert mcp_responses list to string
    if isinstance(mcp_responses_list, list):
        mcp_res = " ".join(mcp_responses_list) if mcp_responses_list else ""
    else:
        mcp_res = str(mcp_responses_list) if mcp_responses_list else ""
    
    # Convert to strings if they're dicts
    if isinstance(context, dict):
        context = json.dumps(context)
    else:
        context = str(context) if context else ""
    
    if isinstance(prompt, dict):
        prompt = json.dumps(prompt)
    else:
        prompt = str(prompt) if prompt else ""
    
    # Extract tags from checkpoint metadata
    metadata = checkpoint_tuple.metadata or {}
    tags = (metadata.get("tags") or 
            channel_values.get("tags"))
    
    # Extract message types by looking at message type attribute
    llm_msg = ""
    
    for msg in messages:
        msg_type = getattr(msg, "type", None)
        msg_content = getattr(msg, "content", "")
        
        if msg_type == "ai":
            llm_msg = msg_content
    
    if not prompt:
        logger.warning(f"No prompt/user message found in checkpoint for request_id {request_id}")

    await cur.execute(
        """
        INSERT INTO r_messages
        (request_id, chat_id, context, tags, user_message, mcp_responses, llm_response, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW()) ON CONFLICT (chat_id, request_id) DO NOTHING
        """,
        (request_id, thread_id, context, tags, prompt, mcp_res, llm_msg))
    logger.info(f"Synced message for thread_id: {thread_id}, request_id: {request_id}")

async def run_supervisor() -> None:
    """
    Run the supervisor - loop over unprocessed threads and sync data.
    """
    conn = await setup_database()
    conn_info = get_db_url()
    async def chat_sync_loop():
        """Loop for syncing chats from thread queue."""
        logger.info("Starting chat sync loop")
        while True:
            try:
                async with conn.cursor() as cur:
                    # Load unprocessed thread_ids
                    await cur.execute("""
                        SELECT thread_id, user_id, active, updated_at
                        FROM r_normalization_thread_queue
                        WHERE processed = FALSE
                        ORDER BY updated_at ASC
                    """)
                    rows = await cur.fetchall()
                    thread_ids = [(row[0], row[1], row[2]) for row in rows]

                    # Process all threads in the same cursor
                    for thread_id, user_id, active in thread_ids:
                        await sync_r_chats(cur, str(thread_id), user_id, active)

                        # Mark as processed
                        await cur.execute("""
                            UPDATE r_normalization_thread_queue
                            SET processed = TRUE
                            WHERE thread_id = %s
                        """, (thread_id,))
                
                await asyncio.sleep(LOOP_R_CHATS_INTERVAL)

            except Exception as e:
                logger.error(f"Error in chat sync loop: {e}")
                await asyncio.sleep(LOOP_R_CHATS_INTERVAL*2)

    async def message_sync_loop():
        """Loop for syncing messages from request queue."""
        logger.info("Starting message sync loop")
        while True:
            try:
                async with conn.cursor() as cur:
                    # Load unprocessed requests
                    await cur.execute("""
                        SELECT thread_id, request_id, updated_at
                        FROM r_normalization_request_queue
                        WHERE processed = FALSE
                        ORDER BY updated_at ASC
                    """)
                    rows = await cur.fetchall()
                    request_pairs = [(row[0], row[1]) for row in rows]

                    # Process all requests in the same cursor
                    for thread_id, request_id in request_pairs:
                        async with AsyncPostgresSaver.from_conn_string(conn_info) as saver:
                            await sync_r_messages(cur, saver, str(thread_id), str(request_id))

                        # Mark as processed
                        await cur.execute("""
                            UPDATE r_normalization_request_queue
                            SET processed = TRUE
                            WHERE thread_id = %s AND request_id = %s
                        """, (thread_id, request_id))

                await asyncio.sleep(LOOP_R_MESSAGES_INTERVAL)

            except Exception as e:
                logger.error(f"Error in message sync loop: {e}")
                await asyncio.sleep(LOOP_R_MESSAGES_INTERVAL*2)

    # Run both loops concurrently
    await asyncio.gather(chat_sync_loop(), message_sync_loop())

if __name__ == "__main__":
    asyncio.run(run_supervisor())
