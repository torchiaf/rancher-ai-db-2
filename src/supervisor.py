import asyncio
import json
import logging

from psycopg import AsyncConnection
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

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
    
    # Initialize LangGraph schema
    try:
        async with AsyncPostgresSaver.from_conn_string(get_db_url()) as checkpointer:
            await checkpointer.setup()
        logger.info("LangGraph schema initialized")
    except Exception as e:
        logger.error(f"Failed to initialize LangGraph schema: {e}", exc_info=True)
    
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

async def update_chat_name_async(thread_id: str, user_id: str) -> None:
    """
    Fire-and-forget task: Generate chat name and update r_chats table.
    Called when a chat is deactivated.
    """
    try:
        messages = await get_chat_messages(thread_id)
        if not messages:
            logger.debug(f"No messages found for chat {thread_id}, skipping name generation")
            return

        generated_name = await get_chat_name(thread_id, messages)

        if generated_name:
            conn_str = get_db_url()
            conn = await AsyncConnection.connect(conn_str, autocommit=True)
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        UPDATE r_chats
                        SET name = %s, updated_at = NOW()
                        WHERE chat_id = %s AND user_id = %s
                        """,
                        (generated_name, thread_id, user_id)
                    )
                    logger.info(f"Updated chat name for deactivated chat {thread_id}: '{generated_name}'")
            finally:
                await conn.close()
    except Exception as e:
        logger.error(f"Error updating chat name for {thread_id}: {e}", exc_info=True)

async def sync_r_chats(
    cur,
    thread_id: str,
    user_id: str,
    active: bool,
) -> None:
    """
    Fetch chat info and sync to r_chats table.
    - If chat is active: set it as active, deactivate other chats
      - If NEW chat: generate default name
      - If EXISTING chat: preserve existing name
    - If chat is inactive: just mark it as inactive, preserve existing name
      - Fire-and-forget: Generate name asynchronously via get_chat_name
    """

    if active:
        # Deactivate other chats for this user
        await cur.execute(
            """
            UPDATE r_chats
            SET active = FALSE, updated_at = NOW()
            WHERE user_id = %s AND chat_id != %s
            """,
            (user_id, thread_id)
        )
        logger.debug(f"Deactivated other chats for user {user_id}")
        
    await cur.execute(
        "SELECT name FROM r_chats WHERE chat_id = %s AND user_id = %s",
        (thread_id, user_id)
    )
    existing = await cur.fetchone()

    if existing and existing[0]:
        name = existing[0]
        logger.debug(f"Existing chat found for thread_id: {thread_id}, preserving name: '{name}'")
    else:
        await cur.execute("SELECT NOW()::timestamp")
        now_row = await cur.fetchone()
        if now_row:
            name = now_row[0].strftime("Chat %Y-%m-%d %H:%M:%S")

    await cur.execute(
        """
        INSERT INTO r_chats (chat_id, user_id, active, name, created_at, updated_at)
        VALUES (%s, %s, %s, %s, NOW(), NOW())
        ON CONFLICT (chat_id, user_id) DO UPDATE SET
        active = EXCLUDED.active,
        name = EXCLUDED.name,
        updated_at = NOW()
        """,
        (thread_id, user_id, active, name))
    logger.info(f"Synced chat for thread_id: {thread_id}, user_id: {user_id}, active: {active}, name: '{name}'")

    # TODO
    # if not active:
    #     asyncio.create_task(update_chat_name_async(thread_id, user_id))

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
    
    # Get the ACTUAL request_id from checkpoint metadata (source of truth)
    metadata = checkpoint_tuple.metadata or {}
    actual_request_id = metadata.get("request_id")
    
    if actual_request_id:
        request_id = actual_request_id
        logger.debug(f"[sync_r_messages] Using checkpoint metadata request_id: {request_id}")
    
    # Extract messages directly from this checkpoint (no chain walking)
    # Each checkpoint contains the full message history at that point
    messages = checkpoint_data.get("channel_values", {}).get("messages", [])
    
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
    
    # Extract tags (already have metadata from above)
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
        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
        ON CONFLICT (chat_id, request_id) DO UPDATE SET
        context = EXCLUDED.context,
        tags = EXCLUDED.tags,
        user_message = EXCLUDED.user_message,
        mcp_responses = EXCLUDED.mcp_responses,
        llm_response = EXCLUDED.llm_response,
        updated_at = NOW()
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
        """
        Loop for syncing chats from thread queue.
        Picks up both unprocessed rows and recently updated rows.
        """
        logger.info("Starting chat sync loop")
        last_sync_time = "1970-01-01 00:00:00"
        
        while True:
            try:
                async with conn.cursor() as cur:
                    # Load unprocessed rows OR rows updated since last sync
                    await cur.execute("""
                        SELECT thread_id, user_id, active, updated_at
                        FROM r_normalization_thread_queue
                        WHERE processed = FALSE OR updated_at > %s::timestamp
                        ORDER BY updated_at ASC
                    """, (last_sync_time,))
                    
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
                    
                    # Get current DB time for next iteration
                    await cur.execute("SELECT NOW()::timestamp")
                    last_sync_row = await cur.fetchone()
                    if last_sync_row:
                        last_sync_time = last_sync_row[0]
                
                await asyncio.sleep(LOOP_R_CHATS_INTERVAL)

            except Exception as e:
                logger.error(f"Error in chat sync loop: {e}")
                await asyncio.sleep(LOOP_R_CHATS_INTERVAL*2)

    async def message_sync_loop():
        """
        Loop for syncing messages directly from LangGraph checkpoints table.
        """
        logger.info("Starting message sync loop")
        synced_requests = set()
        
        while True:
            try:
                async with conn.cursor() as cur:
                    # Query the checkpoints table directly for checkpoints with request_id in metadata
                    await cur.execute("""
                        SELECT DISTINCT
                            thread_id,
                            (metadata->>'request_id') as request_id
                        FROM checkpoints
                        WHERE metadata->>'request_id' IS NOT NULL
                    """)
                    rows = await cur.fetchall()
                    
                    # logger.debug(f"[message_sync_loop] Found {len(rows) if rows else 0} checkpoints to process")
                    
                    if rows:
                        for thread_id, request_id in rows:
                            # Skip if already synced in this session
                            request_key = f"{thread_id}:{request_id}"
                            if request_key in synced_requests:
                                # logger.debug(f"[message_sync_loop] Already synced: {request_key}")
                                continue
                            
                            # Sync this message from the checkpoint
                            logger.debug(f"[message_sync_loop] Syncing checkpoint: {request_key}")
                            async with AsyncPostgresSaver.from_conn_string(conn_info) as saver:
                                await sync_r_messages(cur, saver, str(thread_id), str(request_id))
                            
                            synced_requests.add(request_key)
                
                await asyncio.sleep(LOOP_R_MESSAGES_INTERVAL)

            except Exception as e:
                logger.error(f"Error in message sync loop: {e}", exc_info=True)
                await asyncio.sleep(LOOP_R_MESSAGES_INTERVAL*2)

    # Run both loops concurrently
    await asyncio.gather(chat_sync_loop(), message_sync_loop())

if __name__ == "__main__":
    asyncio.run(run_supervisor())
