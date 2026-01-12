import asyncio
import json
import logging

from psycopg import AsyncConnection
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.agent_client import get_chat_name
from src.config import get_db_url, LOG_LEVEL, LOOP_R_CLEAN_SYNCED_CHECKPOINTS_INTERVAL, LOOP_R_MESSAGES_INTERVAL
from src.utils import (
    extract_messages,
    extract_tags,
    extract_context,
    extract_mcp_responses,
    convert_data_to_strings,
)

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

async def setup_database() -> AsyncConnection:
    """
    Connect to database and create tables/triggers.
    """
    conn_info = get_db_url()
    conn = await AsyncConnection.connect(conn_info, autocommit=True)
    
    logger.info(f"Connected to database at {conn_info}")

    # Initialize LangGraph schema
    try:
        async with AsyncPostgresSaver.from_conn_string(get_db_url()) as checkpointer:
            await checkpointer.setup()
        logger.info("LangGraph schema initialized")
    except Exception as e:
        logger.error(f"Failed to initialize LangGraph schema: {e}", exc_info=True)

    async with conn.cursor() as cur:
        # Create checkpoint tracking table
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS r_synced_checkpoints (
                checkpoint_id VARCHAR(255) NOT NULL PRIMARY KEY,
                request_id VARCHAR(255) NOT NULL,
                synced_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)
        logger.info("Created/verified r_synced_checkpoints table")

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
    Assign a name to inactive chats
    """

async def sync_r_messages(
    cur,
    thread_id: str,
    request_id: str,
    checkpoint_id: str,
    checkpoint_raw,
    metadata_raw,
) -> None:
    """
    Fetch messages from checkpoint_writes (msgpack+pickle) and sync to r_messages table.
    checkpoint_raw and metadata_raw are passed from the loop query to avoid a second SELECT.
    """
    
    logger.info(f"[sync_r_messages] Processing checkpoint_id={checkpoint_id}")
    
    try:
        checkpoint_data = checkpoint_raw if isinstance(checkpoint_raw, dict) else json.loads(checkpoint_raw)
        metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else (metadata_raw or {})
        step = metadata.get('step', '?')
        logger.debug(f"[sync_r_messages] Loaded checkpoint: checkpoint_id={checkpoint_id}, step={step}")
    except Exception as e:
        logger.error(f"Failed to deserialize checkpoint: {e}")
        return
    
    channel_values = checkpoint_data.get("channel_values", {})
    prompt = channel_values.get("prompt", "")
    logger.debug(f"[sync_r_messages] Channel values keys: {list(channel_values.keys())}")

    logger.debug(f"[sync_r_messages] Querying checkpoint_writes for checkpoint_id={checkpoint_id}")
    await cur.execute("""
        SELECT cw.channel, cw.blob
        FROM checkpoint_writes cw
        WHERE cw.checkpoint_id = %s
        ORDER BY cw.channel
    """, (checkpoint_id,))
    
    all_rows = await cur.fetchall()
    logger.info(f"[sync_r_messages] Found {len(all_rows)} checkpoint_writes rows for checkpoint_id={checkpoint_id}")
    
    messages_blob = None
    tags_blob = None
    context_blob = None
    mcp_blob = None
    
    for channel, blob in all_rows:
        logger.info(f"[sync_r_messages] Processing channel='{channel}', blob_size={len(blob) if blob else 0} bytes")
        
        if blob:
            try:
                import msgpack
                deserialized = msgpack.unpackb(blob, raw=False)
                logger.info(f"[sync_r_messages] Deserialized '{channel}': type={type(deserialized).__name__}, content_preview={str(deserialized)[:200]}")
            except Exception as e:
                logger.error(f"[sync_r_messages] Failed to deserialize '{channel}': {e}")

        if channel == 'messages':
            messages_blob = blob
        elif channel == 'tags':
            tags_blob = blob
        elif channel == 'context':
            context_blob = blob
        elif channel == 'mcp_responses':
            mcp_blob = blob
    
    logger.debug(f"[sync_r_messages] Blob assignment: messages={messages_blob is not None}, tags={tags_blob is not None}, context={context_blob is not None}, mcp={mcp_blob is not None}")

    logger.debug(f"[sync_r_messages] Extracting messages...")
    user_msg, llm_msg = extract_messages(messages_blob)
    logger.debug(f"[sync_r_messages] Extracted messages: user_msg_len={len(user_msg) if user_msg else 0}, llm_msg_len={len(llm_msg) if llm_msg else 0}")
    logger.info(f"[sync_r_messages] llm response: {repr(llm_msg[:200] if llm_msg else 'EMPTY')}")
    
    logger.debug(f"[sync_r_messages] Extracting tags...")
    tags = extract_tags(tags_blob)
    logger.debug(f"[sync_r_messages] Extracted tags: {tags}")
    
    logger.debug(f"[sync_r_messages] Extracting context...")
    context = extract_context(context_blob)
    logger.debug(f"[sync_r_messages] Extracted context keys: {list(context.keys()) if context else 'empty'}")
    
    logger.debug(f"[sync_r_messages] Extracting mcp_responses...")
    mcp_responses_list = extract_mcp_responses(mcp_blob)
    logger.debug(f"[sync_r_messages] Extracted mcp_responses: {len(mcp_responses_list)} items")
    
    logger.debug(f"[sync_r_messages] Prompt from checkpoint: {len(str(prompt))} chars")
    
    # Fallback to prompt if no user message found
    if not user_msg and prompt:
        user_msg = prompt
        logger.debug(f"[sync_r_messages] No user_msg extracted, using prompt fallback: {len(user_msg)} chars")
    
    # Convert data to final string formats
    logger.debug(f"[sync_r_messages] Converting data to database formats...")
    context_str, tags_list, mcp_res = convert_data_to_strings(
        user_msg, llm_msg, mcp_responses_list, context, tags
    )
    logger.debug(f"[sync_r_messages] Conversion complete:")
    logger.debug(f"  - context_str: {len(context_str)} chars")
    logger.debug(f"  - tags_list: {tags_list}")
    logger.debug(f"  - mcp_res: {len(mcp_res)} chars")
    
    logger.info(f"[sync_r_messages] FINAL RESULT: step={step}, user_len={len(user_msg)}, llm_len={len(llm_msg)}, tags={tags_list}, context_len={len(context_str)}, mcp_len={len(mcp_res)}")

    # Convert empty strings to None so COALESCE treats them as NULL in the database
    context_val = context_str if context_str else None
    user_val = user_msg if user_msg else None
    llm_val = llm_msg if llm_msg else None
    mcp_val = mcp_res if mcp_res else None

    await cur.execute(
        """
        INSERT INTO r_messages
        (request_id, chat_id, context, tags, user_message, mcp_responses, llm_response, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
        ON CONFLICT (chat_id, request_id) DO UPDATE SET
        context = COALESCE(EXCLUDED.context, r_messages.context),
        tags = COALESCE(EXCLUDED.tags, r_messages.tags),
        user_message = COALESCE(EXCLUDED.user_message, r_messages.user_message),
        mcp_responses = COALESCE(EXCLUDED.mcp_responses, r_messages.mcp_responses),
        llm_response = COALESCE(EXCLUDED.llm_response, r_messages.llm_response),
        updated_at = NOW()
        """,
        (request_id, thread_id, context_val, tags_list, user_val, mcp_val, llm_val))

async def run_supervisor() -> None:
    """
    Run the supervisor - loop over unprocessed threads and sync data.
    """
    conn = await setup_database()
    conn_info = get_db_url()

    # async def chat_sync_loop():
    #     """
    #     Loop for syncing chats from thread queue.
    #     Picks up both unprocessed rows and recently updated rows.
    #     """
    #     logger.info("Starting chat sync loop")
        
    #     while True:
    #         try:
    #             async with conn.cursor() as cur:

    #             # TODO assign a name to history chats if missing
                
    #             await asyncio.sleep(LOOP_R_CHATS_INTERVAL)

    #         except Exception as e:
    #             logger.error(f"Error in chat sync loop: {e}")
    #             await asyncio.sleep(LOOP_R_CHATS_INTERVAL*2)

    async def cleanup_synced_checkpoints_loop():
        """
        Periodically clean up r_synced_checkpoints for chats that no longer exist.
        """
        logger.info("Starting cleanup loop for synced checkpoints")
        
        while True:
            try:
                async with conn.cursor() as cur:
                    # Delete checkpoints for chats that don't exist in r_chats
                    await cur.execute("""
                        DELETE FROM r_synced_checkpoints rsc
                        WHERE rsc.request_id IN (
                            SELECT DISTINCT rm.request_id
                            FROM r_messages rm
                            WHERE rm.chat_id NOT IN (SELECT chat_id FROM r_chats)
                        )
                    """)
                    deleted = cur.rowcount
                    if deleted > 0:
                        logger.debug(f"[cleanup_synced_checkpoints_loop] Deleted {deleted} stale checkpoint tracking records")
                    
                await asyncio.sleep(LOOP_R_CLEAN_SYNCED_CHECKPOINTS_INTERVAL)
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}", exc_info=True)
                await asyncio.sleep(LOOP_R_CLEAN_SYNCED_CHECKPOINTS_INTERVAL * 2)

    async def message_sync_loop():
        """
        Loop for syncing messages from LangGraph checkpoints table.
        Tracks processed checkpoints to avoid re-processing.
        """
        logger.info("Starting message sync loop")
        
        while True:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT DISTINCT
                            c.thread_id,
                            c.checkpoint_id,
                            (c.metadata->>'request_id') as request_id,
                            c.checkpoint,
                            c.metadata
                        FROM checkpoint_writes cw
                        JOIN checkpoints c ON cw.checkpoint_id = c.checkpoint_id
                        WHERE cw.checkpoint_id NOT IN (SELECT checkpoint_id FROM r_synced_checkpoints)
                          AND c.metadata->>'request_id' IS NOT NULL
                        ORDER BY c.checkpoint_id
                        """
                    )
                    rows = await cur.fetchall()
                    
                    if rows:
                        logger.debug(f"[message_sync_loop] Found {len(rows)} new checkpoints in checkpoint_writes")
                    else:
                        logger.debug(f"[message_sync_loop] Found 0 new checkpoints")
                    
                    # Process each checkpoint found
                    for thread_id, checkpoint_id, request_id, checkpoint_raw, metadata_raw in rows:

                        logger.info(f"[message_sync_loop] Processing new checkpoint: {checkpoint_id}, request_id={request_id}")
                        await sync_r_messages(cur, str(thread_id), str(request_id), str(checkpoint_id), checkpoint_raw, metadata_raw)
                        
                        # Mark as synced
                        await cur.execute(
                            """
                            INSERT INTO r_synced_checkpoints (checkpoint_id, request_id, synced_at)
                            VALUES (%s, %s, NOW())
                            ON CONFLICT (checkpoint_id) DO NOTHING
                            """,
                            (str(checkpoint_id), str(request_id))
                        )
                        logger.debug(f"[message_sync_loop] Marked checkpoint as synced: {checkpoint_id}")
                
                await asyncio.sleep(LOOP_R_MESSAGES_INTERVAL)

            except Exception as e:
                logger.error(f"Error in message sync loop: {e}", exc_info=True)
                await asyncio.sleep(LOOP_R_MESSAGES_INTERVAL*2)

    # Run both loops concurrently
    await asyncio.gather(
        message_sync_loop(),
        cleanup_synced_checkpoints_loop(),
    )

if __name__ == "__main__":
    asyncio.run(run_supervisor())
