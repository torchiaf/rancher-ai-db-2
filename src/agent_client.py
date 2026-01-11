import websockets
import asyncio
import re
import logging

from psycopg import AsyncConnection
from .config import get_db_url, AGENT_WS_SUMMARY_URL, AGENT_WS_TIMEOUT

logger = logging.getLogger(__name__)

async def get_chat_summary(messages: list) -> str:
    try:
        async with websockets.connect(AGENT_WS_SUMMARY_URL) as ws:
            text = "\n" + "\n".join(f"- {str(m)}" for m in messages) + "\n"

            logger.info("Sending summary request with text: '%s'", text)

            await ws.send(text)

            buffer = ""
            end_time = asyncio.get_event_loop().time() + AGENT_WS_TIMEOUT
            while True:
                timeout = max(0.1, end_time - asyncio.get_event_loop().time())
                try:
                    chunk = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    logger.debug("Timeout waiting for more chunks from summary service")
                    break
                except websockets.ConnectionClosedOK:
                    logger.debug("WebSocket closed by server")
                    break
                except Exception as e:
                    logger.debug("WebSocket recv error: %s", e)
                    return None

                if isinstance(chunk, (bytes, bytearray)):
                    try:
                        chunk = chunk.decode("utf-8", errors="ignore")
                    except Exception:
                        chunk = str(chunk)

                buffer += chunk
                logger.debug("Message Chunk received (len=%d)", len(chunk))

                if "</message>" in buffer:
                    logger.debug("Terminator '</message>' found in buffer")
                    break

                if asyncio.get_event_loop().time() >= end_time:
                    logger.debug("Deadline reached while waiting for terminator")
                    break

            m = re.search(r"<message>(.*?)</message>", buffer, re.DOTALL)
            if m:
                result = m.group(1).strip()
            else:
                # Fallback to full buffer
                result = buffer.strip() if buffer else None

            logger.debug("Final summary-service response=%s", result)

            return result

    except Exception as e:
        logger.debug("WebSocket connect/send failed to %s: %s", AGENT_WS_SUMMARY_URL, e)
        return None

async def get_chat_name(chat_id: str, messages: list) -> str:
    """
    Obtain a chat name by summarizing its messages via the WebSocket summary service.
    """
    name = ""

    if len(messages) > 0:
        name = await get_chat_summary(messages)
        logger.debug("Obtained name from websocket summary service: %s", name)
    else:
        logger.debug("No messages found for chat_id: %s", chat_id)

    return name
