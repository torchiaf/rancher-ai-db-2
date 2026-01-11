import os

DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "postgres")

AGENT_WS_SUMMARY_URL = os.getenv("AGENT_WS_SUMMARY_URL", "ws://localhost:8000/agent/ws/summary")
AGENT_WS_TIMEOUT = float(os.getenv("AGENT_WS_TIMEOUT", "5"))

LOOP_R_CHATS_INTERVAL = float(os.getenv("LOOP_R_CHATS_INTERVAL", "0.5"))
LOOP_R_MESSAGES_INTERVAL = float(os.getenv("LOOP_R_MESSAGES_INTERVAL", "0.5"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "60"))


def get_db_url() -> str:
    """
    Generate database connection URL.
    """
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"