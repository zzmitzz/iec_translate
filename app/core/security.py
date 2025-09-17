from app.core.settings import get_settings
from fastapi import WebSocket
settings = get_settings()

def verify_api_key(api_key: str):
    """
    Verify if the API key is valid
    """
    return api_key == settings.api_key

async def websocket_auth(websocket: WebSocket) -> bool:
    """
    Verify if the websocket connection is valid
    """
    if not verify_api_key(websocket.headers.get("Authorization")):
        await websocket.close(code=1008, reason="Invalid API key")
        return False
    return True