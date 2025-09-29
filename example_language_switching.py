#!/usr/bin/env python3
"""
Example client demonstrating runtime language switching for the IEC Backend.

This example shows how to:
1. Connect to different language-specific WebSocket streams
2. Switch languages via the REST API
3. Get information about available engines
"""

import asyncio
import websockets
import requests
import json
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageSwitchingClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "your-api-key"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws")
        self.api_key = api_key
        self.headers = {"X-API-Key": api_key}

    def get_engine_info(self) -> dict:
        """Get information about current engine state."""
        try:
            response = requests.get(
                f"{self.base_url}/reconfig_model/info",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get engine info: {e}")
            return {}

    def switch_language(self, language: str, config: Optional[dict] = None) -> dict:
        """Switch the default engine to a new language."""
        try:
            payload = config or {}
            response = requests.post(
                f"{self.base_url}/reconfig_model/switch/{language}",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to switch language to {language}: {e}")
            return {}

    def reinitialize_engine(self, language: str, config: dict) -> dict:
        """Reinitialize an existing engine with new configuration."""
        try:
            response = requests.post(
                f"{self.base_url}/reconfig_model/reinitialize/{language}",
                headers=self.headers,
                json=config
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to reinitialize engine for {language}: {e}")
            return {}

    def remove_engine(self, language: str) -> dict:
        """Remove an engine for a specific language."""
        try:
            response = requests.delete(
                f"{self.base_url}/reconfig_model/remove/{language}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to remove engine for {language}: {e}")
            return {}

    async def connect_websocket(self, room_id: int, language: Optional[str] = None):
        """Connect to WebSocket stream with optional language specification."""
        ws_url = f"{self.ws_url}/stream/{room_id}"
        if language:
            ws_url += f"?language={language}"
        
        logger.info(f"Connecting to WebSocket: {ws_url}")
        
        try:
            async with websockets.connect(ws_url) as websocket:
                logger.info(f"Connected to room {room_id}" + (f" with language {language}" if language else ""))
                
                # Listen for messages
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        logger.info(f"Received: {data}")
                        
                        if data.get("type") == "ready_to_stop":
                            logger.info("Server indicated ready to stop")
                            break
                    except json.JSONDecodeError:
                        logger.warning(f"Received non-JSON message: {message}")
                        
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")

async def main():
    """Example usage of the language switching functionality."""
    client = LanguageSwitchingClient(
        base_url="http://localhost:8000",
        api_key="test-api-key"  # Replace with your actual API key
    )
    
    print("=== Language Switching Example ===\n")
    
    # 1. Get initial engine info
    print("1. Getting initial engine information...")
    info = client.get_engine_info()
    print(f"Current state: {json.dumps(info, indent=2)}\n")
    
    # 2. Switch to Spanish
    print("2. Switching to Spanish...")
    result = client.switch_language("es", {
        "model": "base",
        "target_language": "en"  # Translate Spanish to English
    })
    print(f"Switch result: {json.dumps(result, indent=2)}\n")
    
    # 3. Connect to WebSocket with Spanish
    print("3. Connecting to WebSocket with Spanish language...")
    # This would typically be done with audio data
    # await client.connect_websocket(room_id=1, language="es")
    
    # 4. Switch to French
    print("4. Switching to French...")
    result = client.switch_language("fr", {
        "model": "small",
        "backend": "faster-whisper"
    })
    print(f"Switch result: {json.dumps(result, indent=2)}\n")
    
    # 5. Get updated engine info
    print("5. Getting updated engine information...")
    info = client.get_engine_info()
    print(f"Updated state: {json.dumps(info, indent=2)}\n")
    
    # 6. Reinitialize Spanish engine with different model
    print("6. Reinitializing Spanish engine with different model...")
    result = client.reinitialize_engine("es", {
        "model": "large",
        "backend": "faster-whisper"
    })
    print(f"Reinitialize result: {json.dumps(result, indent=2)}\n")
    
    # 7. Connect to WebSocket with specific language
    print("7. Example: Connect to WebSocket with French language...")
    print("   ws://localhost:8000/stream/1?language=fr")
    print("   This would allow the client to specify language per connection\n")
    
    # 8. Remove Spanish engine
    print("8. Removing Spanish engine...")
    result = client.remove_engine("es")
    print(f"Remove result: {json.dumps(result, indent=2)}\n")
    
    # 9. Final engine info
    print("9. Final engine information...")
    info = client.get_engine_info()
    print(f"Final state: {json.dumps(info, indent=2)}\n")
    
    print("=== Example Complete ===")

if __name__ == "__main__":
    asyncio.run(main()) 