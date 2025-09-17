from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from whisperlivekit import TranscriptionEngine, AudioProcessor
import asyncio
import logging
from starlette.staticfiles import StaticFiles
import pathlib
import whisperlivekit.web as webpkg
import sys
import os
from app.core.settings import get_settings
from app.core.security import websocket_auth
from app.api.ws.connection.connection_manager import connection_manager
from datetime import datetime

settings = get_settings()
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/stream",
    tags=["stream"],
)

async def handle_websocket_results(websocket, results_generator, room: int):
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            # await websocket.send_json(response)
            await connection_manager.broadcast_to_room(room, response)
        # when the results_generator finishes it means all audio has been processed
        logger.info("Results generator finished. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected while handling results (client likely closed connection).")
    except Exception as e:
        logger.exception(f"Error in WebSocket results handler: {e}")

@router.websocket("/{room_id}")
async def stream(websocket: WebSocket, room_id: int):
    # if not await websocket_auth(websocket):
    #     return
    print("--------------------------------")
    # Use the shared TranscriptionEngine from app state instead of creating a new one
    engine = websocket.app.state.transcription_engine
    audio_processor = AudioProcessor(
        transcription_engine=engine,
    )
    await connection_manager.connect(websocket, room_id)
    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator, room_id))
    logger.info("New websocket connection opened. Waiting for audio data...")
    try:
        while True:
            event = await websocket.receive()
            if event.get("type") == "websocket.disconnect":
                break
            data_bytes = event.get("bytes")
            data_text = event.get("text")

            if data_bytes is not None:
                await audio_processor.process_audio(data_bytes)
            elif data_text is not None:
                logger.warning("Received text frame on /asr; expected binary audio bytes. Ignoring.")
            else:
                logger.debug(f"Received event: {event}")
    except KeyError as e:
        if 'bytes' in str(e):
            logger.warning(f"Client has closed the connection.")
        else:
            logger.error(f"Unexpected KeyError in websocket_endpoint: {e}", exc_info=True)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client during message receiving loop.")
    except Exception as e:
        logger.error(f"Unexpected error in websocket_endpoint main loop: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up WebSocket endpoint...")
        if not websocket_task.done():
            websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            logger.info("WebSocket results handler task was cancelled.")
        except Exception as e:
            logger.warning(f"Exception while awaiting websocket_task completion: {e}")
            
        await audio_processor.cleanup()
        logger.info("WebSocket endpoint cleaned up successfully.")
