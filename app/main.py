from ast import arg
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.settings import get_settings
from contextlib import asynccontextmanager
from whisperlivekit import TranscriptionEngine, parse_args
from app.api.restful.reconfig_model import router as reconfig_model_router
from app.api.ws.stream import router as stream_router
from app.api.ws.connection.connection_manager import connection_manager
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
# Get settings instance
settings = get_settings()

from app.core.security import websocket_auth

args = parse_args()
# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format=settings.log_format,
    handlers=[
        logging.FileHandler(settings.log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.transcription_engine = TranscriptionEngine(
        **vars(args),
    )
    yield
    await app.state.transcription_engine.close()


app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan
)

# Add CORS middleware with configurable settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_credentials,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)

app.include_router(reconfig_model_router)
app.include_router(stream_router)

@app.websocket("/audience/{room_id}")
async def audience_websocket_endpoint(websocket: WebSocket, room_id: str):
    # Convert room_id to int for consistency with connection manager
    try:
        room_id_int = int(room_id)
    except ValueError:
        logger.error(f"Invalid room_id format: {room_id}")
        await websocket.close(code=1003, reason="Invalid room ID")
        return
    
    connection_id = await connection_manager.connect(websocket, room_id_int)
    logger.info(f"Audience connection {connection_id} joined room {room_id_int}")
    
    try:
        # Create and broadcast join notification to other connections in the room
        join_message = {
            "type": "user_joined",
            "content": f"Audience member joined the room",
            "connection_id": connection_id,
            "room_id": room_id_int, 
            "timestamp": datetime.now().isoformat(),
            "message_id": f"join_{connection_id}_{datetime.now().timestamp()}"
        }
        
        # Broadcast to other connections in the room (exclude the new connection)
        await connection_manager.broadcast_to_room(room_id_int, join_message, exclude_connection=connection_id)
        
        # Keep the connection alive and handle disconnection
        while True:
            try:
                event = await websocket.receive()
                if event.get("type") == "websocket.disconnect":
                    break
                # Audience connections typically only listen, but we can handle text messages if needed
                if event.get("text"):
                    logger.debug(f"Received text from audience {connection_id}: {event.get('text')}")
            except WebSocketDisconnect:
                logger.info(f"Audience WebSocket {connection_id} disconnected by client.")
                break
            except Exception as e:
                logger.error(f"Error handling audience WebSocket {connection_id}: {e}")
                break
                
    except Exception as e:
        logger.error(f"Error in audience_websocket_endpoint for {connection_id}: {e}")
    finally:
        # Clean up connection and notify other users
        logger.info(f"Cleaning up audience connection {connection_id}")
        
        # Create and broadcast leave notification
        leave_message = {
            "type": "user_left", 
            "content": f"Audience member left the room",
            "connection_id": connection_id,
            "room_id": room_id_int,
            "timestamp": datetime.now().isoformat(),
            "message_id": f"leave_{connection_id}_{datetime.now().timestamp()}"
        }
        
        # Broadcast leave message before disconnecting
        await connection_manager.broadcast_to_room(room_id_int, leave_message, exclude_connection=connection_id)
        
        # Disconnect from connection manager
        connection_manager.disconnect(connection_id)
        logger.info(f"Audience connection {connection_id} cleaned up successfully.")

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI server is starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI server is shutting down...")

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to Realtime Translate API", "status": "running"}

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed")
    return {"status": "healthy", "service": settings.app_name}

@app.get("/settings")
async def get_app_settings():
    """Get current application settings (excluding sensitive information)"""
    logger.info("Settings endpoint accessed")
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "environment": settings.environment,
        "debug": settings.debug,
        "host": settings.host,
        "port": settings.port,
        "log_level": settings.log_level,
        "whisper_model": settings.whisper_model,
        "whisper_device": settings.whisper_device,
        "default_source_language": settings.default_source_language,
        "default_target_language": settings.default_target_language,
        "max_connections": settings.max_connections,
        "websocket_timeout": settings.websocket_timeout,
        "audio_sample_rate": settings.audio_sample_rate,
        "audio_chunk_duration": settings.audio_chunk_duration
    }

if __name__ == "__main__":
    logger.info("Starting server with Uvicorn...")
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )
