#!/usr/bin/env python3
"""
Startup script for the Realtime Translate API server
"""

import uvicorn
import logging
from app.core.settings import get_settings

if __name__ == "__main__":
    # Get settings
    settings = get_settings()
    
    # Configure logging
    logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting {settings.app_name} server...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Host: {settings.host}:{settings.port}")
    
    # Run the server
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        access_log=True
    ) 