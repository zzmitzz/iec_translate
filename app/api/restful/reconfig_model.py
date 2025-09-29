from fastapi import APIRouter, Header, HTTPException, Depends, Request
from app.core.security import verify_api_key
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/reconfig_model",
    tags=["reconfig_model"],
)

class LanguageConfig(BaseModel):
    """Configuration for language switching."""
    target_language: Optional[str] = None
    model: Optional[str] = None
    backend: Optional[str] = None
    task: Optional[str] = None
    additional_config: Optional[Dict[str, Any]] = {}

class EngineInfo(BaseModel):
    """Information about the current engine state."""
    current_language: str
    available_languages: list
    engine_count: int

@router.post("/switch/{language}")
async def switch_language(
    language: str,
    config: Optional[LanguageConfig] = None,
    request: Request = None,
    x_api_key: str = Header(None)
):
    """
    Switch the transcription engine to a different language.
    
    Args:
        language: Target language code (e.g., 'en', 'es', 'fr', 'auto')
        config: Optional additional configuration for the new engine
        x_api_key: API key for authentication
    """
    if not verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        engine_manager = request.app.state.engine_manager
        
        # Prepare configuration overrides
        config_overrides = {}
        if config:
            if config.target_language:
                config_overrides['target_language'] = config.target_language
            if config.model:
                config_overrides['model'] = config.model
            if config.backend:
                config_overrides['backend'] = config.backend
            if config.task:
                config_overrides['task'] = config.task
            if config.additional_config:
                config_overrides.update(config.additional_config)
        
        # Switch to the new language
        engine = await engine_manager.switch_language(language, **config_overrides)
        
        logger.info(f"Successfully switched to language: {language}")
        
        return {
            "message": f"Successfully switched to language: {language}",
            "language": language,
            "config": config_overrides,
            "engine_info": engine_manager.get_engine_info()
        }
        
    except Exception as e:
        logger.error(f"Failed to switch language to {language}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to switch language: {str(e)}")

@router.post("/reinitialize/{language}")
async def reinitialize_engine(
    language: str,
    config: LanguageConfig,
    request: Request = None,
    x_api_key: str = Header(None)
):
    """
    Reinitialize an existing engine with new configuration.
    
    Args:
        language: Language of the engine to reinitialize
        config: New configuration parameters
        x_api_key: API key for authentication
    """
    if not verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        engine_manager = request.app.state.engine_manager
        
        # Prepare configuration
        new_config = {}
        if config.target_language:
            new_config['target_language'] = config.target_language
        if config.model:
            new_config['model'] = config.model
        if config.backend:
            new_config['backend'] = config.backend
        if config.task:
            new_config['task'] = config.task
        if config.additional_config:
            new_config.update(config.additional_config)
        
        # Reinitialize the engine
        await engine_manager.reinitialize_engine(language, **new_config)
        
        logger.info(f"Successfully reinitialized engine for language: {language}")
        
        return {
            "message": f"Successfully reinitialized engine for language: {language}",
            "language": language,
            "config": new_config,
            "engine_info": engine_manager.get_engine_info()
        }
        
    except Exception as e:
        logger.error(f"Failed to reinitialize engine for {language}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reinitialize engine: {str(e)}")

@router.get("/info", response_model=EngineInfo)
async def get_engine_info(
    request: Request = None,
    x_api_key: str = Header(None)
):
    """
    Get information about the current engine state.
    
    Args:
        x_api_key: API key for authentication
    """
    if not verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        engine_manager = request.app.state.engine_manager
        info = engine_manager.get_engine_info()
        
        return EngineInfo(**info)
        
    except Exception as e:
        logger.error(f"Failed to get engine info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get engine info: {str(e)}")

@router.delete("/remove/{language}")
async def remove_engine(
    language: str,
    request: Request = None,
    x_api_key: str = Header(None)
):
    """
    Remove and clean up an engine for a specific language.
    
    Args:
        language: Language of the engine to remove
        x_api_key: API key for authentication
    """
    if not verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        engine_manager = request.app.state.engine_manager
        
        if language not in engine_manager.get_available_languages():
            raise HTTPException(status_code=404, detail=f"No engine found for language: {language}")
        
        await engine_manager.remove_engine(language)
        
        logger.info(f"Successfully removed engine for language: {language}")
        
        return {
            "message": f"Successfully removed engine for language: {language}",
            "engine_info": engine_manager.get_engine_info()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove engine for {language}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove engine: {str(e)}")

# Legacy endpoint for backward compatibility
@router.post("/{language}")
async def reconfig_model(
    language: str,
    request: Request = None,
    x_api_key: str = Header(None)
):
    """
    Legacy endpoint for model reconfiguration. 
    Redirects to switch_language for backward compatibility.
    """
    return await switch_language(language, None, request, x_api_key)