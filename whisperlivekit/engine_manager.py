"""
TranscriptionEngineManager for handling multiple language-specific transcription engines.
"""
import logging
from typing import Dict, Optional
from whisperlivekit.core import TranscriptionEngine
import asyncio

logger = logging.getLogger(__name__)

class TranscriptionEngineManager:
    """
    Manages multiple TranscriptionEngine instances for different languages.
    Provides language switching capabilities and resource management.
    """
    
    def __init__(self, default_config: dict = None):
        """
        Initialize the manager with default configuration.
        
        Args:
            default_config: Default configuration for new engines
        """
        self.default_config = default_config or {}
        self.engines: Dict[str, TranscriptionEngine] = {}
        self.current_language = self.default_config.get('lan', 'auto')
        self._lock = asyncio.Lock()
        
        # Create default engine
        if self.default_config:
            self.engines[self.current_language] = TranscriptionEngine(**self.default_config)

    async def get_engine_for_language(self, language: str, **config_overrides) -> TranscriptionEngine:
        """
        Get or create a TranscriptionEngine for the specified language.
        
        Args:
            language: Language code (e.g., 'en', 'es', 'fr')
            **config_overrides: Additional configuration overrides
            
        Returns:
            TranscriptionEngine instance for the specified language
        """
        async with self._lock:
            if language not in self.engines:
                logger.info(f"Creating new TranscriptionEngine for language: {language}")
                
                # Merge default config with overrides
                config = {**self.default_config, **config_overrides}
                config['lan'] = language
                
                self.engines[language] = TranscriptionEngine(**config)
            
            return self.engines[language]

    async def switch_language(self, new_language: str, **config_overrides) -> TranscriptionEngine:
        """
        Switch to a different language, creating a new engine if necessary.
        
        Args:
            new_language: Target language code
            **config_overrides: Additional configuration overrides
            
        Returns:
            TranscriptionEngine instance for the new language
        """
        logger.info(f"Switching language from {self.current_language} to {new_language}")
        
        engine = await self.get_engine_for_language(new_language, **config_overrides)
        self.current_language = new_language
        
        return engine

    def get_current_engine(self) -> Optional[TranscriptionEngine]:
        """Get the current active engine."""
        return self.engines.get(self.current_language)

    def get_current_language(self) -> str:
        """Get the current active language."""
        return self.current_language

    def get_available_languages(self) -> list:
        """Get list of currently loaded languages."""
        return list(self.engines.keys())

    async def reinitialize_engine(self, language: str, **new_config):
        """
        Reinitialize an existing engine with new configuration.
        
        Args:
            language: Language of the engine to reinitialize
            **new_config: New configuration parameters
        """
        async with self._lock:
            if language in self.engines:
                logger.info(f"Reinitializing engine for language: {language}")
                self.engines[language].reinitialize_for_language(language, **new_config)
            else:
                logger.warning(f"No engine found for language {language}, creating new one")
                await self.get_engine_for_language(language, **new_config)

    async def remove_engine(self, language: str):
        """
        Remove and clean up an engine for a specific language.
        
        Args:
            language: Language of the engine to remove
        """
        async with self._lock:
            if language in self.engines:
                logger.info(f"Removing engine for language: {language}")
                await self.engines[language].close()
                del self.engines[language]
                
                # If we removed the current engine, switch to another one
                if language == self.current_language and self.engines:
                    self.current_language = next(iter(self.engines.keys()))
                    logger.info(f"Switched current language to: {self.current_language}")

    async def close_all(self):
        """Clean up all engines."""
        logger.info("Closing all TranscriptionEngine instances")
        
        async with self._lock:
            for language, engine in self.engines.items():
                try:
                    await engine.close()
                    logger.info(f"Closed engine for language: {language}")
                except Exception as e:
                    logger.error(f"Error closing engine for {language}: {e}")
            
            self.engines.clear()
            logger.info("All engines closed")

    def get_engine_info(self) -> dict:
        """Get information about all loaded engines."""
        return {
            "current_language": self.current_language,
            "available_languages": list(self.engines.keys()),
            "engine_count": len(self.engines)
        } 