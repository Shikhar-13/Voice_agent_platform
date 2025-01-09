from dataclasses import dataclass
import aiohttp
import asyncio
import time
from .services import ServiceType, SERVICES

@dataclass
class RetellConfig:
    api_key: str
    llm_service: str
    transcriber_service: str
    voice_service: str

class RetellClient:
    def __init__(self, config: RetellConfig):
        self.config = config
        self.session = None
        self.is_streaming = False
        self._connection = None
        self._active = False
        self._start_time = None
    
    async def initialize(self):
        self.session = aiohttp.ClientSession()
        await self.setup()
    
    async def setup(self):
        if self.config.llm_service not in SERVICES[ServiceType.LLM]:
            raise ValueError(f"Invalid LLM service: {self.config.llm_service}")
        if self.config.transcriber_service not in SERVICES[ServiceType.TRANSCRIBER]:
            raise ValueError(f"Invalid transcriber service: {self.config.transcriber_service}")
        if self.config.voice_service not in SERVICES[ServiceType.VOICE]:
            raise ValueError(f"Invalid voice service: {self.config.voice_service}")
        
        self._connection = {
            'llm': SERVICES[ServiceType.LLM][self.config.llm_service],
            'transcriber': SERVICES[ServiceType.TRANSCRIBER][self.config.transcriber_service],
            'voice': SERVICES[ServiceType.VOICE][self.config.voice_service]
        }
    
    async def start_stream(self):
        if self.is_streaming:
            return
        self._active = True
        self._start_time = time.time()
        self.is_streaming = True
    
    async def stop_stream(self):
        if not self.is_streaming:
            return
        await self.end_interaction()
    
    async def end_interaction(self):
        if not self._active:
            return
        self._active = False
        self.is_streaming = False
    
    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
