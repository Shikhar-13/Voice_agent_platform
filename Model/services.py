import aiohttp
import asyncio
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class ServiceType(Enum):
    LLM = "llm"
    TRANSCRIBER = "transcriber"
    VOICE = "voice"

@dataclass
class ServiceConfig:
    name: str
    api_endpoint: str
    quota_limit: int

    async def test_connection(self, api_key: str = None) -> bool:
        """Test connection to service endpoint"""
        try:
            logger.info(f"Testing connection to {self.name} at {self.api_endpoint}")
            async with aiohttp.ClientSession() as session:
                headers = {}
                if api_key:
                    if self.name == "Deepgram":
                        headers['Authorization'] = f'Token {api_key}'
                    elif self.name == "OpenAI":
                        headers['Authorization'] = f'Bearer {api_key}'
                    elif self.name == "ElevenLabs":
                        headers['xi-api-key'] = api_key

                async with session.get(self.api_endpoint, headers=headers) as response:
                    logger.info(f"{self.name} connection test status: {response.status}")
                    return response.status < 500  # Return True for any non-server error

        except Exception as e:
            logger.error(f"Connection test failed for {self.name}: {str(e)}")
            return False

SERVICES = {
    "llm": {
        "openai": ServiceConfig(
            "OpenAI",
            "https://api.openai.com/v1/chat/completions",
            1000
        ),
        "deepseek": ServiceConfig(
            "DeepSeek",
            "https://api.deepseek.com/v1/chat/completions",
            1000
        )
    },
    "transcriber": {
        "deepgram": ServiceConfig(
            "Deepgram",
            "https://api.deepgram.com/v1/listen",
            5000
        )
    },
    "voice": {
        "elevenlabs": ServiceConfig(
            "ElevenLabs",
            "https://api.elevenlabs.io/v1/text-to-speech",
            1000
        )
    }
}

async def test_all_connections(api_keys: dict) -> dict:
    """Test connections to all configured services"""
    results = {}
    for service_type, services in SERVICES.items():
        results[service_type] = {}
        for service_name, config in services.items():
            api_key = api_keys.get(service_name)
            is_connected = await config.test_connection(api_key)
            results[service_type][service_name] = is_connected
            logger.info(f"Connection test for {service_name}: {'Success' if is_connected else 'Failed'}")
    return results

def get_service_config(service_type: ServiceType, service_name: str) -> ServiceConfig:
    try:
        return SERVICES[service_type.value][service_name]
    except KeyError:
        logger.error(f"Service lookup failed for {service_type.value}/{service_name}")
        raise ValueError(f"Service not found: {service_type.value}/{service_name}")

def get_service_endpoint(service_type: ServiceType, service_name: str) -> str:
    config = get_service_config(service_type, service_name)
    return config.api_endpoint
