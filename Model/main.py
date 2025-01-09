from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import base64
import json
import os
from pathlib import Path
import logging
from typing import Dict, Any, AsyncGenerator, Optional, Union
import aiohttp
import asyncio
from dotenv import load_dotenv
from enum import Enum
import uvicorn
import numpy as np
from Model.services import ServiceType, get_service_endpoint
import functools
import time
import wave
import io
from datetime import datetime
import asyncio.exceptions

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Voice Agent API",
    description="API for managing voice agent calls and quotas",
    version="1.0.0"
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

class ServiceType(Enum):
    LLM = "llm"
    TRANSCRIBER = "transcriber"
    VOICE = "voice"

# Performance logging decorator
def log_performance(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.debug(f"Starting {func.__name__} with args: {args}, kwargs: {kwargs}")
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"Completed {func.__name__} in {execution_time:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper

class StreamingVoiceAgent:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.StreamingVoiceAgent")
        self.logger.info("Initializing StreamingVoiceAgent")
        self.active_sessions = {}
        self.api_keys = {
            'openai': os.getenv('OPENAI_API_KEY'),
            'deepgram': os.getenv('DEEPGRAM_API_KEY'),
            'elevenlabs': os.getenv('ELEVENLABS_API_KEY'),
            'deepseek': os.getenv('DEEPSEEK_API_KEY'),
            'retell': os.getenv('RETELL_API_KEY')
        }
        logger.debug(f"API keys configured: {', '.join(k for k, v in self.api_keys.items() if v)}")
        self.transcription_buffer = {}
        self.response_buffer = {}
        self.quotas = {
            ServiceType.LLM: {"openai": {"used": 0, "limit": 1000}},
            ServiceType.TRANSCRIBER: {"deepgram": {"used": 0, "limit": 1000}},
            ServiceType.VOICE: {"elevenlabs": {"used": 0, "limit": 1000}}
        }
        logger.info("StreamingVoiceAgent initialized successfully")

    @log_performance
    async def transcribe_audio(self, audio_data: str, service: str) -> Optional[str]:
        """Transcribe audio with improved error handling"""
        if not audio_data or not service:
            raise ValueError("Missing required parameters")

        logger.info(f"Transcribing audio using {service}")
        try:
            if service == 'deepgram':
                endpoint = get_service_endpoint(ServiceType.TRANSCRIBER, service)
                
                # Simplified timeout config
                timeout = aiohttp.ClientTimeout(total=15)
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    headers = {
                        'Authorization': f'Token {self.api_keys["deepgram"]}',
                        'Content-Type': 'audio/wav',
                    }
                    
                    # Simplified audio preparation
                    audio_bytes = (self._prepare_wav_audio(audio_data) 
                                 if isinstance(audio_data, (list, np.ndarray))
                                 else base64.b64decode(audio_data))
                    
                    async with session.post(
                        endpoint,
                        headers=headers,
                        data=audio_bytes,
                        params={'encoding': 'linear16', 'sample_rate': 44100, 'channels': 1}
                    ) as response:
                        if response.status != 200:
                            raise Exception(f"Transcription failed: {await response.text()}")
                            
                        result = await response.json()
                        
                        # Safer response parsing with explicit checks
                        results = result.get('results', {})
                        channels = results.get('channels', [])
                        
                        if not channels:
                            logger.warning("No channels found in transcription response")
                            return ""
                            
                        alternatives = channels[0].get('alternatives', [])
                        if not alternatives:
                            logger.warning("No alternatives found in transcription response")
                            return ""
                            
                        transcript = alternatives[0].get('transcript', '')
                        logger.info(f"Transcription successful: {transcript[:50]}...")
                        return transcript
            else:
                raise ValueError(f"Unsupported transcription service: {service}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"Connection error: {str(e)}")
            raise Exception(f"Failed to connect: {str(e)}")
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}", exc_info=True)
            raise

    def _prepare_wav_audio(self, audio_data: Union[list, np.ndarray]) -> bytes:
        """Helper method to prepare WAV audio data"""
        if not isinstance(audio_data, (list, np.ndarray)):
            raise ValueError("Audio data must be a list or numpy array")
        
        logger.debug("Converting audio array to WAV format")
        try:
            audio_array = np.array(audio_data, dtype=np.int16)
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(44100)
                wav_file.writeframes(audio_array.tobytes())
            audio_bytes = wav_buffer.getvalue()
            logger.debug(f"WAV conversion complete. Size: {len(audio_bytes)} bytes")
            return audio_bytes
        except Exception as e:
            logger.error(f"Error converting audio to WAV: {str(e)}")
            raise

    @log_performance
    async def get_llm_response(self, text: str, service: str) -> str:
        """Get response from LLM service"""
        self.logger.info(f"Getting LLM response using {service}")
        try:
            if service == 'openai':
                endpoint = get_service_endpoint(ServiceType.LLM, 'openai')
                self.logger.debug(f"Using endpoint: {endpoint}")
                
                async with aiohttp.ClientSession() as session:
                    headers = {
                        'Authorization': f'Bearer {self.api_keys["openai"]}',
                        'Content-Type': 'application/json'
                    }
                    data = {
                        'model': 'gpt-3.5-turbo',
                        'messages': [
                            {'role': 'system', 'content': 'You are a helpful voice assistant.'},
                            {'role': 'user', 'content': text}
                        ]
                    }
                    self.logger.debug(f"Sending request: {text[:100]}...")
                    
                    async with session.post(endpoint, headers=headers, json=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            response_text = result['choices'][0]['message']['content']
                            self.logger.info(f"Received response: {response_text[:100]}...")
                            return response_text
                        else:
                            error_text = await response.text()
                            self.logger.error(f"Request failed: {error_text}")
                            raise Exception(f"LLM request failed: {error_text}")
            else:
                raise ValueError(f"Unsupported LLM service: {service}")
        except Exception as e:
            self.logger.error(f"LLM error: {str(e)}", exc_info=True)
            raise

    @log_performance
    async def generate_voice(self, text: str, service: str) -> str:
        """Generate voice using the specified service"""
        logger.info(f"Generating voice using {service}")
        try:
            if service == 'elevenlabs':
                endpoint = get_service_endpoint(ServiceType.VOICE, 'elevenlabs')
                logger.debug(f"Using endpoint: {endpoint}")
                
                async with aiohttp.ClientSession() as session:
                    headers = {
                        'xi-api-key': self.api_keys['elevenlabs'],
                        'Content-Type': 'application/json'
                    }
                    data = {
                        'text': text,
                        'model_id': 'eleven_monolingual_v1',
                        'voice_settings': {
                            'stability': 0.75,
                            'similarity_boost': 0.75
                        }
                    }
                    logger.debug(f"Sending request with text: {text[:50]}...")
                    
                    voice_id = "21m00Tcm4TlvDq8ikWAM"
                    url = f'{endpoint}/{voice_id}/stream'
                    
                    async with session.post(
                        url,
                        headers=headers,
                        json=data
                    ) as response:
                        if response.status == 200:
                            audio_data = await response.read()
                            encoded_audio = base64.b64encode(audio_data).decode('utf-8')
                            logger.info(f"Voice generation successful. Audio size: {len(encoded_audio)} bytes")
                            return encoded_audio
                        else:
                            error_text = await response.text()
                            logger.error(f"Voice generation failed: {error_text}")
                            raise Exception(f"Voice generation failed: {error_text}")
            else:
                raise ValueError(f"Unsupported voice service: {service}")
        except Exception as e:
            logger.error(f"Voice generation error: {str(e)}", exc_info=True)
            raise

    async def process_audio_chunk(self, client_id: str, audio_chunk: bytes) -> Optional[dict]:
        """Process incoming audio chunks with detailed logging and error handling"""
        if not client_id:
            self.logger.error("Invalid client_id provided")
            return None

        if not audio_chunk:
            self.logger.error("Empty audio chunk received")
            return None

        try:
            session = self.active_sessions.get(client_id)
            if not session:
                self.logger.warning(f"No active session for client: {client_id}")
                return None

            # 1. Input Validation
            if len(audio_chunk) > 10 * 1024 * 1024:  # 10MB limit
                self.logger.error("Audio chunk too large")
                return None

            # 2. Transcription with retry
            self.logger.info("=== Starting Transcription ===")
            retry_count = 0
            max_retries = 3
            while retry_count < max_retries:
                try:
                    transcript = await self.transcribe_audio(audio_chunk, 'deepgram')
                    if transcript:
                        self.logger.info(f"Transcription Success: '{transcript}'")
                        break
                    retry_count += 1
                    await asyncio.sleep(1)
                except asyncio.TimeoutError:
                    self.logger.warning(f"Transcription timeout, attempt {retry_count + 1}/{max_retries}")
                    retry_count += 1
                    if retry_count == max_retries:
                        raise

            # 3. LLM Response with timeout
            self.logger.info("=== Getting LLM Response ===")
            input_text = transcript or "Process this input"
            self.logger.info(f"LLM Input: '{input_text}'")
            try:
                async with asyncio.timeout(10):  # 10 second timeout
                    response = await self.get_llm_response(input_text, 'deepseek')
                    if response:
                        self.logger.info(f"LLM Response Success: '{response}'")
                    else:
                        self.logger.warning("LLM returned empty response")
                        return None
            except asyncio.TimeoutError:
                self.logger.error("LLM response timeout")
                return None

            # 4. Voice Generation with size check
            if response:
                self.logger.info("=== Generating Voice ===")
                if len(response) > 1000:  # Limit response length
                    response = response[:1000] + "..."
                self.logger.info(f"Voice Input Text: '{response}'")
                
                try:
                    audio = await self.generate_voice(response, 'elevenlabs')
                    if audio:
                        audio_size = len(audio)
                        if audio_size > 10 * 1024 * 1024:  # 10MB limit
                            self.logger.error("Generated audio too large")
                            return None
                        self.logger.info(f"Voice Generation Success - Size: {audio_size} bytes")
                    else:
                        self.logger.warning("Voice generation returned empty result")
                        return None
                except Exception as e:
                    self.logger.error(f"Voice generation error: {str(e)}")
                    return None

                # 5. Final Response Package with validation
                result = {
                    "transcription": transcript or "",
                    "response": response,
                    "audio": audio
                }

                # Validate result
                if not all(isinstance(v, (str, bytes)) for v in result.values()):
                    self.logger.error("Invalid data types in result")
                    return None

                self.logger.info("=== Processing Complete ===")
                return result

            self.logger.warning("Processing stopped - no LLM response to generate voice from")
            return None

        except Exception as e:
            self.logger.error(f"Processing Error: {str(e)}", exc_info=True)
            # Cleanup on error
            if client_id in self.active_sessions:
                self.active_sessions[client_id]['streaming'] = False
            return None

    async def get_llm_response_stream(self, text: str) -> AsyncGenerator[str, None]:
        """Stream LLM responses"""
        async with aiohttp.ClientSession() as session:
            headers = {
                'Authorization': f'Bearer {self.api_keys["openai"]}',
                'Content-Type': 'application/json'
            }
            data = {
                'model': 'gpt-3.5-turbo',
                'messages': [
                    {'role': 'system', 'content': 'You are a helpful voice assistant. Keep responses concise.'},
                    {'role': 'user', 'content': text}
                ],
                'stream': True
            }
            async with session.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data
            ) as response:
                async for line in response.content:
                    if line:
                        try:
                            json_response = json.loads(line.decode('utf-8').strip('data: '))
                            if 'choices' in json_response:
                                content = json_response['choices'][0].get('delta', {}).get('content')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue

    async def stream_voice(self, text: str) -> AsyncGenerator[bytes, None]:
        """Stream voice synthesis"""
        async with aiohttp.ClientSession() as session:
            headers = {
                'xi-api-key': self.api_keys['elevenlabs'],
                'Content-Type': 'application/json'
            }
            data = {
                'text': text,
                'model_id': 'eleven_monolingual_v1',
                'voice_settings': {
                    'stability': 0.5,
                    'similarity_boost': 0.5
                },
                'optimize_streaming_latency': 4
            }
            async with session.post(
                'https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM/stream',
                headers=headers,
                json=data
            ) as response:
                async for chunk in response.content.iter_chunked(1024):
                    if chunk:
                        yield chunk

voice_agent = StreamingVoiceAgent()

@app.get("/")
async def root():
    # Redirect to the interface page
    return RedirectResponse(url="/interface")

@app.get("/interface", response_class=HTMLResponse)
async def get_interface(request: Request) -> HTMLResponse:
    try:
        return templates.TemplateResponse("interface.html", {"request": request})
    except Exception as e:
        logger.error(f"Template error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    logger.info(f"New WebSocket connection from client: {client_id}")
    await websocket.accept()
    
    # Initialize session with default configuration
    session_config = {
        'websocket': websocket,
        'streaming': False,
        'active': True,
        'config': {
            'llm_service': 'openai',
            'transcriber_service': 'deepgram',
            'voice_service': 'elevenlabs'
        }
    }
    voice_agent.active_sessions[client_id] = session_config
    
    try:
        while voice_agent.active_sessions.get(client_id, {}).get('active', False):
            try:
                message = await websocket.receive_json()
                message_type = message.get('type', '')

                # Get current session
                session = voice_agent.active_sessions.get(client_id)
                if not session:
                    break

                if message_type == 'initialize':
                    # Update session configuration
                    config = message.get('config', {})
                    session['config'].update(config)
                    await websocket.send_json({
                        "type": "initialization",
                        "status": "success",
                        "config": session['config']
                    })

                elif message_type == 'stop':
                    session['streaming'] = False
                    await websocket.send_json({"type": "stop_acknowledged"})

                elif message_type == 'disconnect':
                    session['active'] = False
                    session['streaming'] = False
                    await websocket.send_json({"type": "disconnect_acknowledged"})
                    break

                elif message_type == 'audio_data':
                    audio_data = message.get('data')
                    if audio_data:
                        try:
                            audio_array = np.array(audio_data, dtype=np.int16)
                            result = await voice_agent.process_audio_chunk(client_id, audio_array.tobytes())
                            
                            if result:
                                if result.get('transcription'):
                                    await websocket.send_json({
                                        "type": "transcript",
                                        "text": result['transcription']
                                    })
                                if result.get('response'):
                                    await websocket.send_json({
                                        "type": "response",
                                        "text": result['response']
                                    })
                                if result.get('audio'):
                                    # Send audio as binary data
                                    audio_bytes = base64.b64decode(result['audio'])
                                    await websocket.send_bytes(audio_bytes)
                        except Exception as e:
                            logger.error(f"Error processing audio: {str(e)}")
                            await websocket.send_json({"error": f"Audio processing error: {str(e)}"})

                elif message_type == 'switch_model':
                    model_type = message.get('modelType')
                    model_name = message.get('modelName')
                    if model_type and model_name:
                        session['config'][f'{model_type}_service'] = model_name
                        await websocket.send_json({
                            "type": "model_update",
                            "modelType": model_type,
                            "modelName": model_name,
                            "status": "success"
                        })

            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for client: {client_id}")
                break
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
                await websocket.send_json({"error": "Invalid JSON format"})
            except Exception as e:
                logger.error(f"WebSocket error for client {client_id}: {str(e)}", exc_info=True)
                await websocket.send_json({"error": f"Server error: {str(e)}"})

    except Exception as e:
        logger.error(f"Critical websocket error: {str(e)}")
    finally:
        # Cleanup
        if client_id in voice_agent.active_sessions:
            logger.info(f"Cleaning up session for client: {client_id}")
            voice_agent.active_sessions[client_id]['active'] = False
            voice_agent.active_sessions[client_id]['streaming'] = False
            del voice_agent.active_sessions[client_id]

@app.get("/quotas")
async def get_quotas() -> Dict[ServiceType, Dict[str, Dict[str, int]]]:
    return voice_agent.quotas

@app.get("/test-audio")
async def test_audio():
    """Test endpoint for audio functionality"""
    try:
        # Test voice synthesis
        test_audio = await voice_agent.generate_voice(
            "This is a test of the voice synthesis system.",
            "elevenlabs"
        )
        return JSONResponse({
            "status": "success",
            "message": "Audio test successful",
            "audio": test_audio
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio test failed: {str(e)}")

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
    except Exception as e:
        logger.error(f"Server failed to start: {e}")