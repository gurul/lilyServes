import asyncio
import time
from typing import AsyncIterator, Awaitable, Callable, Optional

from google.cloud.speech_v2 import SpeechAsyncClient
from google.cloud.speech_v2.types import cloud_speech

# Reconnect before Google's ~5 minute streaming limit
STREAM_TIME_LIMIT_SECONDS = 290

# Google limits each request's audio field to 25600 bytes
MAX_AUDIO_CHUNK_BYTES = 25600


class GoogleStreamingSession:
    """
    Manages a Google Speech-to-Text V2 streaming session for one phone call.

    Feeds raw mulaw audio into a gRPC bidirectional stream and fires callbacks
    for interim and final transcription results.  Handles automatic reconnection
    before the ~5-minute streaming limit.
    """

    def __init__(
        self,
        project_id: str,
        location: str = "global",
        model: str = "telephony",
        language_code: str = "en-US",
        on_interim_result: Optional[Callable[[str], Awaitable[None]]] = None,
        on_final_result: Optional[Callable[[str, str], Awaitable[None]]] = None,
    ):
        self._project_id = project_id
        self._location = location
        self._model = model
        self._language_code = language_code
        self._on_interim_result = on_interim_result  # callback(interim_text)
        self._on_final_result = on_final_result      # callback(final_text, full_transcript)

        self._client: Optional[SpeechAsyncClient] = None
        self._audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._running_transcript: list[str] = []
        self._is_running = False
        self._stream_task: Optional[asyncio.Task] = None

    @property
    def running_transcript(self) -> list[str]:
        return self._running_transcript

    @property
    def full_transcript_text(self) -> str:
        return " ".join(self._running_transcript)

    def _get_client(self) -> SpeechAsyncClient:
        if self._client is None:
            self._client = SpeechAsyncClient()
        return self._client

    def _get_recognizer_path(self) -> str:
        return f"projects/{self._project_id}/locations/{self._location}/recognizers/_"

    def _build_streaming_config(self) -> cloud_speech.StreamingRecognitionConfig:
        return cloud_speech.StreamingRecognitionConfig(
            config=cloud_speech.RecognitionConfig(
                explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.MULAW,
                    sample_rate_hertz=8000,
                    audio_channel_count=1,
                ),
                model=self._model,
                language_codes=[self._language_code],
            ),
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=True,
            ),
        )

    async def _request_generator(self) -> AsyncIterator[cloud_speech.StreamingRecognizeRequest]:
        """Yield config first, then audio chunks from the queue."""
        # First message must carry recognizer + config
        yield cloud_speech.StreamingRecognizeRequest(
            recognizer=self._get_recognizer_path(),
            streaming_config=self._build_streaming_config(),
        )

        stream_start = time.monotonic()
        while self._is_running:
            # Proactively reconnect before the time limit
            if time.monotonic() - stream_start >= STREAM_TIME_LIMIT_SECONDS:
                return

            try:
                chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if chunk is None:  # Sentinel: session is closing
                return

            # Split into sub-chunks if needed
            for i in range(0, len(chunk), MAX_AUDIO_CHUNK_BYTES):
                yield cloud_speech.StreamingRecognizeRequest(audio=chunk[i:i + MAX_AUDIO_CHUNK_BYTES])

    async def _consume_responses(self, response_stream) -> None:
        """Read interim and final results from the gRPC response stream."""
        async for response in response_stream:
            for result in response.results:
                if not result.alternatives:
                    continue
                text = result.alternatives[0].transcript.strip()
                if not text:
                    continue

                if result.is_final:
                    self._running_transcript.append(text)
                    if self._on_final_result:
                        await self._on_final_result(text, self.full_transcript_text)
                else:
                    if self._on_interim_result:
                        await self._on_interim_result(text)

    async def _run_stream_loop(self) -> None:
        """Run streaming sessions in a loop, reconnecting as needed."""
        client = self._get_client()
        while self._is_running:
            try:
                response_stream = await client.streaming_recognize(
                    requests=self._request_generator(),
                )
                await self._consume_responses(response_stream)
            except Exception as e:
                if not self._is_running:
                    break
                print(f"[google_transcriber] Stream error: {e}, reconnecting...")
                await asyncio.sleep(0.5)
                continue

            if not self._is_running:
                break
            print("[google_transcriber] Reconnecting stream (time limit)...")

    async def start(self) -> None:
        """Start the streaming session."""
        self._is_running = True
        self._stream_task = asyncio.create_task(self._run_stream_loop())

    def feed_audio(self, mulaw_bytes: bytes) -> None:
        """Feed raw mulaw audio bytes into the session. Non-blocking."""
        if self._is_running:
            self._audio_queue.put_nowait(mulaw_bytes)

    async def stop(self) -> list[str]:
        """Stop the session and return the full transcript."""
        self._is_running = False
        self._audio_queue.put_nowait(None)  # Unblock the generator
        if self._stream_task:
            try:
                await asyncio.wait_for(self._stream_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._stream_task.cancel()
        return self._running_transcript
