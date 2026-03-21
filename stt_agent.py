"""
LiveKit STT Agent using Groq's Whisper API (free, fast).
Joins a LiveKit room, listens to participant audio, and transcribes Armenian speech.
Post-processes transcriptions with Llama to fix typos, inaccuracies, and grammar.

Usage:
    python stt_agent.py connect --room test-room   # Connect directly to a room
    python stt_agent.py dev                        # Dev mode (waits for dispatch)
    python stt_agent.py start                      # Production mode
"""

import asyncio
import io
import logging
import os
import wave

import numpy as np
from dotenv import load_dotenv
from groq import AsyncGroq
from livekit import agents, rtc
from livekit.agents import stt
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.plugins.silero import VAD

from pathlib import Path
load_dotenv(Path(__file__).parent / ".env.local")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stt-agent")

# ─── Configuration ────────────────────────────────────────────────────

WHISPER_LANGUAGE = "hy"  # Armenian
SAMPLE_RATE = 16000
NUM_CHANNELS = 1

POST_PROCESSOR_SYSTEM_PROMPT = """\
You are an Armenian text post-processor for a banking voice assistant STT system.
Your job is to take raw speech-to-text output and return corrected Armenian text.

Fix these issues:
- Spelling mistakes and typos (wrong, missing, or extra Armenian letters)
- Grammatical errors (wrong case endings, verb conjugations, agreements)
- Misheard words: replace with the most likely intended Armenian word based on phonetic similarity and context
- Repeated words or phrases (Whisper hallucinations like the same word 3+ times)
- Wrong or missing punctuation
- Non-Armenian gibberish, artifacts, or random Latin characters mixed in
- Word boundary errors (incorrectly split or merged words)
- Filler sounds transcribed as words

Banking domain context — users commonly say words like:
\u057e\u0561\u0580\u056f, \u057e\u0561\u0580\u056f\u0565\u0580, \u0561\u057e\u0561\u0576\u0564, \u0561\u057e\u0561\u0576\u0564\u0576\u0565\u0580, \u057f\u0578\u056f\u0578\u057d\u0561\u0564\u0580\u0578\u0582\u0575\u0584, \u0574\u0561\u057d\u0576\u0561\u0573\u0575\u0578\u0582\u0572, \u0574\u0561\u057d\u0576\u0561\u0573\u0575\u0578\u0582\u0572\u0576\u0565\u0580, \u0562\u0561\u0576\u056f, \u0570\u0561\u0577\u056b\u057e, \u0583\u0578\u056d\u0561\u0576\u0581\u0578\u0582\u0574, \u057e\u0573\u0561\u0580\u0578\u0582\u0574, \u0584\u0561\u0580\u057f, \u057e\u0561\u0580\u056f\u0561\u0575\u056b\u0576 \u057f\u0578\u056f\u0578\u057d\u0561\u0564\u0580\u0578\u0582\u0575\u0584, \u0561\u0574\u057d\u0561\u056f\u0561\u0576 \u057e\u0573\u0561\u0580, \u0570\u056b\u057a\u0578\u0569\u0565\u056f\u0561\u0575\u056b\u0576 \u057e\u0561\u0580\u056f, \u057d\u057a\u0561\u057c\u0578\u0572\u0561\u056f\u0561\u0576 \u057e\u0561\u0580\u056f, \u0531\u0574\u0565\u0580\u056b\u0561\u0562\u0561\u0576\u056f, \u0535\u057e\u0578\u056f\u0561\u0562\u0561\u0576\u056f, \u0531\u0580\u0564\u0577\u056b\u0576\u0562\u0561\u0576\u056f, \u053b\u0576\u0565\u056f\u0578\u0562\u0561\u0576\u056f
Use this domain knowledge to correct ambiguous or misheard words.

Rules:
- Return ONLY the corrected Armenian text, nothing else
- Do NOT translate — keep everything in Armenian
- You MAY replace wrong words with the correct ones if confident about speaker intent
- Do NOT add extra words or sentences the speaker did not say
- Do NOT explain your changes
- If the input is already clean, return it unchanged
- If the input is just noise or silence artifacts (e.g. 'Subtitle', 'Thank you', repeated single characters), return exactly: NOISE"""


# ─── STT Post-Processor (cleans up Whisper output via LLM) ────────────

class STTPostProcessor:
    """Uses Groq Llama to clean up raw Whisper transcriptions."""

    def __init__(self):
        self._client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        logger.info("STT Post-Processor initialized (Groq Llama).")

    async def process(self, raw_text: str) -> str:
        """Clean up raw STT text using Llama."""
        if not raw_text.strip():
            return ""

        try:
            response = await self._client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": POST_PROCESSOR_SYSTEM_PROMPT},
                    {"role": "user", "content": raw_text},
                ],
                temperature=0.0,
                max_tokens=len(raw_text) * 2 + 50,
            )
            cleaned = response.choices[0].message.content.strip()
            if cleaned == "NOISE":
                return ""
            if cleaned != raw_text:
                logger.info(f"Post-processed: '{raw_text}' -> '{cleaned}'")
            return cleaned
        except Exception as e:
            logger.warning(f"Post-processing failed, using raw text: {e}")
            return raw_text


# ─── Groq Whisper STT adapter for LiveKit ─────────────────────────────

class GroqWhisperSTT(stt.STT):
    """LiveKit-compatible STT using Groq's hosted Whisper API."""

    def __init__(self, language: str = WHISPER_LANGUAGE):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False),
        )
        self._language = language
        self._client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        logger.info("Groq Whisper STT initialized.")

    async def _recognize_impl(
        self,
        buffer: agents.utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Transcribe an audio buffer using Groq's Whisper API."""
        lang = language if language is not NOT_GIVEN else self._language

        # Merge all audio frames and convert to WAV bytes
        merged = agents.utils.merge_frames(buffer)
        audio_data = np.frombuffer(merged.data, dtype=np.int16)

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(merged.num_channels)
            wf.setsampwidth(2)
            wf.setframerate(merged.sample_rate)
            wf.writeframes(audio_data.tobytes())
        wav_buf.seek(0)

        # Send to Groq API
        transcription = await self._client.audio.transcriptions.create(
            file=("audio.wav", wav_buf.read()),
            model="whisper-large-v3",
            language=lang,
            response_format="verbose_json",
        )

        full_text = transcription.text.strip() if transcription.text else ""

        if full_text:
            logger.info(f"Transcribed ({lang}): {full_text}")

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=full_text,
                    language=lang,
                    confidence=1.0 if full_text else 0.0,
                ),
            ],
        )


# ─── LiveKit Agent entrypoint ─────────────────────────────────────────

async def entrypoint(ctx: agents.JobContext):
    """Connects to a LiveKit room, listens to audio, and prints transcriptions."""
    logger.info(f"Connecting to room: {ctx.room.name}")
    await ctx.connect()

    # Initialize Groq Whisper STT, post-processor, and RAG
    whisper_stt = GroqWhisperSTT()
    post_processor = STTPostProcessor()

    from rag import BankRAG
    rag = BankRAG()

    # Load Silero VAD for speech boundary detection
    vad = VAD.load(
        min_speech_duration=0.1,
        min_silence_duration=0.5,
    )

    # Wrap non-streaming Whisper with VAD-based StreamAdapter
    streaming_stt = stt.StreamAdapter(stt=whisper_stt, vad=vad)

    # Wait for a participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    # Subscribe to participant's microphone audio
    audio_stream = rtc.AudioStream.from_participant(
        participant=participant,
        track_source=rtc.TrackSource.SOURCE_MICROPHONE,
        sample_rate=SAMPLE_RATE,
        num_channels=NUM_CHANNELS,
    )

    stt_stream = streaming_stt.stream()

    async def _feed_audio():
        """Forward audio frames from LiveKit to the STT stream."""
        async for event in audio_stream:
            stt_stream.push_frame(event.frame)
        stt_stream.end_input()

    async def _read_transcriptions():
        """Process transcriptions: clean up, retrieve context, generate answer."""
        async for event in stt_stream:
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                raw_text = event.alternatives[0].text if event.alternatives else ""
                if raw_text.strip():
                    # Step 1: Clean up via LLM post-processor
                    cleaned_text = await post_processor.process(raw_text)
                    if not cleaned_text.strip():
                        continue

                    print(f"\n{'='*60}")
                    print(f"[USER] {cleaned_text}")

                    # Step 2: Get answer from RAG
                    answer = await rag.answer(cleaned_text)
                    print(f"[BOT]  {answer}")
                    print(f"{'='*60}\n")

    # Run audio feeding and transcription reading concurrently
    await asyncio.gather(_feed_audio(), _read_transcriptions())


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="whisper-stt-agent",
        ),
    )
