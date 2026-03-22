"""
LiveKit STT Agent using Groq's Whisper API (free, fast).
Joins a LiveKit room, listens to participant audio, and transcribes Armenian speech.
Post-processes transcriptions with Llama to fix typos, inaccuracies, and grammar.
Speaks answers back via Silero TTS (Armenian voice).

Usage:
    python stt_agent.py connect --room test-room   # Connect directly to a room
    python stt_agent.py dev                        # Dev mode (waits for dispatch)
    python stt_agent.py start                      # Production mode
"""

import asyncio
import functools
import io
import logging
import os
import re
import wave

import numpy as np
import torch
from dotenv import load_dotenv
from groq import AsyncGroq
from livekit import agents, rtc
from livekit.agents import stt
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.plugins.silero import VAD
from silero import silero_tts

from pathlib import Path
load_dotenv(Path(__file__).parent / ".env.local")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stt-agent")

# ─── Configuration ────────────────────────────────────────────────────

WHISPER_LANGUAGE = "hy"  # Armenian
SAMPLE_RATE = 16000
NUM_CHANNELS = 1
TTS_SAMPLE_RATE = 48000
TTS_SPEAKER = "hye_zara"
TTS_CHUNK_MS = 50

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

Banking domain context \u2014 users commonly say words like:
\u057e\u0561\u0580\u056f, \u057e\u0561\u0580\u056f\u0565\u0580, \u0561\u057e\u0561\u0576\u0564, \u0561\u057e\u0561\u0576\u0564\u0576\u0565\u0580, \u057f\u0578\u056f\u0578\u057d\u0561\u0564\u0580\u0578\u0582\u0575\u0584, \u0574\u0561\u057d\u0576\u0561\u0573\u0575\u0578\u0582\u0572, \u0574\u0561\u057d\u0576\u0561\u0573\u0575\u0578\u0582\u0572\u0576\u0565\u0580, \u0562\u0561\u0576\u056f, \u0570\u0561\u0577\u056b\u057e, \u0583\u0578\u056d\u0561\u0576\u0581\u0578\u0582\u0574, \u057e\u0573\u0561\u0580\u0578\u0582\u0574, \u0584\u0561\u0580\u057f, \u057e\u0561\u0580\u056f\u0561\u0575\u056b\u0576 \u057f\u0578\u056f\u0578\u057d\u0561\u0564\u0580\u0578\u0582\u0575\u0584, \u0561\u0574\u057d\u0561\u056f\u0561\u0576 \u057e\u0573\u0561\u0580, \u0570\u056b\u057a\u0578\u0569\u0565\u056f\u0561\u0575\u056b\u0576 \u057e\u0561\u0580\u056f, \u057d\u057a\u0561\u057c\u0578\u0572\u0561\u056f\u0561\u0576 \u057e\u0561\u0580\u056f, \u0531\u0574\u0565\u0580\u056b\u0561\u0562\u0561\u0576\u056f, \u0535\u057e\u0578\u056f\u0561\u0562\u0561\u0576\u056f, \u0531\u0580\u0564\u0577\u056b\u0576\u0562\u0561\u0576\u056f, \u053b\u0576\u0565\u056f\u0578\u0562\u0561\u0576\u056f
Use this domain knowledge to correct ambiguous or misheard words.

Rules:
- Return ONLY the corrected Armenian text, nothing else
- Do NOT translate \u2014 keep everything in Armenian
- You MAY replace wrong words with the correct ones if confident about speaker intent
- Do NOT add extra words or sentences the speaker did not say
- Do NOT explain your changes
- If the input is already clean, return it unchanged
- If the input is just noise or silence artifacts (e.g. 'Subtitle', 'Thank you', repeated single characters), return exactly: NOISE"""

TTS_NORMALIZER_PROMPT = """\
You are a text normalizer preparing Armenian text for a TTS (text-to-speech) engine.
The TTS cannot read digits, symbols, or abbreviations \u2014 you must convert them ALL to Armenian words.

Convert:
- Numbers: "15" \u2192 "\u057f\u0561\u057d\u0576\u0570\u056b\u0576\u0563"
- Percentages: "15%" \u2192 "\u057f\u0561\u057d\u0576\u0570\u056b\u0576\u0563 \u057f\u0578\u056f\u0578\u057d"
- Currency: "3,000,000 AMD" \u2192 "\u0565\u0580\u0565\u0584 \u0574\u056b\u056c\u056b\u0578\u0576 \u0564\u0580\u0561\u0574", "$500" \u2192 "\u0570\u056b\u0576\u0563 \u0570\u0561\u0580\u0575\u0578\u0582\u0580 \u0564\u0578\u056c\u0561\u0580"
- Decimals: "10.5%" \u2192 "\u057f\u0561\u057d \u0561\u0574\u0562\u0578\u0572\u057b \u056f\u0565\u057d \u057f\u0578\u056f\u0578\u057d", "18.42%" \u2192 "\u057f\u0561\u057d\u0576\u0578\u0582\u0569 \u0561\u0574\u0562\u0578\u0572\u057b \u0584\u0561\u057c\u0561\u057d\u0578\u0582\u0576\u0565\u0580\u056f\u0578\u0582 \u057f\u0578\u056f\u0578\u057d"
- Ranges: "5-10" \u2192 "\u0570\u056b\u0576\u0563\u056b\u0581 \u057f\u0561\u057d", "18.42-22.79%" \u2192 "\u057f\u0561\u057d\u0576\u0578\u0582\u0569 \u0561\u0574\u0562\u0578\u0572\u057b \u0584\u0561\u057c\u0561\u057d\u0578\u0582\u0576\u0565\u0580\u056f\u0578\u0582\u056b\u0581 \u0584\u057d\u0561\u0576\u0565\u0580\u056f\u0578\u0582 \u0561\u0574\u0562\u0578\u0572\u057b \u0575\u0578\u0569\u0561\u0576\u0561\u057d\u0578\u0582\u0576\u056b\u0576\u0576 \u057f\u0578\u056f\u0578\u057d"
- Abbreviations: common Armenian banking abbreviations to full words
- IMPORTANT: For complex decimal ranges like "18.42-22.79%", break into simple parts: spell out each number fully in Armenian words, use "\u056b\u0581" for the dash/range, and "\u057f\u0578\u056f\u0578\u057d" for %

Rules:
- Return ONLY the normalized Armenian text
- Keep all non-numeric Armenian text EXACTLY unchanged
- Do NOT add explanations
- Do NOT remove or rephrase any Armenian words"""


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


# ─── Silero TTS Helper ─────────────────────────────────────────────────

class SileroTTSHelper:
    """Synthesizes Armenian speech using Silero V5 CIS Base model.
    Normalizes numbers/symbols to Armenian words via LLM before synthesis."""

    def __init__(self):
        self._model, _ = silero_tts(language="ru", speaker="v5_cis_base")
        self._groq = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        logger.info(f"Silero TTS initialized (speaker={TTS_SPEAKER}, sr={TTS_SAMPLE_RATE})")

    async def _normalize_for_tts(self, text: str) -> str:
        """Use Llama to convert numbers/symbols to Armenian words for TTS."""
        if not re.search(r"\d", text):
            return text
        try:
            response = await self._groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": TTS_NORMALIZER_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=len(text) * 3,
            )
            normalized = response.choices[0].message.content.strip()
            if normalized:
                logger.info(f"TTS normalized: numbers converted to words")
                return normalized
        except Exception as e:
            logger.warning(f"TTS normalization failed, using raw text: {e}")
        return text

    @staticmethod
    def _prepare_ssml(text: str) -> str:
        """Convert plain text to SSML with natural pauses."""
        # Split on sentence boundaries: Armenian ։, period, !, ?
        sentences = re.split(r'(?<=[։\.!\?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return f"<speak>{text}</speak>"

        # Add short pauses after commas within each sentence
        parts = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.replace(",", ',<break time="150ms"/>')
            sentence = sentence.replace("՝", '՝<break time="150ms"/>')  # Armenian comma
            parts.append(f"<s>{sentence}</s>")
            if i < len(sentences) - 1:
                parts.append('<break time="350ms"/>')

        return "<speak>" + "".join(parts) + "</speak>"

    def _synthesize_sync(self, text: str) -> bytes:
        """Run TTS inference (CPU-bound, call from executor)."""
        try:
            ssml = self._prepare_ssml(text)
            audio_tensor = self._model.apply_tts(
                ssml_text=ssml, speaker=TTS_SPEAKER, sample_rate=TTS_SAMPLE_RATE,
            )
        except Exception as e:
            logger.warning(f"SSML synthesis failed, falling back to plain text: {e}")
            audio_tensor = self._model.apply_tts(
                text=text, speaker=TTS_SPEAKER, sample_rate=TTS_SAMPLE_RATE,
            )
        pcm_int16 = (audio_tensor.numpy() * 32767).astype(np.int16)
        return pcm_int16.tobytes()

    async def synthesize(self, text: str) -> rtc.AudioFrame:
        """Normalize text and synthesize to an AudioFrame."""
        text = await self._normalize_for_tts(text)
        loop = asyncio.get_event_loop()
        pcm_bytes = await loop.run_in_executor(
            None, functools.partial(self._synthesize_sync, text),
        )
        samples_per_channel = len(pcm_bytes) // 2  # int16 = 2 bytes per sample
        return rtc.AudioFrame(
            data=pcm_bytes,
            sample_rate=TTS_SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            samples_per_channel=samples_per_channel,
        )


def _chunk_audio_frame(frame: rtc.AudioFrame, chunk_ms: int = TTS_CHUNK_MS) -> list[rtc.AudioFrame]:
    """Split a large AudioFrame into smaller chunks for smooth streaming."""
    samples_per_chunk = frame.sample_rate * chunk_ms // 1000
    data = np.frombuffer(frame.data, dtype=np.int16)
    chunks = []
    for i in range(0, len(data), samples_per_chunk):
        chunk_data = data[i:i + samples_per_chunk]
        chunks.append(rtc.AudioFrame(
            data=chunk_data.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=len(chunk_data),
        ))
    return chunks


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

    # Initialize Groq Whisper STT, post-processor, RAG, and TTS
    whisper_stt = GroqWhisperSTT()
    post_processor = STTPostProcessor()
    tts_helper = SileroTTSHelper()

    from rag import BankRAG
    rag = BankRAG()

    # Load Silero VAD for speech boundary detection
    vad = VAD.load(
        min_speech_duration=0.1,
        min_silence_duration=0.5,
    )

    # Wrap non-streaming Whisper with VAD-based StreamAdapter
    streaming_stt = stt.StreamAdapter(stt=whisper_stt, vad=vad)

    # Publish an audio track so the agent can speak back
    audio_source = rtc.AudioSource(TTS_SAMPLE_RATE, NUM_CHANNELS)
    audio_track = rtc.LocalAudioTrack.create_audio_track("agent-voice", audio_source)
    publish_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await ctx.room.local_participant.publish_track(audio_track, publish_options)
    logger.info("Published agent audio track for TTS output")

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

                    # Step 3: Speak the answer via TTS
                    try:
                        audio_frame = await tts_helper.synthesize(answer)
                        for chunk in _chunk_audio_frame(audio_frame):
                            await audio_source.capture_frame(chunk)
                    except Exception as e:
                        logger.error(f"TTS failed: {e}")

    # Run audio feeding and transcription reading concurrently
    await asyncio.gather(_feed_audio(), _read_transcriptions())


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="whisper-stt-agent",
        ),
    )
