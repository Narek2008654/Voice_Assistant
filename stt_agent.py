"""
LiveKit STT Agent using Groq's Whisper API (free, fast).
Joins a LiveKit room, listens to participant audio, and transcribes Armenian speech.
Post-processes transcriptions with Llama to fix typos, inaccuracies, and grammar.
Speaks answers back via OpenAI TTS (Armenian voice).

Usage:
    python stt_agent.py connect --room test-room   # Connect directly to a room
    python stt_agent.py dev                        # Dev mode (waits for dispatch)
    python stt_agent.py start                      # Production mode
"""

import asyncio
import io
import logging
import os
import re
import wave

import numpy as np
from dotenv import load_dotenv
from groq import AsyncGroq
from openai import AsyncOpenAI
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
TTS_CHUNK_MS = 50
OPENAI_TTS_MODEL = "tts-1"
OPENAI_TTS_VOICE = "nova"
OPENAI_TTS_SAMPLE_RATE = 24000

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

# ─── Deterministic Armenian number-to-words converter ─────────────────

_ARM_ONES = {
    0: "\u0566\u0580\u0578", 1: "\u0574\u0565\u056f", 2: "\u0565\u0580\u056f\u0578\u0582", 3: "\u0565\u0580\u0565\u0584",
    4: "\u0579\u0578\u0580\u057d", 5: "\u0570\u056b\u0576\u0563", 6: "\u057e\u0565\u0581", 7: "\u0575\u0578\u0569",
    8: "\u0578\u0582\u0569", 9: "\u056b\u0576\u0576",
}
_ARM_TEENS = {
    10: "\u057f\u0561\u057d", 11: "\u057f\u0561\u057d\u0576\u0574\u0565\u056f", 12: "\u057f\u0561\u057d\u0576\u0565\u0580\u056f\u0578\u0582",
    13: "\u057f\u0561\u057d\u0576\u0565\u0580\u0565\u0584", 14: "\u057f\u0561\u057d\u0576\u0579\u0578\u0580\u057d",
    15: "\u057f\u0561\u057d\u0576\u0570\u056b\u0576\u0563", 16: "\u057f\u0561\u057d\u0576\u057e\u0565\u0581",
    17: "\u057f\u0561\u057d\u0576\u0575\u0578\u0569", 18: "\u057f\u0561\u057d\u0576\u0578\u0582\u0569",
    19: "\u057f\u0561\u057d\u0576\u056b\u0576\u0576",
}
_ARM_TENS = {
    20: "\u0584\u057d\u0561\u0576", 30: "\u0565\u0580\u0565\u057d\u0578\u0582\u0576", 40: "\u0584\u0561\u057c\u0561\u057d\u0578\u0582\u0576",
    50: "\u0570\u056b\u057d\u0578\u0582\u0576", 60: "\u057e\u0561\u0569\u057d\u0578\u0582\u0576", 70: "\u0575\u0578\u0569\u0561\u0576\u0561\u057d\u0578\u0582\u0576",
    80: "\u0578\u0582\u0569\u057d\u0578\u0582\u0576", 90: "\u056b\u0576\u0576\u057d\u0578\u0582\u0576",
}


def _number_to_armenian(n: int) -> str:
    """Convert a non-negative integer to Armenian cardinal words."""
    if n < 0:
        return "\u0574\u056b\u0576\u0578\u0582\u057d " + _number_to_armenian(-n)
    if n <= 9:
        return _ARM_ONES[n]
    if n <= 19:
        return _ARM_TEENS[n]
    if n <= 99:
        tens, ones = divmod(n, 10)
        result = _ARM_TENS[tens * 10]
        if ones:
            result += " " + _ARM_ONES[ones]
        return result
    if n <= 999:
        hundreds, remainder = divmod(n, 100)
        result = ""
        if hundreds == 1:
            result = "\u0570\u0561\u0580\u0575\u0578\u0582\u0580"
        else:
            result = _ARM_ONES[hundreds] + " \u0570\u0561\u0580\u0575\u0578\u0582\u0580"
        if remainder:
            result += " " + _number_to_armenian(remainder)
        return result
    if n <= 999_999:
        thousands, remainder = divmod(n, 1000)
        result = ""
        if thousands == 1:
            result = "\u0570\u0561\u0566\u0561\u0580"
        else:
            result = _number_to_armenian(thousands) + " \u0570\u0561\u0566\u0561\u0580"
        if remainder:
            result += " " + _number_to_armenian(remainder)
        return result
    if n <= 999_999_999:
        millions, remainder = divmod(n, 1_000_000)
        result = ""
        if millions == 1:
            result = "\u0574\u0565\u056f \u0574\u056b\u056c\u056b\u0578\u0576"
        else:
            result = _number_to_armenian(millions) + " \u0574\u056b\u056c\u056b\u0578\u0576"
        if remainder:
            result += " " + _number_to_armenian(remainder)
        return result
    if n <= 999_999_999_999:
        billions, remainder = divmod(n, 1_000_000_000)
        result = ""
        if billions == 1:
            result = "\u0574\u0565\u056f \u0574\u056b\u056c\u056b\u0561\u0580\u0564"
        else:
            result = _number_to_armenian(billions) + " \u0574\u056b\u056c\u056b\u0561\u0580\u0564"
        if remainder:
            result += " " + _number_to_armenian(remainder)
        return result
    return str(n)


def _decimal_to_armenian(integer_part: str, fractional_part: str) -> str:
    """Convert a decimal like '19.98' to Armenian words: 'տասնինն ամբողջ իննdelays ут'."""
    int_val = int(integer_part)
    frac_val = int(fractional_part)
    return _number_to_armenian(int_val) + " \u0561\u0574\u0562\u0578\u0572\u057b " + _number_to_armenian(frac_val)


def _normalize_numbers_armenian(text: str) -> str:
    """Replace all numbers/percentages/currency/ranges in text with Armenian words."""

    def _clean_int(s: str) -> int:
        return int(s.replace(",", "").replace(" ", ""))

    # 1. Decimal ranges with %: "18.42-22.79%"
    def _range_decimal_pct(m):
        a_int, a_frac, b_int, b_frac = m.group(1), m.group(2), m.group(3), m.group(4)
        return (_decimal_to_armenian(a_int, a_frac) + "\u056b\u0581 " +
                _decimal_to_armenian(b_int, b_frac) + " \u057f\u0578\u056f\u0578\u057d")
    text = re.sub(r"(\d+)[.,](\d+)\s*-\s*(\d+)[.,](\d+)\s*%", _range_decimal_pct, text)

    # 2. Integer ranges with %: "5-10%"
    def _range_int_pct(m):
        a, b = _clean_int(m.group(1)), _clean_int(m.group(2))
        return _number_to_armenian(a) + "\u056b\u0581 " + _number_to_armenian(b) + " \u057f\u0578\u056f\u0578\u057d"
    text = re.sub(r"(\d[\d,]*)\s*-\s*(\d[\d,]*)\s*%", _range_int_pct, text)

    # 3. Decimal with %: "10.5%"
    def _decimal_pct(m):
        return _decimal_to_armenian(m.group(1), m.group(2)) + " \u057f\u0578\u056f\u0578\u057d"
    text = re.sub(r"(\d+)[.,](\d+)\s*%", _decimal_pct, text)

    # 4. Integer with %: "15%"
    def _int_pct(m):
        return _number_to_armenian(_clean_int(m.group(1))) + " \u057f\u0578\u056f\u0578\u057d"
    text = re.sub(r"(\d[\d,]*)\s*%", _int_pct, text)

    # 5. Currency AMD
    def _currency_amd(m):
        return _number_to_armenian(_clean_int(m.group(1))) + " \u0564\u0580\u0561\u0574"
    text = re.sub(r"([\d,]+)\s*(?:AMD|\u0564\u0580\u0561\u0574)", _currency_amd, text)

    # 6. Currency $
    def _currency_usd(m):
        return _number_to_armenian(_clean_int(m.group(1))) + " \u0564\u0578\u056c\u0561\u0580"
    text = re.sub(r"\$\s*([\d,]+)", _currency_usd, text)

    # 7. Plain decimals: "19.98"
    def _decimal(m):
        return _decimal_to_armenian(m.group(1), m.group(2))
    text = re.sub(r"(\d+)[.,](\d+)", _decimal, text)

    # 8. Integer ranges: "5-10"
    def _range_int(m):
        a, b = _clean_int(m.group(1)), _clean_int(m.group(2))
        return _number_to_armenian(a) + "\u056b\u0581 " + _number_to_armenian(b)
    text = re.sub(r"(\d[\d,]*)\s*-\s*(\d[\d,]*)", _range_int, text)

    # 9. Plain integers: "500"
    def _plain_int(m):
        return _number_to_armenian(_clean_int(m.group(0)))
    text = re.sub(r"\d[\d,]*", _plain_int, text)

    return text


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


# ─── OpenAI TTS Helper ────────────────────────────────────────────────

class OpenAITTSHelper:
    """Synthesizes Armenian speech using OpenAI TTS API.
    Normalizes numbers/symbols to Armenian words deterministically before synthesis."""

    def __init__(self):
        self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._sample_rate = OPENAI_TTS_SAMPLE_RATE
        logger.info(f"OpenAI TTS initialized (model={OPENAI_TTS_MODEL}, voice={OPENAI_TTS_VOICE})")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _normalize_for_tts(self, text: str) -> str:
        """Convert numbers/symbols to Armenian words for TTS (deterministic, no LLM)."""
        if not re.search(r"\d", text):
            return text
        normalized = _normalize_numbers_armenian(text)
        if normalized != text:
            logger.info(f"TTS normalized: '{text}' -> '{normalized}'")
        return normalized

    async def synthesize(self, text: str) -> rtc.AudioFrame:
        """Normalize text and synthesize to an AudioFrame."""
        text = self._normalize_for_tts(text)
        response = await self._client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_TTS_VOICE,
            input=text,
            response_format="pcm",
        )
        pcm_bytes = response.content
        samples_per_channel = len(pcm_bytes) // 2
        return rtc.AudioFrame(
            data=pcm_bytes,
            sample_rate=self._sample_rate,
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
    tts_helper = OpenAITTSHelper()

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
    audio_source = rtc.AudioSource(tts_helper.sample_rate, NUM_CHANNELS)
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

                    # Step 3: Send answer as chat message in the room
                    await ctx.room.local_participant.publish_data(
                        answer.encode("utf-8"),
                        topic="lk-chat-topic",
                    )

                    # Step 4: Speak the answer via TTS
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
