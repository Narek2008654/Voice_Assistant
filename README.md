# Voice Assistant — Armenian Banking Voice AI Agent

An end-to-end Voice AI customer support agent for Armenian banks, built on the open-source **LiveKit** framework. The agent understands and speaks Armenian, answering questions about **credits**, **deposits**, and **branch locations** using data scraped directly from official bank websites.

## Supported Banks

| Bank | Credits | Deposits | Branches |
|------|---------|----------|----------|
| Ameriabank | ✓ | ✓ | ✓ |
| Evocabank | ✓ | ✓ | ✓ |
| Ardshinbank | ✓ | ✓ | ✓ |
| Inecobank | ✓ | ✓ | ✓ |

## Architecture

```
Microphone → LiveKit Room → Silero VAD → Groq Whisper (STT)
    → GPT-4o-mini Post-Processor (fix Armenian typos)
    → ChromaDB Semantic Retrieval → Llama 3.3 70B (Answer Generation)
    → Armenian Number Normalizer → OpenAI TTS (Speech Output)
```

### Components

| Component | Technology | Why |
|-----------|-----------|-----|
| **Real-time audio** | LiveKit Agents (open-source) | Low-latency WebRTC framework with built-in agent SDK, VAD integration, and room management |
| **Voice Activity Detection** | Silero VAD | Lightweight, accurate speech boundary detection that works well with streaming audio |
| **Speech-to-Text** | Groq Whisper large-v3 | Free hosted API, fast inference, supports Armenian (`hy`) language natively |
| **STT Post-Processing** | OpenAI GPT-4o-mini | Corrects Armenian spelling/grammar errors from Whisper output; strong multilingual understanding with few-shot examples for banking domain |
| **Retrieval** | ChromaDB + `paraphrase-multilingual-MiniLM-L12-v2` | Vector database with multilingual sentence embeddings for semantic search over Armenian bank data; persisted locally for fast startup |
| **Answer Generation** | Groq Llama 3.3 70B Versatile | Strongest open model on Groq for generating accurate Armenian answers from retrieved context |
| **Number Normalization** | Deterministic Python converter | Converts numbers, percentages, decimals, currency, and ranges to Armenian cardinal words before TTS |
| **Text-to-Speech** | OpenAI TTS (`tts-1`, voice: `nova`) | High-quality speech synthesis; raw PCM output streamed in real-time chunks over WebRTC |
| **Web Scraping** | Requests + BeautifulSoup, Selenium | Bank-specific strategies: static HTML for Evocabank, headless Chrome for JS-rendered sites (Ameriabank, Inecobank), REST API extraction for Ardshinbank |

### Why These Models?

#### Speech-to-Text: Groq Whisper large-v3

| | Groq Whisper large-v3 (chosen) | Google Speech-to-Text | Azure Speech | Deepgram | Local Whisper |
|---|---|---|---|---|---|
| **Armenian support** | Native (`hy`), trained on multilingual data | Limited Armenian accuracy | Limited Armenian accuracy | No Armenian support | Same model, same quality |
| **Cost** | Free tier (generous limits) | $0.006/15 sec | $1/audio hour | $0.0043/min | Free but needs GPU |
| **Latency** | ~1-2s (Groq's LPU hardware) | ~2-3s | ~2-3s | ~1s | ~5-10s on CPU, ~2s on GPU |
| **Infrastructure** | API call, no hardware needed | API call | API call | API call | Requires local GPU (4+ GB VRAM) |

**Why Groq Whisper**: Free, fast, good Armenian accuracy, zero infrastructure. Google and Azure have limited Armenian quality and charge per request. Deepgram doesn't support Armenian at all. Running Whisper locally requires a GPU, which adds cost and complexity for deployment.

---

#### STT Post-Processing: OpenAI GPT-4o-mini

| | GPT-4o-mini (chosen) | Llama 3.1 8B (Groq) | Claude Haiku | No post-processing |
|---|---|---|---|---|
| **Armenian understanding** | Strong multilingual, high accuracy | Basic — often misses subtle errors | Good multilingual | N/A |
| **Cost** | $0.15/1M input tokens | Free on Groq | $0.25/1M input tokens | Free |
| **Latency** | ~300-500ms | ~100-200ms (Groq LPU) | ~300-500ms | 0ms |
| **Purpose fit** | Best accuracy for Armenian typo correction | Fast but weak on Armenian edge cases | Similar quality, higher cost | Whisper errors pass through |

**Why GPT-4o-mini**: The post-processor fixes Armenian spelling, removes Whisper hallucinations, and corrects misheard banking terms. Llama 3.1 8B on Groq was the original choice (free, fast) but its Armenian understanding was too weak — it missed subtle errors and sometimes introduced new ones. GPT-4o-mini handles Armenian significantly better, and the cost is negligible ($0.15/1M tokens — a typical correction is ~50 tokens). Claude Haiku is comparable quality but more expensive. Skipping post-processing entirely was tested but Whisper's Armenian errors degraded RAG retrieval quality.

---

#### Retrieval: ChromaDB + paraphrase-multilingual-MiniLM-L12-v2

**Vector Database:**

| | ChromaDB (chosen) | Pinecone | Weaviate | FAISS |
|---|---|---|---|---|
| **Hosting** | Local (embedded, persisted to disk) | Cloud-hosted (managed) | Self-hosted or cloud | Local (in-memory) |
| **Cost** | Free | Free tier limited, then paid | Free (self-hosted) | Free |
| **Setup complexity** | `pip install chromadb`, zero config | Account + API key + index setup | Docker or cloud setup | Manual index management, no metadata |
| **Persistence** | Built-in disk persistence | Managed | Built-in | Manual save/load |
| **Metadata filtering** | Built-in (bank, category, URL) | Built-in | Built-in | No native metadata support |

**Why ChromaDB**: Zero infrastructure — installs as a Python package and persists vectors to a local directory. No external service, no account, no network dependency. Pinecone requires a cloud account and adds network latency. Weaviate requires Docker or cloud hosting. FAISS is fast but has no built-in persistence or metadata filtering, which we need to filter by bank and category.

**Embedding Model:**

| | paraphrase-multilingual-MiniLM-L12-v2 (chosen) | OpenAI text-embedding-3-small | multilingual-e5-large | all-MiniLM-L6-v2 |
|---|---|---|---|---|
| **Armenian support** | 50+ languages including Armenian | Supports Armenian | 100+ languages | English-only |
| **Cost** | Free (runs locally) | $0.02/1M tokens | Free (runs locally) | Free (runs locally) |
| **Model size** | 471 MB | API call | 2.24 GB | 91 MB |
| **Quality on Armenian** | Good semantic matching | Better quality | Better quality | Cannot handle Armenian |
| **Offline** | Yes | No (API required) | Yes | Yes |

**Why multilingual-MiniLM**: Free, runs locally, supports Armenian with good semantic matching quality. OpenAI embeddings would be higher quality but add API cost and network dependency for every query. The e5-large model is better but 5x larger (slower to load, more memory). English-only models like all-MiniLM-L6-v2 cannot handle Armenian text at all.

---

#### Answer Generation: Groq Llama 3.3 70B Versatile

| | Llama 3.3 70B on Groq (chosen) | GPT-4o | Claude Sonnet | Llama 3.1 8B | Gemini 1.5 Flash |
|---|---|---|---|---|---|
| **Armenian quality** | Good — generates coherent, grounded Armenian answers | Excellent Armenian | Excellent Armenian | Poor — short, often inaccurate answers | Good Armenian |
| **Cost** | Free on Groq (100K tokens/day) | $2.50/1M input tokens | $3/1M input tokens | Free on Groq | $0.075/1M input tokens |
| **Latency** | ~1-3s (Groq LPU) | ~2-5s | ~2-5s | ~0.5s | ~1-2s |
| **Context window** | 128K tokens | 128K tokens | 200K tokens | 128K tokens | 1M tokens |
| **Grounding ability** | Strong — follows system prompt to stay within provided context | Excellent | Excellent | Weak — tends to hallucinate or give generic answers | Good |

**Why Llama 70B on Groq**: Free and fast. The 70B model produces significantly better Armenian than the 8B — longer, more detailed, properly grounded answers. GPT-4o and Claude would be higher quality but cost money on every user question, which adds up quickly for a voice agent handling continuous conversations. The Groq free tier (100K tokens/day) is sufficient for development and moderate usage. Llama 8B was tested but its Armenian answers were too short and often inaccurate.

---

#### Text-to-Speech: OpenAI TTS (tts-1, voice: nova)

| | OpenAI TTS (chosen) | Google Cloud TTS | Azure Neural TTS | Facebook MMS-TTS | ElevenLabs |
|---|---|---|---|---|---|
| **Armenian quality** | Good — natural prosody, intelligible | Limited Armenian voices | No Armenian voice | Armenian supported but robotic | No Armenian |
| **Cost** | $15/1M characters | $4/1M characters (Neural) | $16/1M characters | Free (open-source, local) | $0.30/1K characters |
| **Latency** | ~0.5-1s for full response | ~0.5-1s | ~0.5-1s | ~2-3s (CPU inference) | ~1-2s |
| **Voice quality** | Natural, clear | Good (for supported languages) | Good (for supported languages) | Robotic, monotone | Premium (for supported languages) |
| **Streaming** | PCM output, chunked streaming | Streaming supported | Streaming supported | No streaming, full synthesis | Streaming supported |

**Why OpenAI TTS**: Best balance of Armenian speech quality and ease of integration. Facebook MMS-TTS was the first choice (free, open-source, has Armenian) but the audio was too robotic for a customer-facing voice agent — users couldn't understand it well. Google Cloud TTS has limited Armenian voice options. Azure has no Armenian neural voice. ElevenLabs doesn't support Armenian. OpenAI's `nova` voice produces natural-sounding Armenian despite not being specifically trained for it.

---

#### Real-time Audio Framework: LiveKit Agents

| | LiveKit (chosen) | Twilio | Daily.co | Custom WebSocket |
|---|---|---|---|---|
| **Open source** | Yes (server + client + agent SDK) | No | Partially | N/A |
| **Agent SDK** | Built-in Python SDK with VAD, STT, TTS integration | Voice AI is separate product | No agent SDK | Build everything from scratch |
| **Cost** | Free (self-hosted) or cloud free tier | Pay per minute | Pay per minute | Free but high dev cost |
| **WebRTC** | Built-in, handles all complexity | Built-in | Built-in | Manual implementation |
| **VAD integration** | Native `StreamAdapter` wraps any STT with VAD | Separate config | Manual | Manual |

**Why LiveKit**: Open-source with a Python agent SDK designed specifically for voice AI. The `StreamAdapter` lets us wrap non-streaming Whisper with Silero VAD in one line. Twilio and Daily.co charge per minute and don't have the same level of agent SDK integration. Building a custom WebSocket solution would require implementing WebRTC, audio encoding/decoding, VAD integration, and room management from scratch.

---

#### Voice Activity Detection: Silero VAD

| | Silero VAD (chosen) | WebRTC VAD | pyannote VAD | Energy-based VAD |
|---|---|---|---|---|
| **Accuracy** | High — neural network based, handles noise well | Moderate — rule-based, struggles with noise | Very high — state of the art | Low — fails with background noise |
| **Size** | ~1 MB | Built into WebRTC | ~30 MB + dependencies | No model |
| **Latency** | <10ms per frame | <1ms per frame | ~50ms per frame | <1ms per frame |
| **LiveKit integration** | Direct plugin (`livekit-plugins-silero`) | Would need manual integration | Would need manual integration | Would need manual integration |
| **Cost** | Free, runs locally | Free | Free | Free |

**Why Silero VAD**: Direct LiveKit plugin support — `VAD.load()` and it works. High accuracy with a tiny model. WebRTC VAD is simpler but less accurate with real-world noise. pyannote is more powerful but heavier and designed for offline diarization, not real-time streaming. Energy-based approaches fail in noisy environments (which is common for phone/browser microphones).

## Project Structure

```
Voice_Agent/
├── banks_config.py     # Bank URLs and scraping methods (edit this to add banks)
├── scraper.py          # Generic scraping engine (driven by banks_config.py)
├── rag.py              # ChromaDB vector retrieval + Llama answer generation
├── stt_agent.py        # Main LiveKit agent — STT, post-processing, number normalization, TTS
├── generate_token.py   # LiveKit room token generator for testing
├── requirements.txt    # Python dependencies
├── data/
│   ├── bank_data.json  # Scraped bank data (generated by scraper.py)
│   └── chroma_db/      # ChromaDB vector store (auto-generated on first run)
├── .env.example        # Template for required API keys
└── .env.local          # Your API keys (not committed)
```

## Setup

### Prerequisites

- Python 3.11+
- Google Chrome (for Selenium-based scraping)
- A LiveKit server (cloud or self-hosted)

### 1. Clone and install dependencies

```bash
git clone https://github.com/Narek2008654/Voice_Assistant.git
cd Voice_Assistant
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the example and fill in your keys:

```bash
cp .env.example .env.local
```

```env
LIVEKIT_URL=wss://your-livekit-server.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```

| Key | Where to get it |
|-----|----------------|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | [LiveKit Cloud Dashboard](https://cloud.livekit.io) |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) (free tier) |
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) |

### 3. Scrape bank data

```bash
python scraper.py
```

This crawls all 4 bank websites and saves structured data to `data/bank_data.json`. Takes ~10-15 minutes due to Selenium page loads and rate limiting.

### 4. Run the agent

```bash
# Connect directly to a room
python stt_agent.py connect --room test-room

# Or dev mode (waits for dispatch)
python stt_agent.py dev
```

### 5. Generate a test token and join

```bash
python generate_token.py
```

Open the printed Meet URL in your browser to join the room with your microphone.

## How It Works

1. **Audio capture**: LiveKit streams microphone audio from the browser to the agent via WebRTC
2. **VAD**: Silero VAD detects speech start/end boundaries in the audio stream
3. **Transcription**: Complete speech segments are sent to Groq's Whisper API for Armenian transcription
4. **Post-processing**: Raw transcription is cleaned by GPT-4o-mini (fixes Armenian spelling, removes Whisper artifacts, corrects misheard banking terms)
5. **Retrieval**: ChromaDB performs cosine similarity search over multilingual embeddings of chunked bank data
6. **Answer generation**: Top-K relevant chunks are passed as context to Llama 70B, which generates a grounded Armenian answer
7. **Number normalization**: All numbers, percentages, decimals, and currency in the answer are converted to Armenian cardinal words
8. **Speech synthesis**: The normalized text is sent to OpenAI TTS, and the resulting audio is streamed back to the user in real-time chunks

## Scalability

Adding a new bank requires editing **only** `banks_config.py`:

```python
"NewBank": {
    "method": "selenium",  # or "html" or "api"
    "credits": "https://newbank.am/loans",
    "deposits": "https://newbank.am/deposits",
    "branches": "https://newbank.am/branches",
},
```

Just provide the root URL for each category — the scraper automatically discovers and crawls all sub-pages. Then re-run `python scraper.py`. No changes needed in `scraper.py`, `rag.py`, or `stt_agent.py`.
