import asyncio, os, subprocess, tempfile, traceback
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import edge_tts, imageio_ffmpeg
from groq import Groq

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_KEY:
    raise RuntimeError("GROQ_API_KEY not set")
groq_client = Groq(api_key=GROQ_KEY)

VOICES = {
    "US Female · Ava (Warm)":             "en-US-AvaNeural",
    "US Female · Emma (Cheerful)":        "en-US-EmmaNeural",
    "US Female · Jenny (Soft)":           "en-US-JennyNeural",
    "US Female · Aria (Professional)":    "en-US-AriaNeural",
    "US Female · Michelle (Energetic)":   "en-US-MichelleNeural",
    "US Female · Ana (Youthful)":         "en-US-AnaNeural",
    "US Male · Andrew (Natural)":         "en-US-AndrewNeural",
    "US Male · Brian (Casual)":           "en-US-BrianNeural",
    "US Male · Guy (Passionate)":         "en-US-GuyNeural",
    "US Male · Christopher (Authoritative)": "en-US-ChristopherNeural",
    "US Male · Eric (Rational)":          "en-US-EricNeural",
    "US Male · Steffan (Lively)":         "en-US-SteffanNeural",
    "UK Female · Sonia (Warm)":           "en-GB-SoniaNeural",
    "UK Female · Libby (Enthusiastic)":   "en-GB-LibbyNeural",
    "UK Female · Maisie (Youthful)":      "en-GB-MaisieNeural",
    "UK Male · Ryan (Natural)":           "en-GB-RyanNeural",
    "UK Male · Thomas (Refined)":         "en-GB-ThomasNeural",
    "AU Female · Natasha (Assertive)":    "en-AU-NatashaNeural",
    "AU Male · William (Laid-back)":      "en-AU-WilliamNeural",
    "IN Female · Neerja (Warm)":          "en-IN-NeerjaNeural",
    "IN Male · Prabhat (Steady)":         "en-IN-PrabhatNeural",
    "NG Female · Ezinne (Warm)":          "en-NG-EzinneNeural",
    "NG Male · Abeo (Steady)":            "en-NG-AbeoNeural",
}

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

async def _tts(text, voice, out):
    await edge_tts.Communicate(text, voice).save(out)

def _degrade(wav_in):
    """Degrade to real phone-mic voice note: band-limit, compress, add room tone, 16kbps Opus."""
    out = tempfile.mktemp(suffix=".ogg")
    filter_graph = (
        # Band-limit to phone range (real mics cut more aggressively than PSTN)
        "[0:a]highpass=f=200,lowpass=f=3200,"
        # Hard compression simulating phone AGC pumping
        "acompressor=threshold=-22dB:ratio=8:attack=3:release=120:makeup=2,"
        # Small-room reverb (early reflections of a real space)
        "aecho=0.85:0.9:8:0.15,"
        # Soft clip a touch (phone mic preamp overload)
        "alimiter=level_in=1:level_out=0.97:limit=0.97[v];"
        # Pink noise bed = mic self-noise + room tone
        "[1:a]volume=0.012[n];"
        # Mix voice (loud) + noise (quiet)
        "[v][n]amix=inputs=2:duration=first:weights=1 1[out]"
    )
    subprocess.run([
        FFMPEG, "-y",
        "-i", wav_in,
        "-f", "lavfi", "-i", "anoisesrc=color=pink:sample_rate=48000:amplitude=0.5",
        "-filter_complex", filter_graph,
        "-map", "[out]",
        "-ar", "16000",              # WhatsApp wideband rate
        "-ac", "1",
        "-c:a", "libopus",
        "-b:a", "16k",               # WhatsApp low-end bitrate
        "-vbr", "on",
        "-application", "voip",
        "-compression_level", "10",
        out,
    ], check=True, capture_output=True)
    return out

def _transcribe(path):
    with open(path, "rb") as f:
        r = groq_client.audio.transcriptions.create(
            file=(os.path.basename(path), f.read()),
            model="whisper-large-v3-turbo", language="en", response_format="text")
    return r.strip() if isinstance(r, str) else r.text.strip()

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Voice preview cache + samples ─────────────────────────────
_preview_cache = {}

PREVIEW_TEXT = {
    None:   "Hey! This is how I sound. Pretty natural, right?",
    "fr":   "Salut ! Voilà comment je sonne. Naturel, non ?",
    "de":   "Hallo! So klinge ich. Ziemlich natürlich, oder?",
    "es":   "¡Hola! Así sueno. ¿Natural, verdad?",
    "it":   "Ciao! Ecco come suono. Naturale, vero?",
    "pt":   "Olá! É assim que eu soo. Natural, né?",
    "ar":   "مرحبا! هكذا أبدو. طبيعي، أليس كذلك؟",
    "hi":   "नमस्ते! मैं ऐसी लगती हूँ। सहज, है ना?",
    "zh-CN":"你好！这就是我的声音。很自然，对吧？",
    "ja":   "こんにちは！これが私の声です。自然でしょ？",
    "ko":   "안녕하세요! 이게 제 목소리예요. 자연스럽죠?",
    "ru":   "Привет! Вот как я звучу. Естественно, да?",
}

@app.get("/preview")
async def preview(voice: str):
    if voice not in VOICES:
        raise HTTPException(404, f"Unknown voice: {voice}")

    if voice in _preview_cache:
        return {"audio_b64": _preview_cache[voice], "mime": "audio/ogg"}

    voice_id, target_lang = VOICES[voice]
    sample = PREVIEW_TEXT.get(target_lang, PREVIEW_TEXT[None])

    clean = tempfile.mktemp(suffix=".wav")
    try:
        await _tts(sample, voice_id, clean)
    except Exception as e:
        return {"stage": "preview_tts", "error": str(e),
                "trace": traceback.format_exc()[-1000:]}, 500

    try:
        with open(_degrade(clean), "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        _preview_cache[voice] = b64
        return {"audio_b64": b64, "mime": "audio/ogg"}
    except Exception as e:
        return {"stage": "preview_ffmpeg", "error": str(e),
                "trace": traceback.format_exc()[-1000:]}, 500


@app.post("/transform")
async def transform(audio: UploadFile = File(...), profile: str = Form(...)):
    if profile not in VOICES:
        raise HTTPException(400, f"Unknown profile: {profile}")
    suffix = os.path.splitext(audio.filename or "v.m4a")[1] or ".m4a"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(await audio.read())
        inp = f.name
    try:
        try:
            text = _transcribe(inp)
        except Exception as e:
            return {"stage": "transcribe", "error": str(e), "type": type(e).__name__, "trace": traceback.format_exc()[-2000:]}, 500

        if not text:
            raise HTTPException(400, "No speech detected.")

        clean = tempfile.mktemp(suffix=".wav")
        try:
            await _tts(text, VOICES[profile], clean)
        except Exception as e:
            return {"stage": "tts", "error": str(e), "type": type(e).__name__, "trace": traceback.format_exc()[-2000:]}, 500

        try:
            return FileResponse(_degrade(clean), media_type="audio/ogg", filename="accent.ogg")
        except Exception as e:
            return {"stage": "ffmpeg", "error": str(e), "type": type(e).__name__, "trace": traceback.format_exc()[-2000:]}, 500
    finally:
        try: os.unlink(inp)
        except: pass
