import asyncio, os, subprocess, tempfile, traceback, base64, uuid
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import edge_tts, imageio_ffmpeg
from groq import Groq
from deep_translator import GoogleTranslator

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_KEY:
    raise RuntimeError("GROQ_API_KEY not set")
groq_client = Groq(api_key=GROQ_KEY)

_audio_store = {}
_audio_order = []
_preview_cache = {}

VOICES = {
    "US Female · Ava (Warm)":             ("en-US-AvaNeural", None),
    "US Female · Emma (Cheerful)":        ("en-US-EmmaNeural", None),
    "US Female · Jenny (Soft)":           ("en-US-JennyNeural", None),
    "US Female · Aria (Professional)":    ("en-US-AriaNeural", None),
    "US Female · Michelle (Energetic)":   ("en-US-MichelleNeural", None),
    "US Female · Ana (Youthful)":         ("en-US-AnaNeural", None),
    "US Male · Andrew (Natural)":         ("en-US-AndrewNeural", None),
    "US Male · Brian (Casual)":           ("en-US-BrianNeural", None),
    "US Male · Guy (Passionate)":         ("en-US-GuyNeural", None),
    "US Male · Christopher (Authoritative)": ("en-US-ChristopherNeural", None),
    "US Male · Eric (Rational)":          ("en-US-EricNeural", None),
    "US Male · Steffan (Lively)":         ("en-US-SteffanNeural", None),
    "UK Female · Sonia (Warm)":           ("en-GB-SoniaNeural", None),
    "UK Female · Libby (Enthusiastic)":   ("en-GB-LibbyNeural", None),
    "UK Female · Maisie (Youthful)":      ("en-GB-MaisieNeural", None),
    "UK Male · Ryan (Natural)":           ("en-GB-RyanNeural", None),
    "UK Male · Thomas (Refined)":         ("en-GB-ThomasNeural", None),
    "AU Female · Natasha (Assertive)":    ("en-AU-NatashaNeural", None),
    "AU Male · William (Laid-back)":      ("en-AU-WilliamNeural", None),
    "IN Female · Neerja (Warm)":          ("en-IN-NeerjaNeural", None),
    "IN Male · Prabhat (Steady)":         ("en-IN-PrabhatNeural", None),
    "NG Female · Ezinne (Warm)":          ("en-NG-EzinneNeural", None),
    "NG Male · Abeo (Steady)":            ("en-NG-AbeoNeural", None),
    "FR Female · Denise":                  ("fr-FR-DeniseNeural", "fr"),
    "FR Male · Henri":                     ("fr-FR-HenriNeural", "fr"),
    "DE Female · Katja":                   ("de-DE-KatjaNeural", "de"),
    "DE Male · Conrad":                    ("de-DE-ConradNeural", "de"),
    "ES Female · Elvira":                  ("es-ES-ElviraNeural", "es"),
    "ES Male · Alvaro":                    ("es-ES-AlvaroNeural", "es"),
    "IT Female · Elsa":                    ("it-IT-ElsaNeural", "it"),
    "IT Male · Diego":                     ("it-IT-DiegoNeural", "it"),
    "PT Female · Francisca":               ("pt-BR-FranciscaNeural", "pt"),
    "PT Male · Antonio":                   ("pt-BR-AntonioNeural", "pt"),
    "AR Female · Salma":                   ("ar-EG-SalmaNeural", "ar"),
    "AR Male · Shakir":                    ("ar-EG-ShakirNeural", "ar"),
    "HI Female · Swara":                   ("hi-IN-SwaraNeural", "hi"),
    "HI Male · Madhur":                    ("hi-IN-MadhurNeural", "hi"),
    "ZH Female · Xiaoxiao":                ("zh-CN-XiaoxiaoNeural", "zh-CN"),
    "ZH Male · Yunxi":                     ("zh-CN-YunxiNeural", "zh-CN"),
    "JA Female · Nanami":                  ("ja-JP-NanamiNeural", "ja"),
    "JA Male · Keita":                     ("ja-JP-KeitaNeural", "ja"),
    "KO Female · SunHi":                   ("ko-KR-SunHiNeural", "ko"),
    "KO Male · InJoon":                    ("ko-KR-InJoonNeural", "ko"),
    "RU Female · Svetlana":                ("ru-RU-SvetlanaNeural", "ru"),
    "RU Male · Dmitry":                    ("ru-RU-DmitryNeural", "ru"),
}

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

def _save_audio(ogg_path):
    aid = str(uuid.uuid4())
    _audio_store[aid] = ogg_path
    _audio_order.append(aid)
    while len(_audio_order) > 50:
        old = _audio_order.pop(0)
        p = _audio_store.pop(old, None)
        if p and os.path.exists(p):
            try: os.unlink(p)
            except: pass
    return aid

async def _tts(text, voice, out):
    await edge_tts.Communicate(text, voice).save(out)

def _degrade(wav_in):
    out = tempfile.mktemp(suffix=".ogg")
    subprocess.run([
        FFMPEG, "-y",
        "-i", wav_in,
        "-f", "lavfi", "-i", "anoisesrc=color=pink:sample_rate=48000:amplitude=0.5",
        "-filter_complex",
        "[0:a]highpass=f=200,lowpass=f=3200,"
        "acompressor=threshold=-22dB:ratio=8:attack=3:release=120:makeup=2,"
        "aecho=0.85:0.9:8:0.15,"
        "alimiter=level_in=1:level_out=0.97:limit=0.97[v];"
        "[1:a]volume=0.012[n];"
        "[v][n]amix=inputs=2:duration=first:weights=1 1[out]",
        "-map", "[out]",
        "-ar", "16000", "-ac", "1",
        "-c:a", "libopus", "-b:a", "16k", "-vbr", "on",
        "-application", "voip", "-compression_level", "10", out,
    ], check=True, capture_output=True)
    return out

def _transcribe(path):
    with open(path, "rb") as f:
        r = groq_client.audio.transcriptions.create(
            file=(os.path.basename(path), f.read()),
            model="whisper-large-v3-turbo", language="en", response_format="text")
    return r.strip() if isinstance(r, str) else r.text.strip()

PREVIEW_TEXT = {
    None:    "Hey! This is how I sound. Natural, right?",
    "fr":    "Salut ! Voilà comment je sonne. Naturel, non ?",
    "de":    "Hallo! So klinge ich. Ziemlich natürlich, oder?",
    "es":    "¡Hola! Así sueno. ¿Natural, verdad?",
    "it":    "Ciao! Ecco come suono. Naturale, vero?",
    "pt":    "Olá! É assim que eu soo. Natural, né?",
    "ar":    "مرحبا! هكذا أبدو. طبيعي، أليس كذلك؟",
    "hi":    "नमस्ते! मैं ऐसी लगती हूँ। सहज, है ना?",
    "zh-CN": "你好！这就是我的声音。很自然，对吧？",
    "ja":    "こんにちは！これが私の声です。自然でしょ？",
    "ko":    "안녕하세요! 이게 제 목소리예요. 자연스럽죠?",
    "ru":    "Привет! Вот как я звучу. Естественно, да?",
}

@app.get("/health")
async def health():
    return {"status": "ok", "voices": len(VOICES)}

@app.get("/audio/{audio_id}")
async def get_audio(audio_id: str):
    path = _audio_store.get(audio_id)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "Audio not found or expired")
    return FileResponse(path, media_type="audio/ogg")

@app.get("/preview")
async def preview(voice: str):
    if voice not in VOICES:
        raise HTTPException(404, f"Unknown voice: {voice}")
    if voice in _preview_cache and _preview_cache[voice] in _audio_store:
        return {"audio_id": _preview_cache[voice], "mime": "audio/ogg"}
    voice_id, target_lang = VOICES[voice]
    sample = PREVIEW_TEXT.get(target_lang, PREVIEW_TEXT[None])
    clean = tempfile.mktemp(suffix=".wav")
    try:
        await _tts(sample, voice_id, clean)
    except Exception as e:
        return {"stage": "preview_tts", "error": str(e), "trace": traceback.format_exc()[-1000:]}, 500
    try:
        ogg = _degrade(clean)
        aid = _save_audio(ogg)
        _preview_cache[voice] = aid
        return {"audio_id": aid, "mime": "audio/ogg"}
    except Exception as e:
        return {"stage": "preview_ffmpeg", "error": str(e), "trace": traceback.format_exc()[-1000:]}, 500

@app.post("/transform")
async def transform(audio: UploadFile = File(...), profile: str = Form(...)):
    if profile not in VOICES:
        raise HTTPException(400, f"Unknown profile: {profile}")
    voice_id, target_lang = VOICES[profile]
    suffix = os.path.splitext(audio.filename or "v.m4a")[1] or ".m4a"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(await audio.read())
        inp = f.name
    try:
        try:
            text = _transcribe(inp)
        except Exception as e:
            return {"stage": "transcribe", "error": str(e), "trace": traceback.format_exc()[-1000:]}, 500
        if not text:
            raise HTTPException(400, "No speech detected.")
        if target_lang:
            try:
                text = GoogleTranslator(source="en", target=target_lang).translate(text) or text
            except Exception as e:
                return {"stage": "translate", "error": str(e), "trace": traceback.format_exc()[-1000:]}, 500
        clean = tempfile.mktemp(suffix=".wav")
        try:
            await _tts(text, voice_id, clean)
        except Exception as e:
            return {"stage": "tts", "error": str(e), "trace": traceback.format_exc()[-1000:]}, 500
        try:
            ogg = _degrade(clean)
            aid = _save_audio(ogg)
            return {"audio_id": aid, "mime": "audio/ogg"}
        except Exception as e:
            return {"stage": "ffmpeg", "error": str(e), "trace": traceback.format_exc()[-1000:]}, 500
    finally:
        try: os.unlink(inp)
        except: pass
