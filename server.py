import asyncio, os, subprocess, tempfile, traceback, base64, uuid, re
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import edge_tts, imageio_ffmpeg
from groq import Groq
from gradio_client import Client, handle_file
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

ENGLISH_VOICES = {
    "US Female": "en-US-AriaNeural",
    "US Male":   "en-US-GuyNeural",
    "UK Female": "en-GB-SoniaNeural",
    "UK Male":   "en-GB-RyanNeural",
}


REFERENCE_MAP = {
    "US Female": "references/us_female.mp3",
    "US Male":   "references/us_male.wav",
    "UK Female": "references/uk_female.mp3",
    "UK Male":   "references/uk_male.mp3",
}

_omnivoice_client = None

def _omnivoice_client_get():
    global _omnivoice_client
    if _omnivoice_client is None:
        _omnivoice_client = Client("k2-fsa/OmniVoice")
    return _omnivoice_client

def _omnivoice_speak(text: str, profile: str) -> str:
    """Speak text in the reference voice using OmniVoice Space. Returns WAV path."""
    ref_path = REFERENCE_MAP.get(profile)
    if not ref_path or not os.path.exists(ref_path):
        raise RuntimeError(f"Reference missing for {profile}")

    client = _omnivoice_client_get()
    # Estimate duration: ~1 sec per 12 characters, minimum 3s
    duration = max(3.0, min(30.0, len(text) / 12.0))
    result = client.predict(
        text=text,
        lang="Auto",
        ref_aud=handle_file(ref_path),
        ref_text="",
        instruct="female, young adult, high pitch, american accent" if "Female" in profile else "male, young adult, low pitch, american accent",
        ns=32,
        gs=2.0,
        dn=True,
        sp=1.0,
        du=duration,
        pp=True,
        po=True,
        api_name="/_clone_fn",
    )
    if isinstance(result, dict):
        return result.get("path") or result.get("name")
    return result

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

def _save_audio(opus_path):
    aid = str(uuid.uuid4())
    _audio_store[aid] = opus_path
    _audio_order.append(aid)
    while len(_audio_order) > 50:
        old = _audio_order.pop(0)
        p = _audio_store.pop(old, None)
        if p and os.path.exists(p):
            try: os.unlink(p)
            except: pass
    return aid

async def _tts(text, voice, out):
    await edge_tts.Communicate(text, voice, rate="-12%", pitch="-2Hz").save(out)

def _degrade(wav_in):
    """WhatsApp-native voice note: 16kHz mono Opus in OGG, with duration metadata."""
    out = tempfile.mktemp(suffix=".ogg")
    subprocess.run([
        FFMPEG, "-y",
        "-i", wav_in,
        "-f", "lavfi", "-i", "anoisesrc=color=pink:sample_rate=16000:amplitude=0.35",
        "-f", "lavfi", "-i", "anoisesrc=color=brown:sample_rate=16000:amplitude=0.2",
        "-filter_complex",
        "[0:a]highpass=f=250,lowpass=f=3400,"
        "acompressor=threshold=-26dB:ratio=10:attack=5:release=150:makeup=3,"
        "aecho=0.9:0.7:12:0.12,"
        "alimiter=level_in=1:level_out=0.95:limit=0.95[voice];"
        "[1:a]volume=0.028[room];"
        "[2:a]volume=0.012[handling];"
        "[voice][room][handling]amix=inputs=3:duration=first:weights=1 0.7 0.4[out]",
        "-map", "[out]",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "libopus",
        "-b:a", "24k",
        "-vbr", "on",
        "-application", "voip",
        "-compression_level", "10",
        out,
    ], check=True, capture_output=True)
    return out

def _transcribe(path, lang="en"):
    with open(path, "rb") as f:
        if lang:
            r = groq_client.audio.transcriptions.create(
                file=(os.path.basename(path), f.read()),
                model="whisper-large-v3-turbo", language=lang, response_format="text")
            return (r.strip() if isinstance(r, str) else r.text.strip()), lang
        else:
            r = groq_client.audio.transcriptions.create(
                file=(os.path.basename(path), f.read()),
                model="whisper-large-v3-turbo", response_format="verbose_json")
            return (r.text or "").strip(), getattr(r, "language", "unknown") or "unknown"

_LANG_NAMES = {
    "fr": "French", "de": "German", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "ar": "Arabic", "hi": "Hindi", "zh-CN": "Simplified Chinese",
    "ja": "Japanese", "ko": "Korean", "ru": "Russian",
}

def _translate(text, target):
    lang_name = _LANG_NAMES.get(target, target)
    try:
        r = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": f"Translate the user's English text to {lang_name}. Reply with ONLY the translation."},
                {"role": "user", "content": text},
            ],
            temperature=0.2, max_tokens=1000,
        )
        t = r.choices[0].message.content.strip()
        if t: return t
    except Exception:
        pass
    for fn in (
        lambda: GoogleTranslator(source="en", target=target).translate(text),
        lambda: __import__("deep_translator").MyMemoryTranslator(source="en-US", target=target).translate(text),
    ):
        try:
            r = fn()
            if r: return r
        except Exception:
            continue
    return None

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
    return FileResponse(path, media_type="audio/opus")

@app.get("/preview")
async def preview(voice: str):
    if voice not in VOICES:
        raise HTTPException(404, f"Unknown voice: {voice}")
    if voice in _preview_cache and _preview_cache[voice] in _audio_store:
        return {"audio_id": _preview_cache[voice], "mime": "audio/opus"}
    voice_id, target_lang = VOICES[voice]
    sample = PREVIEW_TEXT.get(target_lang, PREVIEW_TEXT[None])
    clean = tempfile.mktemp(suffix=".wav")
    try:
        await _tts(sample, voice_id, clean)
    except Exception as e:
        return {"stage": "preview_tts", "error": str(e)}, 500
    try:
        opus_file = _degrade(clean)
        aid = _save_audio(opus_file)
        _preview_cache[voice] = aid
        return {"audio_id": aid, "mime": "audio/opus"}
    except Exception as e:
        return {"stage": "preview_ffmpeg", "error": str(e)}, 500

@app.post("/translate-incoming")
async def translate_incoming(audio: UploadFile = File(...), voice: str = Form("US Female")):
    if voice not in ENGLISH_VOICES:
        raise HTTPException(400, f"Unknown voice: {voice}")
    suffix = os.path.splitext(audio.filename or "in.m4a")[1] or ".m4a"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(await audio.read())
        inp = f.name
    try:
        try:
            original, detected = _transcribe(inp, lang=None)
        except Exception as e:
            return {"stage": "transcribe", "error": str(e)}, 500
        if not original:
            raise HTTPException(400, "No speech detected.")
        english = original
        if not str(detected).lower().startswith("en"):
            t = _translate(original, "en")
            if t: english = t
        clean = tempfile.mktemp(suffix=".wav")
        try:
            await _tts(english, ENGLISH_VOICES[voice], clean)
        except Exception as e:
            return {"stage": "tts", "error": str(e)}, 500
        try:
            opus_file = _degrade(clean)
            aid = _save_audio(opus_file)
            return {
                "audio_id": aid, "mime": "audio/opus",
                "detected_language": detected,
                "original_text": original,
                "translated_text": english,
            }
        except Exception as e:
            return {"stage": "ffmpeg", "error": str(e)}, 500
    finally:
        try: os.unlink(inp)
        except: pass

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
            text, _ = _transcribe(inp, lang="en")
        except Exception as e:
            return {"stage": "transcribe", "error": str(e)}, 500
        if not text:
            raise HTTPException(400, "No speech detected.")
        if target_lang:
            t = _translate(text, target_lang)
            if t: text = t
        clean = tempfile.mktemp(suffix=".wav")
        try:
            pl = profile.lower()
            if "uk" in pl and "female" in pl:   ref_profile = "UK Female"
            elif "uk" in pl and "male" in pl:   ref_profile = "UK Male"
            elif "us" in pl and "female" in pl: ref_profile = "US Female"
            else:                                ref_profile = "US Male"
            converted = _omnivoice_speak(text, ref_profile)
            import shutil as _sh
            _sh.copy(converted, clean)
        except Exception as e:
            return {"stage": "tts", "error": str(e)}, 500
        try:
            opus_file = _degrade(clean)
            aid = _save_audio(opus_file)
            return {"audio_id": aid, "mime": "audio/opus"}
        except Exception as e:
            return {"stage": "ffmpeg", "error": str(e)}, 500
    finally:
        try: os.unlink(inp)
        except: pass
