import asyncio, os, subprocess, tempfile
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
    "US Female": "en-US-AriaNeural",
    "US Male":   "en-US-GuyNeural",
    "UK Female": "en-GB-SoniaNeural",
    "UK Male":   "en-GB-RyanNeural",
}

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

async def _tts(text, voice, out):
    await edge_tts.Communicate(text, voice).save(out)

def _degrade(wav_in):
    out = tempfile.mktemp(suffix=".ogg")
    subprocess.run([
        FFMPEG, "-y", "-i", wav_in,
        "-af", "highpass=f=300,lowpass=f=3400,acompressor=threshold=-18dB:ratio=4:attack=5:release=50",
        "-ar", "48000", "-ac", "1",
        "-c:a", "libopus", "-b:a", "32k", "-vbr", "on",
        "-application", "voip", "-compression_level", "10", out,
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

@app.post("/transform")
async def transform(audio: UploadFile = File(...), profile: str = Form(...)):
    if profile not in VOICES:
        raise HTTPException(400, f"Unknown profile: {profile}")
    suffix = os.path.splitext(audio.filename or "v.m4a")[1] or ".m4a"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(await audio.read())
        inp = f.name
    try:
        text = _transcribe(inp)
        if not text:
            raise HTTPException(400, "No speech detected.")
        clean = tempfile.mktemp(suffix=".wav")
        asyncio.run(_tts(text, VOICES[profile], clean))
        return FileResponse(_degrade(clean), media_type="audio/ogg", filename="accent.ogg")
    finally:
        try: os.unlink(inp)
        except: pass
