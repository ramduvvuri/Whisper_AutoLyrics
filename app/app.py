"""AUTOLYRICS — side-by-side baseline vs fine-tuned Gradio demo."""
import os
import time
import subprocess
import tempfile
import wave
from pathlib import Path
import numpy as np
import torch
import torchaudio
import gradio as gr
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

BASE_MODEL = "openai/whisper-small"
ADAPTER_REPO = os.environ.get(
    "ADAPTER_REPO", "Petercoder/autolyrics-whisper-small-lora")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ---------- Lazy model loading ----------
print(f"Loading models on {DEVICE}…")
processor = WhisperProcessor.from_pretrained(BASE_MODEL)

baseline_model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=DTYPE).to(DEVICE).eval()
# Configure generation_config only (not model.config) — avoids reconciliation
# warnings in transformers 4.47+ when both configs hold conflicting values.
baseline_model.generation_config.language = "de"
baseline_model.generation_config.task = "transcribe"
baseline_model.generation_config.forced_decoder_ids = None
baseline_model.generation_config.no_repeat_ngram_size = 3

base_for_ft = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=DTYPE)
ft_model = PeftModel.from_pretrained(base_for_ft, ADAPTER_REPO).to(DEVICE).eval()
ft_model.generation_config.language = "de"
ft_model.generation_config.task = "transcribe"
ft_model.generation_config.forced_decoder_ids = None
ft_model.generation_config.no_repeat_ngram_size = 3
print("Models ready.")


def load_audio(path: str) -> torch.Tensor:
    """Load any browser-uploaded audio format → 16 kHz mono float32 tensor.

    Strategy (two-stage, zero libsndfile dependency):
      1. ffmpeg transcodes ANY browser format (webm/opus, ogg, mp3, m4a, wav)
         into a clean 16-bit PCM WAV at 16 kHz mono.  ffmpeg handles every
         container/codec that browsers produce, including Gradio mic recordings.
      2. Python's built-in `wave` module reads the raw PCM bytes directly.
         This **completely bypasses soundfile / libsndfile**, which cannot decode
         webm, ogg/opus, or partially-encoded containers and raises
         ``soundfile.LibsndfileError: Format not recognised`` on HF Spaces.

    ffmpeg is pre-installed on HF Spaces via packages.txt — no extra Python
    package is needed.  `wave` and `numpy` are always available.

    Returns
    -------
    torch.Tensor
        1-D float32 waveform on CPU, normalised to [-1, 1], at 16 000 Hz.
    """
    src = Path(path)
    if not src.exists() or src.stat().st_size == 0:
        raise ValueError(f"Audio file missing or empty: {path}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # ── Step 1: transcode to clean PCM WAV via ffmpeg ──────────────────
        result = subprocess.run(
            [
                "ffmpeg", "-y",         # overwrite without prompting
                "-i", str(src),         # any browser-upload format
                "-ac", "1",            # force mono
                "-ar", "16000",        # resample to 16 kHz
                "-sample_fmt", "s16",  # 16-bit signed PCM
                "-f", "wav",           # output container: wav
                tmp_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace").strip().splitlines()
            raise RuntimeError(
                f"ffmpeg failed (code {result.returncode}): "
                f"{err[-1] if err else 'unknown error'}"
            )

        # ── Step 2: read PCM bytes with stdlib `wave` — no soundfile ───────
        with wave.open(tmp_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth  = wf.getsampwidth()  # bytes per sample: 2 for s16
            framerate  = wf.getframerate()
            n_frames   = wf.getnframes()
            if n_frames == 0:
                raise ValueError("ffmpeg produced an empty audio file.")
            raw = wf.readframes(n_frames)

        # Parse raw bytes → float32 in [-1, 1]
        # ffmpeg guarantees s16, but use sampwidth defensively.
        if sampwidth == 2:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

        # Mix down multi-channel (guard: -ac 1 already handles this)
        if n_channels > 1:
            arr = arr.reshape(-1, n_channels).mean(axis=1)

        wav = torch.from_numpy(arr.copy())  # copy() avoids non-writable buffer warning

        # Resample if framerate drifted (guard: -ar 16000 already handles this)
        if framerate != 16000:
            wav = torchaudio.functional.resample(
                wav.unsqueeze(0), framerate, 16000
            ).squeeze(0)

        return wav  # 1-D float32 CPU tensor

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@torch.inference_mode()
def transcribe_with(model, audio_tensor, num_beams: int):
    feats = processor(audio_tensor.numpy(), sampling_rate=16000,
                      return_tensors="pt").input_features.to(DEVICE, dtype=DTYPE)
    t0 = time.perf_counter()
    # ── KEY FIX: pass input_features as a KEYWORD argument. ────────────────
    # PeftModelForSeq2SeqLM.generate() only accepts **kwargs (no positional
    # args beyond self). Passing `feats` positionally raises:
    #   TypeError: generate() takes 1 positional argument but 2 were given
    # WhisperForConditionalGeneration also accepts it as a keyword, so this
    # call is correct for BOTH the bare baseline model and the PEFT wrapper.
    ids = model.generate(
        input_features=feats,
        num_beams=num_beams,
        max_new_tokens=225,
        return_dict_in_generate=True,
        output_scores=True,
    )
    dt = time.perf_counter() - t0
    # ids is GenerateBeamEncoderDecoderOutput when return_dict_in_generate=True.
    # .sequences holds the token-id tensor; decode it to text.
    text = processor.batch_decode(ids.sequences, skip_special_tokens=True)[0].strip()
    # Confidence proxy: exponentiate the beam score (sum of log-probs).
    # sequences_scores is None for greedy/num_beams=1 — guard accordingly.
    if hasattr(ids, "sequences_scores") and ids.sequences_scores is not None:
        conf = float(torch.exp(ids.sequences_scores[0]).clamp(0, 1))
    else:
        conf = None
    return text, dt, conf


def run(audio_path: str, num_beams: int, model_choice: str):
    if audio_path is None:
        return "—", "—", "—", "—", "Please upload audio."

    # Load and decode audio — raises ValueError/RuntimeError on bad input.
    try:
        audio = load_audio(audio_path)
    except (ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        err_msg = f"⚠️ Audio error: {exc}"
        return err_msg, err_msg, "—", "—", "Audio could not be decoded — try a different file."
    except Exception as exc:  # noqa: BLE001
        err_msg = f"⚠️ Unexpected error loading audio: {exc}"
        return err_msg, err_msg, "—", "—", "Audio could not be decoded — try a different file."

    if audio.numel() == 0:
        return "—", "—", "—", "—", "⚠️ Audio file appears to be empty or silent."

    duration = audio.shape[-1] / 16000

    if model_choice == "Baseline only":
        b_text, b_dt, b_conf = transcribe_with(baseline_model, audio, num_beams)
        return b_text, "—", f"{b_dt:.2f}s · RTF {b_dt/duration:.2f}", "—", \
               f"Audio: {duration:.1f}s"
    if model_choice == "Fine-tuned only":
        f_text, f_dt, f_conf = transcribe_with(ft_model, audio, num_beams)
        return "—", f_text, "—", f"{f_dt:.2f}s · RTF {f_dt/duration:.2f}", \
               f"Audio: {duration:.1f}s"
    # both
    b_text, b_dt, _ = transcribe_with(baseline_model, audio, num_beams)
    f_text, f_dt, _ = transcribe_with(ft_model, audio, num_beams)
    return b_text, f_text, \
           f"{b_dt:.2f}s · RTF {b_dt/duration:.2f}", \
           f"{f_dt:.2f}s · RTF {f_dt/duration:.2f}", \
           f"Audio: {duration:.1f}s"

# ---------- UI ----------
THEME = gr.themes.Monochrome(
    primary_hue="neutral", neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#000000",
    body_text_color="#fafafa",
    block_background_fill="#0a0a0a",
    block_border_color="#1a1a1a",
    button_primary_background_fill="#fafafa",
    button_primary_text_color="#000000",
)

CSS = """
#title { letter-spacing: -0.02em; }
.gradio-container { max-width: 1100px !important; }
footer { display: none !important; }
"""

with gr.Blocks(theme=THEME, css=CSS, title="AUTOLYRICS") as demo:
    gr.HTML("""
      <div style='padding: 28px 0 8px 0;'>
        <h1 id='title' style='font-size: 44px; font-weight: 600; margin: 0;'>
          AUTOLYRICS
        </h1>
        <p style='color: #888; margin: 8px 0 0 0; font-size: 15px;'>
          Transcribing the voice inside music. Whisper-small fine-tuned with LoRA on singing.
        </p>
      </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(type="filepath", label="Upload or record",
                             sources=["upload", "microphone"])
            with gr.Row():
                beams = gr.Slider(1, 8, value=5, step=1, label="Beam search width")
                choice = gr.Radio(
                    ["Both (compare)", "Baseline only", "Fine-tuned only"],
                    value="Both (compare)", label="Mode")
            run_btn = gr.Button("Transcribe", variant="primary")
            meta = gr.Markdown("")

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Baseline · Whisper-small")
                base_out = gr.Textbox(lines=4, show_label=False,
                                      placeholder="Baseline transcription will appear here…")
                base_meta = gr.Markdown("")
            with gr.Group():
                gr.Markdown("### Fine-tuned · AUTOLYRICS (LoRA)")
                ft_out = gr.Textbox(lines=4, show_label=False,
                                    placeholder="Fine-tuned transcription will appear here…")
                ft_meta = gr.Markdown("")

    gr.Examples(
        examples=[
            ["examples/pop_clip.wav",   5, "Both (compare)"],
            ["examples/ballad_clip.wav",5, "Both (compare)"],
            ["examples/rap_clip.wav",   5, "Both (compare)"],
        ],
        inputs=[audio, beams, choice],
        outputs=[base_out, ft_out, base_meta, ft_meta, meta],
        fn=run,
        cache_examples=False,
    )

    run_btn.click(
        run,
        inputs=[audio, beams, choice],
        outputs=[base_out, ft_out, base_meta, ft_meta, meta],
    )

demo.queue(max_size=12).launch(
    server_name="0.0.0.0",   # Required: HF Spaces proxy expects this binding
    server_port=7860,         # HF Spaces standard port
    show_error=True,          # Surface tracebacks in the UI during debugging
)