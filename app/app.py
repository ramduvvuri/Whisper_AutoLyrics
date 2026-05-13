"""AUTOLYRICS — side-by-side baseline vs fine-tuned Gradio demo."""
import os
import time
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
for m in (baseline_model.config, baseline_model.generation_config):
    m.language = "de"; m.task = "transcribe"
    m.forced_decoder_ids = None; m.suppress_tokens = []
baseline_model.generation_config.no_repeat_ngram_size = 3

base_for_ft = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=DTYPE)
ft_model = PeftModel.from_pretrained(base_for_ft, ADAPTER_REPO).to(DEVICE).eval()
for m in (ft_model.config, ft_model.generation_config):
    m.language = "de"; m.task = "transcribe"
    m.forced_decoder_ids = None; m.suppress_tokens = []
ft_model.generation_config.no_repeat_ngram_size = 3
print("Models ready.")


def load_audio(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0)


@torch.inference_mode()
def transcribe_with(model, audio_tensor, num_beams: int):
    feats = processor(audio_tensor.numpy(), sampling_rate=16000,
                      return_tensors="pt").input_features.to(DEVICE, dtype=DTYPE)
    t0 = time.perf_counter()
    ids = model.generate(feats, num_beams=num_beams, max_new_tokens=225,
                         return_dict_in_generate=True, output_scores=True)
    dt = time.perf_counter() - t0
    text = processor.batch_decode(ids.sequences, skip_special_tokens=True)[0].strip()
    # crude confidence: mean negative log-likelihood normalized
    if hasattr(ids, "sequences_scores") and ids.sequences_scores is not None:
        conf = float(torch.exp(ids.sequences_scores[0]).clamp(0, 1))
    else:
        conf = None
    return text, dt, conf


def run(audio_path: str, num_beams: int, model_choice: str):
    if audio_path is None:
        return "—", "—", "—", "—", "Please upload audio."
    audio = load_audio(audio_path)
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