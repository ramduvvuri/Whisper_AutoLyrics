"""LoRA fine-tuning of Whisper for singing transcription."""
from __future__ import annotations
from pathlib import Path
import argparse
import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from autolyrics.training.collator import DataCollatorSpeechSeq2SeqWithPadding
from autolyrics.data.dataset import load_split_as_hf, make_prepare_fn
from autolyrics.eval.metrics import normalize_for_scoring
import jiwer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",   default="openai/whisper-small")
    p.add_argument("--train_csv",    default="data/manifests/train.csv")
    p.add_argument("--val_csv",      default="data/manifests/val.csv")
    p.add_argument("--output_dir",   default="outputs/checkpoints/lora_run1")
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--per_device_bs",type=int,   default=8)
    p.add_argument("--grad_accum",   type=int,   default=2)
    p.add_argument("--eval_bs",      type=int,   default=4)
    p.add_argument("--lora_r",       type=int,   default=32)
    p.add_argument("--lora_alpha",   type=int,   default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", nargs="+",
                   default=["q_proj", "v_proj"])
    p.add_argument("--use_8bit",     action="store_true")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--wandb_project", default="autolyrics")
    p.add_argument("--run_name",     default=None)
    p.add_argument("--num_workers",  type=int, default=2)
    return p.parse_args()


def build_compute_metrics(processor):
    """Returns a HF-compatible compute_metrics that runs WER/CER."""
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        # Replace -100 with pad_token_id for decoding
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True)
        # normalize
        pred_str  = [normalize_for_scoring(s) or "<empty>" for s in pred_str]
        label_str = [normalize_for_scoring(s) or "<empty>" for s in label_str]
        return {
            "wer": float(jiwer.wer(label_str, pred_str)),
            "cer": float(jiwer.cer(label_str, pred_str)),
        }
    return compute_metrics


def detect_main_language(csv_path):
    import pandas as pd
    if "language" not in pd.read_csv(csv_path, nrows=1).columns:
        return None
    df = pd.read_csv(csv_path)
    lang_counts = df["language"].value_counts()
    if len(lang_counts) == 0:
        return None
    lang = lang_counts.idxmax()
    print(f"[LANG] dominant dataset language: {lang}")
    return lang


def main():
    try:
        print("[DEBUG] entered main()")
        args = parse_args()
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.run_name:
            os.environ.setdefault("WANDB_NAME", args.run_name)

        torch.manual_seed(args.seed)

        main_language = detect_main_language(args.train_csv)

    # ---------- processor + raw datasets ----------
        processor = WhisperProcessor.from_pretrained(
            args.base_model,
            language=main_language,
            task="transcribe",
        )
        print(f"[WHISPER] training language = {main_language}")
        print("[DEBUG] processor loaded")

        train_ds = load_split_as_hf(args.train_csv)
        val_ds   = load_split_as_hf(args.val_csv)
        print("[DEBUG] raw datasets loaded")
        prepare = make_prepare_fn(processor)
        train_ds = train_ds.map(prepare, remove_columns=train_ds.column_names,
                                num_proc=1)
        val_ds   = val_ds.map(prepare,   remove_columns=val_ds.column_names,
                                num_proc=1)
        print("[DEBUG] datasets mapped")
        print(f"[DATA] train={len(train_ds)} val={len(val_ds)}")

        # ---------- base model ----------
        load_kwargs = {}
        if args.use_8bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["device_map"] = "auto"

        model = WhisperForConditionalGeneration.from_pretrained(
            args.base_model, **load_kwargs)
        print("[DEBUG] model loaded")
        print(f"[CUDA] available = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[CUDA] device = {torch.cuda.get_device_name(0)}")

        if args.use_8bit:
            model = prepare_model_for_kbit_training(model)

        # Critical Whisper FT settings
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        model.config.use_cache = False
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = []
        model.generation_config.language = main_language
        model.generation_config.task = "transcribe"

        # ---------- LoRA wrap ----------
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            bias="none",
            task_type=None,                # Whisper isn't in PEFT TASK_TYPE enum
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        # Sanity: should print something like "trainable: 1.2M / 244M (~0.5%)"
        # Required: gradient checkpointing detaches input embeddings from the graph;
        # enable_input_require_grads() hooks them back so the first backward pass works.
        model.enable_input_require_grads()
        model.train()
        print("[DEBUG] peft attached")

        # ---------- collator ----------
        collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )
        print("[DEBUG] collator created")

        # ---------- training args ----------
        targs = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_bs,
            per_device_eval_batch_size=args.eval_bs,
            gradient_accumulation_steps=args.grad_accum,
            gradient_checkpointing=True,
            learning_rate=args.lr,
            warmup_steps=50,
            num_train_epochs=args.epochs,
            fp16=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            logging_steps=25,
            logging_first_step=True,
            predict_with_generate=True,
            generation_max_length=225,
            generation_num_beams=1,         # fast eval during training
            report_to=["wandb"],            # restore wandb logging
            run_name=args.run_name or Path(args.output_dir).name,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            remove_unused_columns=False,
            label_names=["labels"],         # required for PEFT
            seed=args.seed,
        )

        # ---------- trainer ----------
        trainer = Seq2SeqTrainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            compute_metrics=build_compute_metrics(processor),
            tokenizer=processor.feature_extractor,   # needed for save_pretrained
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        print("[DEBUG] trainer initialized")
        print("[TRAINER] initialized")

        print("[DEBUG] starting trainer.train()")
        print("[TRAIN] starting trainer.train()")
        trainer.train()
        print("[TRAIN] training finished")
        print("[DEBUG] trainer.train() finished")

        # save final adapter
        final_dir = Path(args.output_dir) / "final"
        model.save_pretrained(str(final_dir))
        processor.save_pretrained(str(final_dir))
        print(f"Saved adapter + processor to {final_dir}")

        # save best checkpoint adapter to a stable path
        best_dir = Path(args.output_dir) / "best"
        model.save_pretrained(str(best_dir))   # trainer already loaded best via load_best_model_at_end
        processor.save_pretrained(str(best_dir))
        print(f"[BEST] saved best adapter to {best_dir}")

    except Exception as e:
        import traceback
        print("\n[CRASH]")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()