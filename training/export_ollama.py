#!/usr/bin/env python3
"""Merge Axolotl QLoRA adapter → GGUF + Ollama Modelfile.

Pipeline:
  1. Load base + LoRA, merge_and_unload → HF dir
  2. llama.cpp convert_hf_to_gguf.py → .gguf
  3. optional llama-quantize
  4. write Modelfile; optional `ollama create`

Examples:
  docker compose run --rm train python export_ollama.py
  python export_ollama.py --adapter outputs/qwen3-4b-cfm-cot
  cd outputs/ollama-qwen3-cfm-cot && ollama create qwen3-4b-cfm-cot -f Modelfile
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_BASE = "Qwen/Qwen3-4B"
DEFAULT_ADAPTER = Path("outputs/qwen3-4b-cfm-cot")
DEFAULT_OUT = Path("outputs/ollama-qwen3-cfm-cot")
DEFAULT_NAME = "qwen3-4b-cfm-cot"
LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp.git"


def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def find_adapter_dir(path: Path) -> Path:
    """Prefer final adapter; else newest checkpoint-* with adapter files."""
    if (path / "adapter_config.json").is_file():
        return path
    checkpoints = sorted(
        (
            p
            for p in path.glob("checkpoint-*")
            if (p / "adapter_config.json").is_file()
        ),
        key=lambda p: p.stat().st_mtime,
    )
    if checkpoints:
        print(f"Using checkpoint adapter: {checkpoints[-1]}")
        return checkpoints[-1]
    die(
        f"No adapter_config.json under {path} "
        "(train finished? path correct?)"
    )


def merge_lora(
    base_model: str,
    adapter_dir: Path,
    merged_dir: Path,
    *,
    cpu: bool,
) -> None:
    if merged_dir.exists() and (merged_dir / "config.json").is_file():
        print(f"Merged HF already exists, skip: {merged_dir}")
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_map = "cpu" if cpu else "auto"
    dtype = torch.bfloat16
    print(f"Loading base {base_model} (device_map={device_map}) ...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"Loading adapter {adapter_dir} ...")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    print("Merging LoRA into base ...")
    model = model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged HF → {merged_dir}")
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_llama_cpp(tools_dir: Path) -> Path:
    """Shallow-clone llama.cpp; return convert_hf_to_gguf.py path."""
    tools_dir.mkdir(parents=True, exist_ok=True)
    repo = tools_dir / "llama.cpp"
    script = repo / "convert_hf_to_gguf.py"
    if script.is_file():
        return script
    if repo.exists():
        shutil.rmtree(repo)
    print(f"Cloning {LLAMA_CPP_REPO} (depth=1) → {repo}")
    if not shutil.which("git"):
        die("git required to fetch llama.cpp convert script")
    subprocess.run(
        ["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(repo)],
        check=True,
    )
    if not script.is_file():
        die(f"convert_hf_to_gguf.py missing after clone: {repo}")
    return script


def ensure_convert_deps() -> None:
    missing: list[str] = []
    for pkg, mod in (
        ("gguf", "gguf"),
        ("sentencepiece", "sentencepiece"),
        ("protobuf", "google.protobuf"),
    ):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if not missing:
        return
    print(f"Installing convert deps: {missing}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *missing],
        check=True,
    )


def convert_to_gguf(
    merged_dir: Path,
    gguf_path: Path,
    convert_script: Path,
    out_type: str,
) -> None:
    if gguf_path.is_file():
        print(f"GGUF already exists, skip: {gguf_path}")
        return
    gguf_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(convert_script),
        str(merged_dir),
        "--outfile",
        str(gguf_path),
        "--outtype",
        out_type,
    ]
    print("Running:", " ".join(cmd))
    env = os.environ.copy()
    # Prefer pip gguf; also allow llama.cpp bundled gguf-py
    gguf_py = convert_script.parent / "gguf-py"
    if gguf_py.is_dir():
        env["PYTHONPATH"] = (
            f"{gguf_py}{os.pathsep}{env['PYTHONPATH']}"
            if env.get("PYTHONPATH")
            else str(gguf_py)
        )
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        die(
            f"GGUF convert failed (exit {e.returncode}). "
            "Try: pip install gguf sentencepiece protobuf numpy"
        )


def quantize_gguf(src: Path, dst: Path, quant: str) -> Path:
    if quant in {"f16", "fp16", "bf16", "f32", "q8_0"}:
        # q8_0 can already be from convert --outtype; treat as no extra step
        if quant == "q8_0" and src.name.endswith("-q8_0.gguf"):
            return src
        if quant in {"f16", "fp16", "bf16", "f32"}:
            return src
    if dst.is_file():
        print(f"Quant GGUF exists, skip: {dst}")
        return dst
    binary = shutil.which("llama-quantize") or shutil.which("quantize")
    if not binary:
        print(
            "WARN: llama-quantize not in PATH — ship F16/outtype GGUF. "
            "Build llama.cpp for q4_k_m, or ollama create from F16 "
            "(bigger download)."
        )
        return src
    cmd = [binary, str(src), str(dst), quant.upper()]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return dst


def write_modelfile(out_dir: Path, gguf_name: str, num_ctx: int) -> Path:
    # Must match Axolotl `type: alpaca` (empty input) used in training; a chat
    # template or SYSTEM prompt here makes the model fall back to base behavior.
    # Ollama uses Go templates; double braces for Python .format
    body = """FROM ./{gguf_name}

TEMPLATE \"\"\"Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{{{{ .Prompt }}}}

### Response:
\"\"\"

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "### Instruction:"
PARAMETER stop "</Output>"
PARAMETER num_ctx {num_ctx}
PARAMETER temperature 0.2
PARAMETER top_p 0.9
"""
    path = out_dir / "Modelfile"
    path.write_text(
        body.format(gguf_name=gguf_name, num_ctx=num_ctx),
        encoding="utf-8",
    )
    print(f"Wrote {path}")
    return path


def ollama_create(name: str, modelfile: Path) -> None:
    binary = shutil.which("ollama")
    if not binary:
        die("ollama not in PATH")
    cmd = [binary, "create", name, "-f", str(modelfile.name)]
    print("Running:", " ".join(cmd), f"(cwd={modelfile.parent})")
    subprocess.run(cmd, check=True, cwd=str(modelfile.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export QLoRA adapter to Ollama (merge → GGUF → Modelfile)"
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=None,
        help="HF merge output (default: <out-dir>/merged)",
    )
    parser.add_argument(
        "--outtype",
        default="f16",
        choices=["f32", "f16", "bf16", "q8_0", "auto"],
        help="convert_hf_to_gguf --outtype",
    )
    parser.add_argument(
        "--quant",
        default="q5_k_m",
        help="llama-quantize type after convert. Use f16 to skip. "
        "Needs llama-quantize in PATH; else keeps --outtype GGUF.",
    )
    parser.add_argument("--ollama-name", default=DEFAULT_NAME, help="Ollama model name; also the GGUF file stem")
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Merge with device_map=auto (default = CPU, safer on 6GB)",
    )
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-gguf", action="store_true")
    parser.add_argument(
        "--create",
        action="store_true",
        help="Run `ollama create` after writing Modelfile",
    )
    parser.add_argument(
        "--tools-dir",
        type=Path,
        default=Path("outputs/ollama-tools"),
        help="Cache llama.cpp clone here",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = args.merged_dir or (out_dir / "merged")

    if not args.skip_merge:
        adapter_dir = find_adapter_dir(args.adapter)
        merge_lora(args.base_model, adapter_dir, merged_dir, cpu=not args.gpu)
    elif not args.skip_gguf and not (merged_dir / "config.json").is_file():
        die(f"--skip-merge but missing merged model at {merged_dir}")

    stem = args.ollama_name.replace(":", "-").replace("/", "-")
    raw_gguf = out_dir / f"{stem}-{args.outtype}.gguf"
    quant = args.quant.lower()
    want_extra_quant = quant not in {
        "f16",
        "fp16",
        "bf16",
        "f32",
        args.outtype.lower(),
    }
    final_gguf = (
        out_dir / f"{stem}-{quant}.gguf" if want_extra_quant else raw_gguf
    )

    if not args.skip_gguf:
        ensure_convert_deps()
        convert_script = ensure_llama_cpp(args.tools_dir)
        convert_to_gguf(merged_dir, raw_gguf, convert_script, args.outtype)
        final_gguf = quantize_gguf(raw_gguf, final_gguf, quant)
    else:
        candidates = list(out_dir.glob("*.gguf"))
        if final_gguf.is_file():
            pass
        elif raw_gguf.is_file():
            final_gguf = raw_gguf
        elif candidates:
            final_gguf = max(candidates, key=lambda p: p.stat().st_mtime)
            print(f"Using existing GGUF: {final_gguf}")
        else:
            die("--skip-gguf but no .gguf in out-dir")

    modelfile = write_modelfile(out_dir, final_gguf.name, args.num_ctx)

    print()
    print("=== Ollama ===")
    print(f"  GGUF:      {final_gguf}")
    print(f"  Modelfile: {modelfile}")
    print(f"  Create:    cd {out_dir} && ollama create {args.ollama_name} -f Modelfile")
    print(f"  Run:       ollama run {args.ollama_name}")
    print()

    if args.create:
        ollama_create(args.ollama_name, modelfile)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
