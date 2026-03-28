"""
Test partial KV cache recompute for image tokens.

Uses the same large image (4K, ~10k image tokens) with two different prompts:
  1st request (cold): populates image_kv_cache
  2nd request (warm): should reuse ~90% of image KV via partial recompute

Compares TTFT between baseline (recompute_ratio=0) and partial recompute (0.1).
"""

import os
import sys
import json
import subprocess
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "/opt/tiger/vlcache/model_vl_3b"
IMAGE_PATH = os.path.join(SCRIPT_DIR, "assets", "test_4k.png")

# Generate 4K test image if not exists
if not os.path.exists(IMAGE_PATH):
    arr = np.zeros((2160, 3840, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, 3840, dtype=np.uint8)[None, :]
    arr[:, :, 1] = np.linspace(0, 255, 2160, dtype=np.uint8)[:, None]
    arr[:, :, 2] = 128
    Image.fromarray(arr).save(IMAGE_PATH)
    print(f"Generated test image: {IMAGE_PATH}")


WORKER_CODE = r'''
import os, sys, json, torch, math
from time import perf_counter
sys.path.insert(0, os.environ["SCRIPT_DIR"])
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from nanovllm import LLM, SamplingParams

MODEL_PATH = os.environ["MODEL_PATH"]
IMAGE_PATH = os.environ["IMAGE_PATH"]
recompute_ratio = float(os.environ["RECOMPUTE_RATIO"])

def build_inputs(processor, image_path, prompt_text):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": prompt_text},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt")
    prompt_ids = inputs.input_ids[0].tolist()
    mm_input = {"pixel_values": inputs.pixel_values,
                "image_grid_thw": inputs.image_grid_thw}
    n_img = prompt_ids.count(151655)
    return prompt_ids, mm_input, n_img

processor = AutoProcessor.from_pretrained(MODEL_PATH)
sampling_params = SamplingParams(temperature=0.6, max_tokens=128)

llm = LLM(MODEL_PATH, enforce_eager=True, tensor_parallel_size=1,
           recompute_ratio=recompute_ratio)

# Build two requests with the same image, different prompts
prompt1, mm1, n_img = build_inputs(processor, IMAGE_PATH, "Describe this image in detail.")
prompt2, mm2, _     = build_inputs(processor, IMAGE_PATH, "What colors are in this image?")

R = math.ceil(recompute_ratio * n_img) if recompute_ratio > 0 else 0
print(f"Image tokens: {n_img}, R: {R}, Reused: {n_img - R}, "
      f"Prompt1: {len(prompt1)}, Prompt2: {len(prompt2)}", flush=True)

# Req 1 (cold)
out1 = llm.generate([prompt1], sampling_params, mm_inputs=[mm1], use_tqdm=False)
print(f"After req1: image_kv_cache={len(llm.scheduler.block_manager.image_kv_cache)} entries", flush=True)

# Req 2 (warm)
out2 = llm.generate([prompt2], sampling_params, mm_inputs=[mm2], use_tqdm=False)

result = {
    "n_img_tokens": n_img,
    "n_prompt1": len(prompt1),
    "n_prompt2": len(prompt2),
    "recompute_tokens": R,
    "req1_ttft": out1[0]["ttft"],
    "req1_vit":  out1[0]["vit_time"],
    "req1_text": out1[0]["text"][:150],
    "req2_ttft": out2[0]["ttft"],
    "req2_vit":  out2[0]["vit_time"],
    "req2_text": out2[0]["text"][:150],
}
print("__RESULT__" + json.dumps(result), flush=True)
'''


def run_phase(label: str, recompute_ratio: float) -> dict:
    print(f"\n{'=' * 60}")
    print(f"{label}  (recompute_ratio={recompute_ratio})")
    print("=" * 60)
    env = os.environ.copy()
    env["SCRIPT_DIR"] = SCRIPT_DIR
    env["MODEL_PATH"] = MODEL_PATH
    env["IMAGE_PATH"] = IMAGE_PATH
    env["RECOMPUTE_RATIO"] = str(recompute_ratio)
    proc = subprocess.run(
        [sys.executable, "-c", WORKER_CODE],
        env=env, capture_output=True, text=True, timeout=600,
    )
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            result = json.loads(line[len("__RESULT__"):])
        else:
            print(f"  {line}")
    if proc.returncode != 0:
        print(f"  STDERR (last 3000 chars):\n{proc.stderr[-3000:]}")
        sys.exit(1)
    if result is None:
        print("  ERROR: no result")
        print(f"  STDERR:\n{proc.stderr[-3000:]}")
        sys.exit(1)
    print(f"  [Req 1 cold] TTFT: {result['req1_ttft']*1000:8.1f} ms  "
          f"VIT: {result['req1_vit']*1000:.1f} ms")
    print(f"  [Req 1] {result['req1_text'][:100]}")
    print(f"  [Req 2 warm] TTFT: {result['req2_ttft']*1000:8.1f} ms  "
          f"VIT: {result['req2_vit']*1000:.1f} ms")
    print(f"  [Req 2] {result['req2_text'][:100]}")
    return result


def main():
    base = run_phase("Phase 1: Baseline", recompute_ratio=0.0)
    pr   = run_phase("Phase 2: Partial recompute", recompute_ratio=0.1)

    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    print(f"  Image tokens: {pr['n_img_tokens']},  R: {pr['recompute_tokens']}")
    print()
    print(f"  {'':16s} {'Req1 cold':>12s}  {'Req2 warm':>12s}")
    print(f"  {'Baseline':16s} {base['req1_ttft']*1000:9.1f} ms  "
          f"{base['req2_ttft']*1000:9.1f} ms")
    print(f"  {'Partial(0.1)':16s} {pr['req1_ttft']*1000:9.1f} ms  "
          f"{pr['req2_ttft']*1000:9.1f} ms")
    print()
    if pr["req2_ttft"] > 0:
        speedup = base["req2_ttft"] / pr["req2_ttft"]
        saving = (base["req2_ttft"] - pr["req2_ttft"]) * 1000
        print(f"  Req2 warm speedup: {speedup:.2f}x  ({saving:+.1f} ms)")
    print()
    if pr["req2_ttft"] < base["req2_ttft"] * 0.9:
        print("  PASS: Partial recompute significantly reduces warm TTFT")
    elif pr["req2_ttft"] < base["req2_ttft"]:
        print("  MARGINAL: Partial recompute slightly faster")
    else:
        print("  FAIL: Partial recompute did NOT reduce warm TTFT")
    print()
    print("  Output quality:")
    print(f"    Baseline Req2: {base['req2_text'][:120]}")
    print(f"    Partial  Req2: {pr['req2_text'][:120]}")


if __name__ == "__main__":
    main()
