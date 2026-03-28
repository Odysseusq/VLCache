import atexit
from dataclasses import fields
from time import perf_counter, time
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.encoder_cache_manager import EncoderCacheManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams, mm_inputs: dict = None):
        image_hashes = None
        if mm_inputs:
            pixel_values = mm_inputs.get("pixel_values")
            grid_thw = mm_inputs.get("image_grid_thw")
            if pixel_values is not None and grid_thw is not None:
                img_lens = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
                pixel_values_list = torch.split(pixel_values, img_lens)
                image_hashes = [EncoderCacheManager.compute_hash(pv, g_thw) for pv, g_thw in zip(pixel_values_list, grid_thw)]
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params, mm_inputs, image_hashes)
        self.scheduler.add(seq)

    def _do_image_kv_copy(self, seqs: list[Sequence]):
        """Copy reused image KV cache from old blocks to new blocks for partial recompute sequences."""
        block_manager = self.scheduler.block_manager
        block_size = block_manager.block_size
        for seq in seqs:
            if not seq.is_partial_recompute:
                continue
            image_hash = seq.image_hashes[0]
            entry = block_manager.get_image_kv_entry(image_hash)
            if entry is None:
                continue
            img_start, img_end = seq.image_token_range
            R = seq.num_recompute_tokens
            reuse_start = img_start + R
            reuse_end = img_end
            if reuse_start >= reuse_end:
                continue
            # Compute old slots (source: from stored image KV entry)
            old_slots = []
            for p in range(reuse_start, reuse_end):
                # Old sequence had image at same relative positions
                old_p = entry.img_start + (p - img_start)
                block_idx = old_p // block_size
                offset = old_p % block_size
                old_slots.append(entry.block_table[block_idx] * block_size + offset)
            # Compute new slots (destination: new sequence's blocks)
            new_slots = []
            for p in range(reuse_start, reuse_end):
                block_idx = p // block_size
                offset = p % block_size
                new_slots.append(seq.block_table[block_idx] * block_size + offset)
            self.model_runner.call("copy_image_kv", old_slots, new_slots)

    def _store_image_kv(self, seqs: list[Sequence]):
        """Store image KV block info after prefill for future reuse."""
        block_manager = self.scheduler.block_manager
        image_token_id = self.scheduler.image_token_id
        for seq in seqs:
            if not seq.image_hashes or not seq.block_table:
                continue
            if seq.image_token_range is None:
                seq.find_image_token_range(image_token_id)
            if seq.image_token_range is not None:
                block_manager.store_image_kv(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        if is_prefill:
            # Copy reused image KV before model forward pass
            self._do_image_kv_copy(seqs)
        token_ids, vit_time = self.model_runner.call("run", seqs, is_prefill)
        if is_prefill:
            now = time()
            for seq in seqs:
                if seq.mm_inputs:
                    seq.vit_time = vit_time
                seq.ttft = now - seq.start_time
            # Store image KV for future reuse
            self._store_image_kv(seqs)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [seq for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        mm_inputs: list[dict] | None = None,
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if mm_inputs is None:
            mm_inputs = [None] * len(prompts)
        for prompt, sp, mp in zip(prompts, sampling_params, mm_inputs):
            self.add_request(prompt, sp, mp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq in output:
                outputs[seq.seq_id] = seq
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{
            "text": self.tokenizer.decode(seq.completion_token_ids),
            "token_ids": seq.completion_token_ids,
            "vit_time": seq.vit_time,
            "ttft": seq.ttft,
        } for seq in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
