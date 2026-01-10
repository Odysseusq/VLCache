import pickle
from time import perf_counter
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.encoder_cache_manager import EncoderCacheManager
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        if "Qwen2_5_VLForConditionalGeneration" in hf_config.architectures:
            self.model = Qwen2_5_VLForConditionalGeneration(hf_config)
        else:
            self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        kv_block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        encoder_block_bytes = self.block_size * hf_config.hidden_size * hf_config.torch_dtype.itemsize
        available_memory = total * config.gpu_memory_utilization - used - peak + current
        encoder_memory = available_memory * config.encoder_cache_ratio
        kv_memory = available_memory - encoder_memory
        num_encoder_blocks = int(encoder_memory) // encoder_block_bytes
        assert num_encoder_blocks > 0
        self.encoder_cache_manager = EncoderCacheManager(
            num_encoder_blocks, 
            self.block_size, 
            hf_config.hidden_size, 
            hf_config.torch_dtype, 
            device="cuda"
        )
        config.num_kvcache_blocks = int(kv_memory) // kv_block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        mm_inputs = {"pixel_values": [], "image_grid_thw": []}
        has_multimodal = False
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if seq.mm_inputs:
                has_multimodal = True
                if "pixel_values" in seq.mm_inputs and seq.mm_inputs["pixel_values"] is not None:
                    mm_inputs["pixel_values"].append(seq.mm_inputs["pixel_values"])
                if "image_grid_thw" in seq.mm_inputs and seq.mm_inputs["image_grid_thw"] is not None:
                    mm_inputs["image_grid_thw"].append(seq.mm_inputs["image_grid_thw"])
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)

        if has_multimodal:
            for key in ["pixel_values", "image_grid_thw"]:
                if mm_inputs[key]:
                    mm_inputs[key] = torch.cat(mm_inputs[key], dim=0).cuda(non_blocking=True)
                else:
                    mm_inputs.pop(key, None)
        else:
            mm_inputs = None

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions, mm_inputs

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def _process_visual_cache(self, mm_inputs: dict) -> tuple[torch.Tensor, float]:
        pixel_values = mm_inputs["pixel_values"]
        grid_thw = mm_inputs["image_grid_thw"]
        img_lens = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
        pixel_values_list = torch.split(pixel_values, img_lens)
        image_embeds_list = [None] * len(img_lens)
        miss_indices = []
        miss_pixel_values_list = []
        miss_grid_thw_list = []
        miss_hashes = []
        miss_out_lens = []
        vit_time = 0.0
        spatial_merge_size = self.model.visual.spatial_merge_size
        for i, (pv, g_thw) in enumerate(zip(pixel_values_list, grid_thw)):
            h = EncoderCacheManager.compute_hash(pv, g_thw)
            t, height, width = g_thw.tolist()
            out_h = height // spatial_merge_size
            out_w = width // spatial_merge_size
            output_len = t * out_h * out_w
            block_ids = self.encoder_cache_manager.get_block_ids(h)
            if block_ids:
                image_embeds_list[i] = self.encoder_cache_manager.read(block_ids, output_len)
            else:
                miss_indices.append(i)
                miss_pixel_values_list.append(pv)
                miss_grid_thw_list.append(g_thw)
                miss_hashes.append(h)
                miss_out_lens.append(output_len)

        if miss_indices:
            miss_pv = torch.cat(miss_pixel_values_list, dim=0)
            miss_g = torch.stack(miss_grid_thw_list, dim=0)
            torch.cuda.synchronize()
            st = perf_counter()
            miss_embeds = self.model.get_visual_features(miss_pv, miss_g)
            torch.cuda.synchronize()
            vit_time = perf_counter() - st
            miss_embeds_split = torch.split(miss_embeds, miss_out_lens)
            for i, idx in enumerate(miss_indices):
                emb = miss_embeds_split[i]
                image_embeds_list[idx] = emb
                h = miss_hashes[i]
                out_len = miss_out_lens[i]
                if self.encoder_cache_manager.can_allocate(out_len):
                    block_ids = self.encoder_cache_manager.allocate(h, out_len)
                    self.encoder_cache_manager.write(block_ids, emb)
        return torch.cat(image_embeds_list, dim=0), vit_time

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool, mm_inputs: dict = None):
        visual_embeds = None
        vit_time = 0.0
        if is_prefill and mm_inputs:
            visual_embeds, vit_time = self._process_visual_cache(mm_inputs)
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            hidden_states = self.model(input_ids, positions, visual_embeds)
            if visual_embeds is None:
                vit_time = getattr(self.model, "last_vit_time", 0.0)
            return self.model.compute_logits(hidden_states), vit_time
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs]), 0.0

    def run(self, seqs: list[Sequence], is_prefill: bool) -> tuple[list[int], float]:
        if is_prefill:
            input_ids, positions, mm_inputs = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)
            mm_inputs = None
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits, vit_time = self.run_model(input_ids, positions, is_prefill, mm_inputs)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids, vit_time

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
