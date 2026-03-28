import math
from collections import deque
from dataclasses import dataclass

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0

    def reset(self):
        self.ref_count = 1


@dataclass
class ImageKVEntry:
    block_table: list[int]     # full block table of the source sequence
    img_start: int             # image start position in the source sequence
    img_end: int               # image end position (exclusive)


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, recompute_ratio: float = 0.0):
        self.block_size = block_size
        self.recompute_ratio = recompute_ratio
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.image_kv_cache: dict[int, ImageKVEntry] = {}

    def _allocate_block(self) -> Block:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        block.reset()
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        for _ in range(seq.num_blocks):
            block = self._allocate_block()
            seq.block_table.append(block.block_id)

    def try_partial_recompute(self, seq: Sequence, image_token_id: int = 151655):
        """Check if this sequence can reuse image KV cache with partial recompute."""
        if self.recompute_ratio <= 0 or not seq.image_hashes:
            return False
        seq.find_image_token_range(image_token_id)
        if seq.image_token_range is None:
            return False
        image_hash = seq.image_hashes[0]  # single image
        entry = self.image_kv_cache.get(image_hash)
        if entry is None:
            return False
        img_start, img_end = seq.image_token_range
        N = img_end - img_start
        old_N = entry.img_end - entry.img_start
        if N != old_N:
            return False
        R = math.ceil(self.recompute_ratio * N)
        seq.num_recompute_tokens = R
        seq.is_partial_recompute = True
        return True

    def get_image_kv_entry(self, image_hash: int) -> ImageKVEntry | None:
        return self.image_kv_cache.get(image_hash)

    def store_image_kv(self, seq: Sequence):
        """Store image KV block info after prefill for future reuse."""
        if not seq.image_hashes or seq.image_token_range is None:
            return
        image_hash = seq.image_hashes[0]
        img_start, img_end = seq.image_token_range
        # Free old entry's retained blocks if replacing
        old_entry = self.image_kv_cache.get(image_hash)
        if old_entry is not None:
            self._release_image_blocks(old_entry)
        # Retain blocks that contain image tokens by incrementing ref_count
        start_block = img_start // self.block_size
        end_block = (img_end - 1) // self.block_size + 1
        for i in range(start_block, end_block):
            self.blocks[seq.block_table[i]].ref_count += 1
        self.image_kv_cache[image_hash] = ImageKVEntry(
            block_table=list(seq.block_table),
            img_start=img_start,
            img_end=img_end,
        )

    def _release_image_blocks(self, entry: ImageKVEntry):
        """Release retained image KV blocks."""
        start_block = entry.img_start // self.block_size
        end_block = (entry.img_end - 1) // self.block_size + 1
        for i in range(start_block, end_block):
            block_id = entry.block_table[i]
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        if len(seq) % self.block_size == 1:
            block = self._allocate_block()
            block_table.append(block.block_id)
