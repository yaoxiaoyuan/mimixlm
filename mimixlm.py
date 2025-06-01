#encoding=utf-8
#◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆
#MIMIXLM: PYTHON SINGLE-FILE LLM IMPLEMENTATION FROM SCRATCH
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Author: Xiaoyuan Yao
# GitHub: https://github.com/yaoxiaoyuan/mimixlm/
# Contact: yaoxiaoyuan1990@gmail.com
# Created: Thu Oct 24 19:35:17 2024
# License: MIT
# Version: 0.1.0
#
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Core Features
# > FROM-SCRATCH IMPLEMENTATION(Based on pytorch)
# > Modern LLM Architectures: Attention(MHA/MQA/GQA) • GLU • RoPE • etc.
# > Full Pipeline demo: Processing → Training (Pretrain/SFT/DPO) → Inference
# > Basic Multimodal Support: Image captioning/vqa (proof-of-concept)
#   Welcome to explore!
#
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ► IDEAL FOR:
#    └─ Students and practitioners curious about transformer LLM mechanics
#    └─ Anyone who prefers doing from scratch over high-level libraries
#    └─ Hands-on learners focused on practice rather than pure theory
#
# ► NOT FOR:
#    ├─ Rapid app development
#    └─ Production-ready APIs
#    └─ Beginners lacking Python/math fundamentals
#
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ⚠️  WARNING  ⚠️
# - Performance NOT optimized
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

import sys
import argparse
import os
import glob
import io
import shutil
import logging
from functools import partial
import math
import time
import datetime
import json
import re
import base64
import csv
import random
import heapq
from collections import defaultdict
import traceback
import regex
import numpy as np
from PIL import Image
import zipfile
import gzip
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import multiprocessing as mp
from torchvision.transforms import Lambda, Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.distributed import init_process_group, destroy_process_group

IMPORT_MATPLOTLIB_SUCCESS = False
try:
    import matplotlib.pyplot as plt
    IMPORT_MATPLOTLIB_SUCCESS = True
except:
    pass

IMPORT_TIKTOKEN_SUCCESS = False
try:
    import tiktoken
    IMPORT_TIKTOKEN_SUCCESS = True
except:
    pass

"""
Define logger
"""
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logger")
if not os.path.exists(LOG_DIR):
    try:
        os.mkdir(LOG_DIR)
    except:
        pass

class ColorFormatter(logging.Formatter):
    """Custom logging formatter that adds ANSI color codes to log messages based on their level.

    Colors different log levels for better visual distinction in terminals that support ANSI colors.
    """
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m'
    }
    RESET = '\033[0m'

    def format(self, record):
        """Formats the log record, prepending appropriate color code and appending reset code.

        Args:
            record (LogRecord): The log record to be formatted

        Returns:
            str: Colorized log message string
        """
        log_color = self.COLORS.get(record.levelname, self.RESET)
        log_message = super().format(record)
        return f"{log_color}{log_message}{self.RESET}"

def build_logger():
    """
    Initialize and configure a logger with console output handling.

    Returns:
        logging.Logger: Configured logger instance with console handler
    """
    logger = logging.getLogger(os.path.basename(__file__))
    logger.setLevel(logging.DEBUG)

    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_format = ColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)

    logger.addHandler(console_handler)
    return logger


def add_file_handlers(log_path):
    """
    Add file handlers to global logger for writing DEBUG+ logs to specified path

    Parameters:
    log_path (str) : Target file path for logs (use absolute path recommended)
    """
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)


logger = build_logger()

local_rank = 0
world_size = 1
rank = 0

def local_rank0_log(info, level=logging.INFO):
    """
    Print log only on local_rank==0 in distributed training environments

    Parameters:
    info (str) : Message to log (supports f-string formatting)
    level (int) :
    """
    global local_rank

    if local_rank == 0:
        if level == logging.DEBUG:
            logger.debug(info)
        elif level == logging.INFO:
            logger.info(info)
        elif level == logging.WARNING:
            logger.warning(info)
        elif level == logging.ERROR:
            logger.error(info)
        elif level == logging.CRITICAL:
            logger.critical(info)
        else:
            logger.info(info)


def get_logo(concat=False, add_desc=True):
    """
    """
    logo_strs = []
    logo_strs.append(r"■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ ")
    logo_strs.append(r"  ╭───────────────────────────────────────────────────────────────╮   ")
    logo_strs.append(r"  │ ➤ MIMIXLM: PYTHON SINGLE-FILE LLM IMPLEMENTATION FROM SCRATCH │   ")
    logo_strs.append(r"  ╞═══════════════════════════════════════════════════════════════╡   ")
    logo_strs.append(r"  │               __  __  ___  __  __  ___  __    __              │   ")
    logo_strs.append(r"  │              |  \/  ||_ _||  \/  ||_ _| \ \  / /              │   ")
    logo_strs.append(r"  │              | \  / | | | | \  / | | |   \ \/ /               │   ")
    logo_strs.append(r"  │              | |\/| | | | | |\/| | | |    >  <                │   ")
    logo_strs.append(r"  │              | |  | | | | | |  | | | |   / /\ \               │   ")
    logo_strs.append(r"  │              |_|  |_||___||_|  |_||___| /_/  \_|              │   ")
    logo_strs.append(r"  ╞═══════════════════════════════════════════════════════════════╡   ")
    logo_strs.append(r"  ╰───────────────────────────────────────────────────────────────╯   ")
    logo_strs.append(r"■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ ")

    desc_str = "Mimixlm is a single-file Python implementation of large language models (LLM)."

    if add_desc:
        logo_strs.append(desc_str)

    if concat:
        logo_strs = "\n".join(logo_strs)

    return logo_strs


def show_logo(use_logger=True):
    """
    Displays application logo in terminal, with optional logging integration.

    Supports both direct printing and logger output for compatibility with distributed
    environments.
    Ensures logo only appears once in multi-process setups by using rank-aware logging.

    Args:
        use_logger (bool): When True, uses rank-aware logging (local_rank0_log).
                          When False, uses standard print(). Defaults to True.
    """
    for logo_str in get_logo():
        if use_logger:
            local_rank0_log(logo_str)
        else:
            print(logo_str)


def print_formated_args(args):
    """
    Prints configuration arguments in human-readable JSON format.

    Formats arguments for debugging/configuration tracking purposes. Ensures
    consistent output in distributed environments through rank-aware logging.

    Args:
        args (Namespace): Configuration arguments object (typically from argparse)
    """
    formatted_args = json.dumps(vars(args), ensure_ascii=False, indent=4)
    local_rank0_log(f"List all args: {formatted_args}")


class Registry:
    _registry = {}

    @classmethod
    def register(cls, path, obj=None):
        if obj is None:
            def decorator(o):
                cls._register(path, o)
                return o
            return decorator
        else:
            cls._register(path, obj)

    @classmethod
    def _register(cls, path, obj):
        parts = cls._split_path(path)
        current = cls._registry
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                raise ValueError(f"Path conflict at '{part}' in path '{path}'")
            current = current[part]

        last_part = parts[-1]
        if isinstance(current.get(last_part), dict):
            raise ValueError(f"Cannot override namespace at '{path}'")
        current[last_part] = obj

    @classmethod
    def get(cls, path, default=None):
        parts = cls._split_path(path)
        current = cls._registry
        
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    @classmethod
    def get_required(cls, path):
        result = cls.get(path)
        if result is None:
            raise KeyError(f"No object registered at path '{path}'")
        return result

    @staticmethod
    def _split_path(path):
        if not isinstance(path, str):
            raise TypeError("Path must be a string")
            
        parts = path.split('.')
        if not parts or any(not p for p in parts):
            raise ValueError(f"Invalid path format: '{path}'")
        return parts

    @classmethod
    def clear(cls):
        cls._registry = {}


"""
Define tokenizer
"""
class Node():
    """ 
    Represents a symbol (base or merged) in the BPE tokenization structure.
    """
    __slots__ = 'word', 'left', 'right'
    def __init__(self, word, left=None, right=None):
        """
        Initialize a symbol node in BPE's merge hierarchy.
        Args:
            word (str): The symbol string (e.g., a character or merged subword).
            left (Node): Left child node in the merge hierarchy (None for base symbols).
            right (Node): Right child node in the merge hierarchy (None for base symbols).
        """
        self.word = word
        self.left = left
        self.right = right


class PairNode():
    """
    Tracks a symbol pair's statistics during BPE training.

    Used to prioritize merges in a priority queue based on frequency.
    """
    __slots__ = 'word_pair', 'cnt', 'pos'
    def __init__(self, word_pair, cnt, pos):
        """
        Initialize a symbol pair tracking node.
        Args:
            word_pair (tuple): Symbol pair to track (e.g., ('e', 's'))
            cnt (int): Current frequency count of this pair in the corpus
            pos (int): Positional index in the priority heap (for efficient updates)
        """
        self.word_pair = word_pair
        self.cnt = cnt
        self.pos = pos

    @property
    def key(self):
        """
        For comparison
        """
        #return (self.cnt, self.word_pair)
        return self.cnt


def sift_up(heap, index, delta_value):
    """Bubble up a value at given index in max-heap"""
    heap[index].cnt += delta_value
    while index > 0:
        parent = (index - 1) // 2
        if heap[index].key <= heap[parent].key:
            break
        # Move parent down
        heap[index], heap[parent] = heap[parent], heap[index]
        heap[index].pos,heap[parent].pos = heap[parent].pos, heap[index].pos
        index = parent


def sift_down(heap, index, delta_value):
    """Sink down a value at given index in max-heap"""
    heap[index].cnt += delta_value
    n = len(heap)
    while index < n:
        left = 2 * index + 1
        right = 2 * index + 2
        largest = index

        if left < n and heap[left].key > heap[largest].key:
            largest = left
        if right < n and heap[right].key > heap[largest].key:
            largest = right

        if largest == index or heap[index].key >= heap[largest].key:
            break

        # Move child up
        heap[index], heap[largest] = heap[largest], heap[index]
        heap[index].pos, heap[largest].pos = heap[largest].pos, heap[index].pos

        index = largest


def update_heap(heap, heap_index, update_word_pair):
    """
    Update the heap structure based on the given word pairs and their deltas.

    This function processes each word pair in the update_word_pair dictionary. If a pair is not
    present in the heap, a new PairNode is created, added to the heap, and the heap is adjusted.
    If the pair already exists, the heap is adjusted based on whether the delta is positive
    (sift up) or negative (sift down). Finally, the update_word_pair dictionary is cleared.

    Args:
        heap (list): The heap structure, represented as a list of PairNode objects.
        heap_index (dict): A dictionary mapping word pairs to their corresponding PairNode objects.
                           Each PairNode stores its current position in the heap via the `pos` attribute.
        update_word_pair (dict): A dictionary of word pairs to update, where keys are word pairs
                                 and values are delta values (positive/negative) to adjust their
                                 priority in the heap.
    """
    for word_pair, delta_value in update_word_pair.items():
        if word_pair not in heap_index:
            pair_node = PairNode(word_pair, delta_value, len(heap))
            heap_index[word_pair] = pair_node
            heap.append(pair_node)
            sift_up(heap, len(heap)-1, 0)
        elif delta_value > 0:
            sift_up(heap, heap_index[word_pair].pos, delta_value)
        elif delta_value < 0:
            sift_down(heap, heap_index[word_pair].pos, delta_value)

    update_word_pair.clear()


def train_bpe(
        data_path_list,
        vocab_size,
        pat_str=None,
        text_fields=["text"],
        max_train_bpe_lines=50000,
        min_bpe_pairs_occurrence=5):
    """
    Train a Byte Pair Encoding (BPE) tokenizer on specified text data.

    Processes text files to learn BPE merge operations, producing a token-to-rank mapping
    where lower ranks indicate higher priority merge operations.

    Args:
        data_path_list (list[str]): List of file paths containing training text data.
                                    Supports JSON, CSV or raw text formats.
        vocab_size (int): Target size of the final BPE vocabulary.
        pat_str (str, optional): Regex pattern for text preprocessing. If None,
                                uses default whitespace-based tokenization.
                                Example: r"'s| 't| 're| ..." for GPT-style splitting.
        text_fields (list[str]): Field names to extract text from when processing
                                structured data (JSON/CSV). Defaults to ["text"].
        max_train_bpe_lines (int): Maximum number of text lines to sample for training.
                                  Helps balance speed vs. coverage (default: 50,000).
        min_bpe_pairs_occurrence (int): Minimum occurrence threshold for BPE token pairs;
                                        only pairs with frequency ≥ this value are eligible for merging
    Returns:
        dict[bytes, int]: Mergeable token ranks mapping, where:
            - keys: Byte sequences representing tokens (e.g. b"ing")
            - values: Merge priority rank (lower = merged earlier)

    """
    start = time.time()
    #init vocab
    mergeable_ranks =  {bytes([i]):i for i in range(256)}
    id2word = {mergeable_ranks[i]:i for i in mergeable_ranks}

    #init pattern
    pat = None
    if pat_str:
        pat = regex.compile(pat_str)

    #word_pair_cnt : Tracks global (word1, word2) frequencies
    word_pair_cnt = defaultdict(int)

    #word_pair_index : Stores (word1, word2) -> list of (left_node, right_node) pairs
    word_pair_index = defaultdict(list)

    done = 0
    for data in read_data_shards(data_path_list, max_sample_size=max_train_bpe_lines):

        text = get_data_from_dict(data, text_fields, mode="join", connector='\n')

        if len(text) == 0:
            continue

        words = [text]
        if pat:
            words = pat.findall(text)

        for word in words:
            word = word.encode("utf-8")

            # Build initial linked list of individual characters
            prev = Node(int(word[0]))
            for i in range(1, len(word)):

                # Create node storing word id and Link previous node
                prev.right = node = Node(int(word[i]), left=prev)

                word_pair = (prev.word, node.word)

                # Update global frequency counter
                word_pair_cnt[word_pair] += 1

                # Register node references for merge operations
                word_pair_index[word_pair].append((prev, node))

                # Move pointer forward
                prev,node = node,prev

        done += 1
        if done % 1000 == 0:
            local_rank0_log(f"done {done} lines")

    heap = []
    heap_index = {}
    update_word_pair = defaultdict(int)
    for word_pair,cnt in word_pair_cnt.items():
        pair_node = PairNode(word_pair, cnt, len(heap))
        update_word_pair[word_pair] = cnt
    update_heap(heap, heap_index, update_word_pair)

    steps = 0
    total_steps = vocab_size - len(mergeable_ranks)
    cost = time.time() - start
    local_rank0_log(f"Initialization cost {cost:.2f}s, Start merge now.")

    while len(mergeable_ranks) < vocab_size:
        pair_node = heap[0]

        if pair_node.cnt < min_bpe_pairs_occurrence:
            break

        left_word, right_word = pair_node.word_pair
        merge_word = id2word[left_word] + id2word[right_word]
        merge_id = len(mergeable_ranks)
        mergeable_ranks[merge_word] = merge_id
        id2word[merge_id] = merge_word

        steps += 1
        local_rank0_log(f"merge {steps}/{total_steps}")
        local_rank0_log(f"{left_word}, {right_word} -> {merge_word} had {pair_node.cnt} occurrences")
        local_rank0_log(f"add word {merge_word}, id {len(mergeable_ranks)}")

        #do merge
        update_word_pair[(left_word, right_word)] -= pair_node.cnt
        for left,right in word_pair_index[pair_node.word_pair]:
            if left.word != left_word or left.right != right or right.word != right_word:
                continue

            if left.left != None:
                update_word_pair[(left.left.word, left.word)] -= 1

            if right.right != None:
                update_word_pair[(right.word, right.right.word)] -= 1

            left.word = merge_id
            left.right = right.right
            right.right = None

            if left.left != None:
                update_word_pair[(left.left.word, left.word)] += 1
                word_pair_index[(left.left.word, left.word)].append((left.left, left))

            if left.right != None:
                left.right.left = left
                update_word_pair[(left.word, left.right.word)] += 1
                word_pair_index[(left.word, left.right.word)].append((left, left.right))

        #do stat update
        update_heap(heap, heap_index, update_word_pair)

    cost = time.time() - start
    local_rank0_log(f"Train tokenizer done, cost {cost:.2f}s.")

    return mergeable_ranks


def train_bpe_from_config(args):
    """
    Train BPE tokenizer using configuration parameters and save the resulting tokenizer.

    Loads training settings from model_config.json in the model directory, computes effective vocabulary
    size by reserving special tokens, then trains and persists the BPE tokenizer. Requires raw text
    corpus paths in arguments.

    Args:
        args (Namespace): Configuration object containing:
            - model_path (str): Directory containing model_config.json and for saving tokenizer.model
            - raw_data_path (list[str]): Path(s) to training text data (JSON/CSV/raw)
            - text_fields (list[str]): For structured data, fields containing text
            - max_train_bpe_lines (int): Maximum training samples to process
            - min_bpe_pairs_occurrence (int): Minimum occurrence threshold for BPE token pairs;
                                              only pairs with frequency ≥ this value are eligible for merging
    Returns:
        None
    """
    config_path = os.path.join(args.model_path, "model_config.json")
    with open(config_path, "rb") as f:
        config = json.load(f)
    vocab_size = config["model_config"]["vocab_size"] - config["tokenizer_config"]["num_reserved_special_tokens"]

    if vocab_size < 256:
        local_rank0_log("BPE Vocab Size must larger than 256", level=logging.error)
        return

    pat_str = config["tokenizer_config"]["pat_str"]

    mergeable_ranks = train_bpe(
            args.raw_data_path,
            vocab_size,
            pat_str,
            args.text_fields,
            args.max_train_bpe_lines,
            args.min_bpe_pairs_occurrence
            )
    local_rank0_log(f"BPE Vocab Size: {len(mergeable_ranks)}")

    with open(os.path.join(args.model_path, "tokenizer.model"), "wb") as f:
        for word in mergeable_ranks:
            encoded,idx = base64.b64encode(word), mergeable_ranks[word]
            f.write(encoded + f" {idx}\n".encode("utf-8"))


class BPE():
    """
    Byte Pair Encoding (BPE) tokenizer with configurable processing pattern and vocab
    """
    def __init__(self, pat_str=None, mergeable_ranks={}):
        """
        Initialize BPE tokenizer with processing rules

        Args:
            pat_str (str) : Regex pattern for text splitting
            mergeable_ranks (dict) : Pre-trained byte pair mappings
        """
        self.mergeable_ranks = mergeable_ranks
        self.id2token = {token_id: token_bytes for token_bytes, token_id in mergeable_ranks.items()}
        self.pat = None
        if pat_str:
            self.pat = regex.compile(pat_str)


    def bpe_encode(self, s, return_id=True):
        """
        Perform Byte Pair Encoding (BPE) on the input string using a priority queue approach.
        Args:
            s (str): Input string to be encoded using BPE

        Returns:
            list[int]: Token IDs from mergeable_ranks corresponding to the final merged symbols
        """
        if len(s) == 0:
            return []

        # Priority queue to track mergible pairs
        heap = []

        # Set head for first character
        head = prev = Node(s[:1])
        # 1. Build initial linked list of individual characters
        for i in range(1, len(s)):

            # Create node storing character(s) and Link previous node
            prev.right = node = Node(s[i:i+1], left=prev)

            combined = prev.word + node.word
            score = self.mergeable_ranks.get(combined, None)
            if score:
                # Push (score, position, right_word, left_word, left_node) to heap
                # Using min-heap, so lower score (higher priority) pops first
                heapq.heappush(heap, (score, i, prev.word, node.word, prev))

            # Move pointer forward
            prev,node = node,prev

        # 2. Greedily merge highest priority pairs until no merges remain
        while len(heap) > 0:
            pos,left_word,right_word,left = heapq.heappop(heap)[-4:]

            # Skip invalid pairs where nodes were already merged (size mismatch)
            # or right node no longer exists in the linked list
            if left_word != left.word or not left.right or right_word != left.right.word:
                continue

            # 3. Perform the merge: combine left and right nodes
            right = left.right
            # Update left node to contain merged symbol
            left.word = left.word + right.word

            # Update linked list pointers: bypass the right node
            left.right = right.right
            if right.right:
                left.right.left = left
                right.right = None

            # 4. Add new potential pairs formed with neighbors
            # Check left neighbor (left.left <-> merged node)
            if left.left:
                score = self.mergeable_ranks.get(left.left.word + left.word, None)
                if score:
                    heapq.heappush(heap, (score, pos, left.left.word, left.word, left.left))

            # Check right neighbor (merged node <-> left.right)
            if left.right:
                score = self.mergeable_ranks.get(left.word + left.right.word, None)
                if score:
                    heapq.heappush(heap, (score, pos-1, left.word, left.right.word, left))

        # 5. Convert merged symbols in linked list to token IDs
        tokens = []
        # Traverse linked list
        while head:
            if return_id:
                tokens.append(self.mergeable_ranks[head.word])
            else:
                tokens.append(head.word)
            head = head.right

        return tokens


    def encode(self, s):
        """
        Tokenize and encode an input string into subword tokens using BPE.

        Args:
            s (str): Input string to be encoded

        Returns:
            list: Sequence of token IDs representing the encoded string
        """
        if len(self.mergeable_ranks) == 0:
            raise TypeError("BPE's mergeable_ranks can't be empty!")
        words = self.pat.findall(s)
        token_ids = []
        for word in words:
            word = word.encode("utf-8")
            if word in self.mergeable_ranks:
                ids = [self.mergeable_ranks[word]]
            else:
                ids = self.bpe_encode(word)
            token_ids.extend(ids)
        return token_ids


    def decode(self, ids):
        """
        Convert token IDs back to human-readable string.

        Args:
            ids (list): Sequence of token IDs to decode

        Returns:
            str: Reconstructed string with proper UTF-8 decoding
        """
        s = b"".join(map(lambda x:self.id2token[x], ids))
        return s.decode("utf-8", errors="replace")


@Registry.register("tokenizer.simple")
class SimpleTokenizer():
    """
    A basic word-level tokenizer that handles a predefined vocabulary and special tokens.
    Maps words to IDs and vice versa, with support for reserved special tokens.
    """
    def __init__(self, **kwargs):
        """
        Initialize the tokenizer with vocabulary and special token configurations.
        
        Args:
            vocab_path (str): Path to vocabulary file containing base64-encoded tokens
            special_tokens (dict): Dictionary of special token tuples (token_str, token_id)
            num_reserved_special_tokens (int): Number of reserved IDs for special tokens
        """
        vocab_path = kwargs["vocab_path"]
        with open(vocab_path, "rb") as f:
            contents = f.read()
            self.vocab = {
                base64.b64decode(token).decode("utf-8"): int(wid)
                for token, wid in (
                    line.split() for line in contents.splitlines() if not line.startswith(b"#"))
            }
        self.id2word = {self.vocab[word]:word for word in self.vocab}

        self.num_reserved_special_tokens = kwargs["num_reserved_special_tokens"]
        self.n_words = len(self.vocab) + self.num_reserved_special_tokens

        # Initialize special token mappings
        special_tokens = kwargs["special_tokens"]
        self.bos_token, self.bos_id = special_tokens["bos"]
        self.eos_token, self.eos_id = special_tokens["eos"]
        self.eot_token, self.eot_id = special_tokens["eot"]
        self.start_header_token, self.start_header_id = special_tokens["start_header"]
        self.end_header_token, self.end_header_id = special_tokens["end_header"]
        self.pad_token, self.pad_id = special_tokens["pad"]
        self.unk_token, self.unk_id = special_tokens["unk"]

        # Build special token ID mappings
        self.id2special_tokens = {
                self.bos_id: self.bos_token,
                self.eos_id: self.eos_token,
                self.eot_id: self.eot_token,
                self.start_header_id: self.start_header_token,
                self.end_header_id: self.end_header_token,
                self.pad_id: self.pad_token,
                self.unk_id: self.unk_token
                }

        self.id2word.update(self.id2special_tokens)


    def encode(self, s, bos=False, eos=False):
        """
        Convert input string to sequence of token IDs.
        
        Args:
            s (str): Input text to tokenize
            bos (bool): Whether to prepend Beginning-Of-Sequence token
            eos (bool): Whether to append End-Of-Sequence token
        
        Returns:
            list: Token IDs with optional special tokens. Preserves existing BOS/EOS
                  if already present at string boundaries.
        """
        token_ids = [self.vocab.get(w, self.unk_id) for w in re.split("[ ]+", s)]

        if token_ids and token_ids[0] == self.bos_id and token_ids[-1] == self.eos_id:
            return token_ids

        if bos:
            token_ids.insert(0, self.bos_id)
        if eos:
            token_ids.append(self.eos_id)
        return token_ids


    def decode(self, ids, keep_special=False):
        """
        Convert token IDs back to human-readable string.
        
        Args:
            ids (list): Sequence of token IDs to decode
            keep_special (bool): Whether to preserve special tokens in output
        
        Returns:
            str: Decoded text with special tokens filtered or retained. Unknown IDs
                 become <unk> placeholders.
        """
        if not keep_special:
            ids = list(filter(lambda x: x not in self.id2special_tokens, ids))
        return " ".join([self.id2word.get(i, "<unk>") for i in ids])


    def encode_header(self, message):
        """
        Constructs structured token sequence for message metadata

        Args:
            message (dict): Must contain 'role' key with string value indicating
                message origin (e.g., 'user', 'system').

        Returns:
            list[int]: Token sequence structured as:
                [start_header, role_tokens, end_header, newline_tokens]

        """
        token_ids = []
        token_ids.append(self.start_header_id)
        token_ids.extend(self.encode(message["role"], bos=False, eos=False))
        token_ids.append(self.end_header_id)
        token_ids.extend(self.encode("\n\n", bos=False, eos=False))
        return token_ids


    def encode_message(self, message):
        """
        Builds complete message token sequence with content and termination

        Args:
            message (dict): Requires both 'role' and 'content' keys.

        Returns:
            list[int]: Structured tokens in format:
                [header_tokens, content_tokens, eot_token]
        """
        token_ids = self.encode_header(message)
        token_ids.extend(
            self.encode(message["content"], bos=False, eos=False)
        )
        token_ids.append(self.eot_id)
        return token_ids


    def encode_dialog_prompt(self, messages, bos=True):
        """
        Convert a conversation history into token sequence for prompt formatting.

        Args:
            messages (list[dict]): Conversation history containing 'role' and 'content' entries
            bos (bool): Whether to prepend Beginning-of-Sequence token. It should be added if it's fisrt turn.

        Returns:
            list: Token IDs representing the formatted conversation prompt
        """
        token_ids = []
        if bos:
            token_ids.append(self.bos_id)
        for message in messages:
            token_ids.extend(self.encode_message(message))
        token_ids.extend(self.encode_header({"role": "assistant", "content": ""}))
        return token_ids


    def encode_dialog_and_targets(self, messages):
        """
        Prepare training data format with input tokens and prediction targets.

        Args:
            messages (list[dict]): Conversation history with alternating roles

        Returns:
            tuple:
                - x (list): Input token sequence
                - targets (list): Target token sequence with padding
        """
        x = []
        targets = []

        x.append(self.bos_id)
        targets.append(self.pad_id)

        for message in messages:
            tokens = self.encode_header(message)
            x = x + tokens
            targets = targets + [self.pad_id] * len(tokens)

            tokens = self.encode(message["content"].strip(), bos=False, eos=False)
            x = x + tokens
            if message["role"] == "assistant":
                targets = targets + tokens
            else:
                targets = targets + [self.pad_id] * len(tokens)

            x.append(self.eot_id)
            if message["role"] == "assistant":
                targets.append(self.eot_id)
            else:
                targets.append(self.pad_id)

        x = x[:-1]
        targets = targets[1:]
        return x, targets


@Registry.register("tokenizer.bpe")
class BPETokenizer(SimpleTokenizer):
    """
    Tokenizer implementation supporting both tiktoken and custom BPE backends.
    """
    def __init__(self, **kwargs):
        """
        Initialize tokenizer with vocabulary and configuration.

        Args:
            **kwargs: Configuration parameters containing:
                - vocab_path (str): Path to vocabulary file
                - special_tokens (dict): Mapping of special token roles to (token, id) pairs
                - use_tiktoken (bool): Flag to enable tiktoken backend
                - num_reserved_special_tokens (int): Number of reserved special token slots
                - pat_str (str): Regex pattern for tokenization
        """
        vocab_path = kwargs["vocab_path"]
        special_tokens = kwargs["special_tokens"]
        use_tiktoken = kwargs.get("use_tiktoken", False)

        if use_tiktoken and not IMPORT_TIKTOKEN_SUCCESS:
            local_rank0_log("IMPORT TIKTOKEN FAILED, USE PURE PYTHON VERSION.",
                    level=logging.WARNING)
            use_tiktoken = False
        self.use_tiktoken = use_tiktoken

        self.num_reserved_special_tokens = kwargs["num_reserved_special_tokens"]
        self.pat_str = kwargs["pat_str"]

        # Load vocabulary according to backend choice
        with open(vocab_path, "rb") as f:
            contents = f.read()
            mergeable_ranks = {
                base64.b64decode(token): int(rank)
                for token, rank in (
                    line.split() for line in contents.splitlines() if not line.startswith(b"#"))
            }
        self.num_base_tokens = len(mergeable_ranks)

        # Initialize special token mappings
        self.bos_token, self.bos_id = special_tokens["bos"]
        self.eos_token, self.eos_id = special_tokens["eos"]
        self.eot_token, self.eot_id = special_tokens["eot"]
        self.start_header_token, self.start_header_id = special_tokens["start_header"]
        self.end_header_token, self.end_header_id = special_tokens["end_header"]
        self.pad_token, self.pad_id = special_tokens["pad"]

        # Build special token ID mappings
        self.id2special_tokens = {
                self.bos_id: self.bos_token,
                self.eos_id: self.eos_token,
                self.eot_id: self.eot_token,
                self.start_header_id: self.start_header_token,
                self.end_header_id: self.end_header_token,
                self.pad_id: self.pad_token
                }

        # Handle reserved special tokens
        idx = 0
        self.special_tokens = {}
        for i in self.id2special_tokens:
            self.special_tokens[self.id2special_tokens[i]] = i
        for i in range(self.num_base_tokens, self.num_base_tokens+self.num_reserved_special_tokens):
            if i not in self.id2special_tokens:
                self.special_tokens[f"<|reserved_special_token_{idx}|>"] = i
                idx += 1
        self.special_tokens_set = set(self.special_tokens.keys())

        # Initialize encoding backend
        if use_tiktoken:
            self.model = tiktoken.Encoding(
                name=os.path.basename(vocab_path),
                pat_str=self.pat_str,
                mergeable_ranks=mergeable_ranks,
                special_tokens=self.special_tokens,
            )
        else:
            self.model = BPE(self.pat_str, mergeable_ranks)

        # Calculate total vocabulary size
        self.n_words = self.num_base_tokens + self.num_reserved_special_tokens

        local_rank0_log(f"Reloaded tiktoken model from {vocab_path}. Total words: {self.n_words}")


    def encode(self, s, bos=False, eos=False):
        """
        Converts a given string `s` into a list of tokens using an internal tokenization model.
        Args:
            s (str): Input string to tokenize.
            bos (bool, default=False): Prepend BOS token or not
            eos (bool, default=False): Append EOS token or not
        Returns:
            list[int]: Token IDs with format:
                - Base case: [tok1, tok2, ..., tokN]
                - With BOS: [bos_id, tok1, ..., tokN]
                - With EOS: [tok1, ..., tokN, eos_id]
                - Both flags: [bos_id, tok1, ..., tokN, eos_id]
        """
        assert type(s) is str

        # The tiktoken tokenizer can handle <=400k chars without
        # pyo3_runtime.PanicException.
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000

        # https://github.com/openai/tiktoken/issues/195
        # Here we iterate over subsequences and split if we exceed the limit
        # of max consecutive non-whitespace or whitespace characters.
        MAX_NO_WHITESPACES_CHARS = 25_000

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        token_ids = []
        for substr in substrs:
            if self.use_tiktoken:
                token_ids.extend(self.model.encode(substr, allowed_special=self.special_tokens_set))
            else:
                token_ids.extend(self.model.encode(substr))

        if token_ids and token_ids[0] == self.bos_id and token_ids[-1] == self.eos_id:
            return token_ids

        if bos:
            token_ids.insert(0, self.bos_id)
        if eos:
            token_ids.append(self.eos_id)
        return token_ids


    def decode(self, ids, keep_special=False):
        """
        Decodes token IDs back to string using the internal model

        Args:
            ids (list[int]): Sequence of token IDs to decode.

        Returns:
            str: Decoded human-readable string.
        """
        if not keep_special:
            ids = list(filter(lambda x: x not in self.id2special_tokens, ids))
        return self.model.decode(ids)


    def _split_whitespaces_or_nonwhitespaces(self, s, max_consecutive_slice_len):
        """
        Splits string into chunks.
        Args:
            s (str): Input string to split. Empty strings return zero chunks
            max_consecutive_slice_len (int): Maximum allowed consecutive characters with no space.

        Yields:
            str: Substrings where no substring exceeds max_consecutive_slice_len length with no space
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]


"""
Define All Layers
"""

#Define all activation functions.
def get_act_fn(activation):
    """
    """
    act2fn = {
            "relu": F.relu,
            "gelu":F.gelu,
            "gelu_new":F.gelu,
            "swish":F.silu,
            "gelutanh": lambda x:F.gelu(x, approximate="tanh"),
            "geluquick": lambda x:x * torch.sigmoid(1.702 * x)
            }
    return act2fn[activation]


#Define all layers.
class Linear(nn.Module):
    """
    Custom linear layer implementation with optional weight sharing.
    """
    def __init__(self, d_in, d_out, use_bias=True, use_shared_weight=False, lora_rank=None):
        """
        Initialize linear layer parameters.

        Args:
            d_in (int): Input dimension
            d_out (int): Output dimension
            use_bias (bool): Enable/disable bias term
            use_shared_weight (bool): Bypass weight creation for external sharing
            lora_rank (int): # Specifies the rank of low-rank matrices for LoRA adaptation
        """
        super(Linear, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.use_shared_weight = use_shared_weight
        if not self.use_shared_weight:
            self.W = nn.Parameter(torch.Tensor(d_out, d_in))
        self.b = None
        if use_bias:
            self.b = nn.Parameter(torch.zeros(d_out))
        self.lora_rank = lora_rank
        if self.lora_rank:
            self.lora_A = nn.Parameter(torch.Tensor(d_in, lora_rank))
            self.lora_B = nn.Parameter(torch.Tensor(d_out, lora_rank))


    def reset_parameters(self, initializer_range=0.2):
        """
        Initialize layer parameters.
        """
        if not self.use_shared_weight:
            nn.init.normal_(self.W, mean=0.0, std=initializer_range)
        if self.b is not None:
            self.b.data.fill_(0)


    def forward(self, x, W=None):
        """
        Perform linear transformation.

        Args:
            x (Tensor): Input tensor of shape (..., d_in)
            W (Tensor, optional): External weight matrix for sharing

        Returns:
            Tensor: Output tensor of shape (..., d_out)
        """
        if not self.use_shared_weight:
            W = self.W
        else:
            assert W is not None
        y = F.linear(x, W)
        if self.b is not None:
            y = y + self.b

        if self.lora_rank:
            y = y + F.linear(F.linear(x, self.lora_A.T), self.lora_B)

        return y


class FactorizedEmbedding(nn.Module):
    """
    Memory-efficient embedding layer using matrix factorization

    Implements E = W * P decomposition where:
    - W: (vocab_size, factorized_size) token embeddings
    - P: (factorized_size, embedding_size) projection matrix
    Total parameters reduced from V*D to V*K + K*D (K << D)
    """
    def __init__(self, vocab_size, embedding_size, factorized_size):
        """

        """
        super(FactorizedEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.factorized_size = factorized_size
        self.W = nn.Parameter(torch.Tensor(vocab_size, factorized_size))
        self.proj = Linear(factorized_size, embedding_size)


    def reset_parameters(self, initializer_range=0.02):
        """
        Initializes weights using scaled uniform distribution
        """
        nn.init.normal_(self.W, mean=0.0, std=initializer_range)
        self.proj.reset_parameters(initializer_range)


    def forward(self, x):
        """
        Embeds token indices through factorized projection

        Args:
            x (LongTensor): Input token indices of shape [batch_size, seq_len]

        Returns:
            FloatTensor: Embedded sequence of shape [batch_size, seq_len, D]
        """
        return self.proj(self.W[x])


    def get_embedding(self):
        """
        Retrieves full pre-projection embedding matrix

        Returns:
            FloatTensor: Combined embeddings [V, D] = W @ P^T
            Where V=vocab_size, D=embedding_size
        """
        return self.proj(self.W)


class Embedding(nn.Module):
    """
    Implements a trainable token embedding layer with custom initialization.
    """
    def __init__(self, vocab_size, embedding_size):
        """
        Initialize embedding layer parameters.
        """
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.W = nn.Parameter(torch.Tensor(vocab_size, embedding_size))


    def reset_parameters(self, initializer_range=0.02):
        """
        Initialize embedding weights using scaled uniform distribution.
        """
        nn.init.normal_(self.W, mean=0.0, std=initializer_range)


    def forward(self, x):
        """
        Perform embedding lookup for input token indices.

        Args:
            x (LongTensor): Input tensor containing token indices, shape (..., )

        Returns:
            Tensor: Embedded representations, shape (..., embedding_size)
        """
        return self.W[x]


    def get_embedding(self):
        """
        Retrieve the full embedding matrix.

        Returns:
            Parameter: Learnable embedding matrix of shape (vocab_size, embedding_size)
        """
        return self.W


class Dropout(nn.Module):
    """
    Stochastic regularization layer that zeros inputs with probability `p`

    Implements inverted dropout for stable training when p > 0:
    ▸ During training:   x' = (mask(x) * x) / (1 - p)
    ▸ During inference:  x' = x
    """
    def __init__(self, p=0):
        """
        Initialize probability `p`
        """
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability incorrect!")
        self.p = p

    def forward(self, x):
        """
        Applies dropout mask during training with magnitude preservation

        Args:
            x (Tensor): Input features of shape [N, C, ...]

        Returns:
            Tensor: Mask-scaled tensor during training, else original
        """
        if self.training and self.p > 0:
            mask = torch.rand_like(x, device = x.device) < self.p
            x = torch.masked_fill(x, mask, 0)
            scale = (1.0/(1-self.p))
            return x * scale
        return x


class PositionEmbedding(nn.Module):
    """
    Implements positional embeddings with configurable initialization strategies.
    """
    def __init__(self, max_len, embedding_size, init="sinusoidal", freeze=True, base=10000):
        """
        Initialize positional embedding layer.

        Args:
            max_len (int): Maximum sequence length to handle
            embedding_size (int): Dimensionality of positional embeddings
            init (str): Initialization method - "sinusoidal" or others (random)
            freeze (bool): Whether to freeze positional embeddings (non-trainable)
            base (float): Base value for sinusoidal wavelength calculation
        """
        super(PositionEmbedding, self).__init__()
        self.max_len = max_len
        self.embedding_size = embedding_size
        self.freeze = freeze
        self.init = init

        if init == "sinusoidal":
            weight = np.zeros([max_len, embedding_size])
            angle = np.arange(max_len).reshape([-1,1])/np.power(base, np.arange(0, embedding_size, 2).reshape(1,-1)/embedding_size)
            weight[:,0::2] = np.sin(angle)
            weight[:,1::2] = np.cos(angle)
            weight = torch.from_numpy(weight).float()
            if freeze:
                self.register_buffer('W', weight)
            else:
                self.W = nn.Parameter(weight)
        else:
            self.W = nn.Parameter(torch.zeros([max_len, embedding_size]).float())


    def reset_parameters(self, initializer_range=0.02):
        """
        Initialize trainable parameters using scaled uniform distribution.
        """
        if not self.freeze and self.init == "random":
            nn.init.normal_(self.W, mean=0.0, std=initializer_range)


    def forward(self, pos_ids):
        """
        Retrieve positional embeddings for given position indices.

        Args:
            pos_ids (LongTensor): Tensor containing position indices, shape (..., )

        Returns:
            Tensor: Positional embeddings corresponding to indices, shape (..., embedding_size)
        """
        return self.W[pos_ids]


    def get_embedding(self):
        """
        Retrieve the full embedding matrix.

        Returns:
            Parameter: Embedding matrix of shape (vocab_size, embedding_size)
        """
        return self.W


def apply_rope(x, pos_embedding):
    """
    Applies Rotary Position Embedding (RoPE) to input tensors

    Implements positional encoding via rotation matrix transformation:
    ▸ x' = x * cos(pos) + x_rotated * sin(pos)
    ▸ Preserves relative positional information in attention mechanisms

    Args:
        x (Tensor): Input features of shape [B, n, L, d_qk]
            B - batch size, n - attention heads
            L - sequence length, d_qk - query/key dimension
        pos_embedding (Tensor): Precomputed positional embeddings [B, L, 2d]
            Contains interleaved [sin, cos] pairs for all positions

    Returns:
        Tensor: Position-enhanced features [B, n, L, d_qk]
    """
    embedding_size = pos_embedding.size(-1)

    #B x L x d_qk -> B x 1 x L x d_qk
    cos_pos = pos_embedding[:, :, 1::2].repeat([1, 1, 2]).unsqueeze(2).transpose(1,2)
    sin_pos = pos_embedding[:, :, 0::2].repeat([1, 1, 2]).unsqueeze(2).transpose(1,2)

    #B x n x L x d_qk -> B x n x L x d_qk
    x2 = torch.cat([-x[..., embedding_size//2:], x[..., :embedding_size//2]], -1)

    x = x * cos_pos + x2 * sin_pos

    return x


class RoPE(PositionEmbedding):
    """
    Rotary Position Embedding (RoPE) layer with sinusoidal initialization

    Inherits from base positional embedding and implements:
    """
    def __init__(self, max_len, embedding_size, base=10000):
        """
        Args:
            max_len (int): Maximum sequence length supported (>0)
            embedding_size (int): Dimension of positional features (even number)
            base (float): Frequency scaling base (default=10000)
        """
        super(RoPE, self).__init__(max_len, embedding_size, init="sinusoidal", freeze=True, base=base)


    def forward(self, pos_ids, x=None):
        """
        Retrieves positional embeddings or applies rotation to inputs

        Args:
            pos_ids (LongTensor): Position indices [B, L]
            x (Tensor, optional): Features to rotate [B, n, L, d]

        Returns:
            Tensor: If x=None → [B, L, 2d] position embeddings
                    Else → [B, n, L, d] rotated features
        """
        assert (pos_ids < self.max_len).all()
        pos_embedding = self.W[pos_ids]
        if not x:
            return pos_embedding
        return apply_rope(x, pos_embedding)


class RelativePositionEmbedding(PositionEmbedding):
    """
    Implements relative positional embeddings for capturing position relationships.

    Inherits From:
        PositionEmbedding: Reuses core positional embedding logic with offset handling
    """
    def __init__(self, max_relative_len, embedding_size, init="sinusoidal", freeze=True, base=10000):
        """
        Initialize relative position embedding layer.

        Args:
            max_relative_len (int): Maximum absolute relative distance to handle
            embedding_size (int): Dimensionality of position embeddings
            init (str): Initialization method (see PositionEmbedding)
            freeze (bool): Freeze embedding weights or not
            base (float): Base value for sinusoidal wavelength calculation
        """
        super(RelativePositionEmbedding, self).__init__(2*max_relative_len+1, embedding_size, init=init, freeze=freeze, base=base)
        self.max_relative_len = max_relative_len

    def forward(self, relative_dis):
        """
        Retrieve embeddings for relative position offsets.

        Args:
            relative_dis (LongTensor): Relative position distances (can be negative)

        Returns:
            Tensor: Position embeddings for each relative distance
        """
        relative_dis = torch.clamp(relative_dis, -self.max_relative_len, self.max_relative_len)
        ids = relative_dis + self.max_relative_len
        return self.W[ids]


class Alibi(nn.Module):
    """
    Implements Attention with Linear Biases (ALiBi) for transformer models.
    """
    def __init__(self, n_heads):
        """
        Args:
            n_heads (int): Number of attention heads (>0)
        """
        self.n_heads = n_heads
        start = (2**(-2**-(math.log2(self.n_heads)-3)))
        ratio = start
        slopes = torch.tensor([start*ratio**i for i in range(self.n_heads)])
        self.register_buffer('alibi_slopes', slopes)


    def forward(self, relative_dis):
        """
        Generates position-aware attention bias tensor

        Args:
            relative_dis (Tensor): Relative token distances [B, L_q, L_k]

        Returns:
            Tensor: Additive biases [B, n_heads, L_q, L_k]
        """
        return torch.einsum("bqk,n->bnqk", relative_dis, self.alibi_slopes)


class GatedFeedForward(nn.Module):
    """
    Implements a gated feed-forward network with activation-based gating mechanism.
    """
    def __init__(self, d_model, d_ff, activation="relu", dropout=0, use_bias=True):
        """
        Initialize gated feed-forward components.

        Args:
            d_model (int): Input/output dimension
            d_ff (int): Hidden layer dimension
            activation (str): Activation function name (supports keys in get_act_fn)
            dropout (float): Dropout probability (0 = no dropout)
            use_bias (bool): Enable bias terms in linear projections
        """
        super(GatedFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation

        self.up_proj = Linear(self.d_model, self.d_ff, use_bias)
        self.gate_proj = Linear(self.d_model, self.d_ff, use_bias)
        self.down_proj = Linear(self.d_ff, self.d_model, use_bias)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)


    def reset_parameters(self, initializer_range=0.02):
        """
        Initialize all sublayer parameters using custom initialization.

        Initialization propagates through:
            - up_proj linear layer
            - gate_proj linear layer
            - down_proj linear layer
        """
        self.up_proj.reset_parameters(initializer_range)
        self.gate_proj.reset_parameters(initializer_range)
        self.down_proj.reset_parameters(initializer_range)


    def forward(self, x):
        """
        Process input through gated feed-forward network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.down_proj(get_act_fn(self.activation)(self.gate_proj(x)) * self.up_proj(x))

        if self.dropout:
            x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    """
    Transformer-style Position-wise Feed-Forward Network (FFN)

    Implements FFN(x) = down_proj(activation(up_proj(x)))
    """
    def __init__(self, d_model, d_ff, activation="relu", dropout=0, use_bias=True):
        """
        Args:
            d_model (int): Hidden dimension of transformer (>0)
            d_ff (int): Inner layer dimension (>d_model recommended)
            activation (str): Nonlinearity type (default="relu")
            dropout (float): Drop probability [0,1) (default=0)
            use_bias (bool): Add bias term in linear layers (default=True)

        """
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation

        self.up_proj = Linear(self.d_model, self.d_ff, use_bias)
        self.down_proj = Linear(self.d_ff, self.d_model, use_bias)

        self.dropout = None
        if dropout > 0:
            self.dropout = Dropout(dropout)


    def reset_parameters(self, initializer_range=0.02):
        """
        Initializes weights
        """
        self.up_proj.reset_parameters(initializer_range)
        self.down_proj.reset_parameters(initializer_range)


    def forward(self, x):
        """
        Process input through feed-forward network.

        Args:
            x (Tensor): Input features [..., d_model]

        Returns:
            Tensor: Processed features [..., d_model]
        """
        x = self.down_proj(get_act_fn(self.activation)(self.up_proj(x)))

        if self.dropout:
            x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    """
    Custom Layer Normalization with optional scale/bias parameters
    """
    def __init__(self, d_model, eps=1e-5, use_scale=True, use_bias=True):
        """
        Args:
            d_model (int): Feature dimension of input (>0)
            eps (float): Small value for numerical stability (default=1e-5)
            use_scale (bool): Enable learnable scaling (default=True)
            use_bias (bool): Enable learnable bias (default=True)
        """
        super(LayerNorm, self).__init__()

        self.eps = eps
        self.d_model = d_model
        self.use_scale = use_scale
        if use_scale:
            self.alpha = nn.Parameter(torch.ones(self.d_model))
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(self.d_model))


    def forward(self, x):
        """
        Normalizes input features with optional affine transformation

        Args:
            x (Tensor): Input features [..., d_model]

        Returns:
            Tensor: Normalized features [..., d_model]
        """
        x_dtype = x.dtype
        x = x.float()

        mean = x.mean(dim=-1, keepdim=True)

        #std = x.std(dim=-1, unbiased=False, keepdim=True)
        #norm = (x - mean) / (std + self.eps)
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.use_scale:
            norm = self.alpha * norm
        if self.use_bias:
            norm = norm + self.bias

        norm = norm.to(x_dtype)

        return norm


class RMSNorm(nn.Module):
    """
    Implements Root Mean Square Layer Normalization (RMSNorm) without bias term.
    """
    def __init__(self, d_model, eps=1e-5):
        """
        Initialize RMS normalization layer.

        Args:
            d_model (int): Dimension of input features
            eps (float): Small value to prevent division by zero
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """
        Apply RMS normalization to input tensor.

        Args:
            x (Tensor): Input tensor of shape (..., d_model)

        Returns:
            Tensor: Normalized output with same shape as input
        """
        x_dtype = x.dtype
        x = x.float()

        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        norm = self.alpha * (x * rms)

        norm = norm.to(x_dtype)

        return norm


def scaled_dot_product_attention(query,
                                 key,
                                 value,
                                 attn_mask=None,
                                 attn_scale=None,
                                 attn_dropout=None,
                                 pos_embedding_type="none",
                                 pos_embedding_query=None,
                                 pos_embedding_key=None,
                                 pos_embedding_value=None,
                                 pos_bias=None,
                                 attention_backend = "none"):
    """
    Computes scaled dot-product attention with multiple positional encoding support.

    Args:
        query (Tensor): Query tensor [B, n_heads, L_q, d_qk]
        key (Tensor): Key tensor [B, n_heads, L_kv, d_qk]
        value (Tensor): Value tensor [B, n_heads, L_kv, d_v]
        attn_mask (Tensor): Attention mask [B, L_q, L_kv]
        attn_scale (float): Scaling factor for attention scores
        attn_dropout (nn.Dropout): Dropout layer for attention weights
        pos_embedding_type (str): Type of positional encoding ("none"/"rope"/"relative"/"alibi")
        pos_embedding_query (Tensor): Positional embeddings for queries
        pos_embedding_key (Tensor): Positional embeddings for keys
        pos_embedding_value (Tensor): Positional embeddings for values
        pos_bias (Tensor): Precomputed positional bias for ALiBi
        attention_backend (str): Specifies the backend implementation for attention computation.
                                 "none": Uses a naive, unoptimized implementation.
                                 "torch_native": Leverages PyTorch's built-in attention.
    """
    batch_size, n_heads, len_q, d_qk = query.shape[0], query.shape[1], query.shape[2], query.shape[3]
    n_kv_heads, len_kv = key.shape[1], key.shape[2]
    if n_heads != n_kv_heads:
        n_rep = n_heads // n_kv_heads
        key = key[:, :, None, :, :].repeat(1, 1, n_rep, 1, 1).reshape(batch_size, n_heads, -1,  d_qk)
        value = value[:, :, None, :, :].repeat(1, 1, n_rep, 1, 1).reshape(batch_size, n_heads, -1,  d_qk)

    if pos_embedding_type == "rope":
        query = apply_rope(query, pos_embedding_query)
        if pos_embedding_key is not None:
            key = apply_rope(key, pos_embedding_key)

    attn_bias = None
    if attn_mask is not None or pos_embedding_type == "alibi" or pos_embedding_type.startswith("relative"):
        attn_bias = torch.zeros([batch_size, n_heads, len_q, len_kv], device=query.device)
        if attn_mask is not None:
            attn_bias.masked_fill_(attn_mask.logical_not(), -torch.inf)

        if pos_embedding_type == "alibi":
            attn_bias += pos_bias
        elif pos_embedding_type and pos_embedding_type.startswith("relative"):
            if pos_embedding_key.dim() == 3:
                #p_k:L_q x L_k x d_qk
                attn_bias += torch.einsum("bnqd,qkd->bnqk", query, pos_embedding_key)
            else:
                #p_k:B x L_q x L_k x d_qk
                attn_bias += torch.einsum("bnqd,bqkd->bnqk", query, pos_embedding_key)

        attn_bias = attn_bias.to(query.dtype)

    if attention_backend == "torch_native":
        output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_bias, scale=attn_scale)
        attn_weight = None
    else:
        #q:B x n_heads x L_q x d_qk
        #k:B x n_heads x L_kv x d_v
        #scores:B x n_heads x L_q x L_kv
        scores = torch.matmul(query, key.transpose(-1, -2))
        if attn_bias is not None:
            attn_bias = attn_bias.masked_fill(attn_bias.isinf(), -1e20)
            scores += attn_bias

        if attn_scale:
            scores = scores * attn_scale

        attn_weight = F.softmax(scores, dim = -1, dtype=torch.float32).to(query.dtype)

        if attn_dropout:
            attn_weight = attn_dropout(attn_weight)

        #scores:B x n_heads x L_q x L_kv
        #v:B x n_heads x L_kv x d_v
        output = torch.matmul(attn_weight, value)

    if pos_embedding_type and pos_embedding_type.startswith("relative") and pos_embedding_value:
        if pos_embedding_value.dim() == 3:
            #pe_v:B x L_q x L_kv x d_v
            output += torch.einsum("bnqk,qkd->bnqd", attn_weight, pos_embedding_value)
        else:
            #pe_v:L_q x L_kv x d_v
            output += torch.einsum("bnqk,bqkd->bnqd", attn_weight, pos_embedding_value)

    return output, attn_weight


class MultiHeadAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) Enhanced Multi-Head Attention Module

    Implements scaled dot-product attention with modern optimizations:
    ▸ Supports standard MHA and memory-efficient GQA (n_kv_heads ≤ n_heads)
    ▸ Optional position-aware attention mechanisms via extension hooks
    """
    def __init__(self,
                 n_heads,
                 d_q_input,
                 d_k_input,
                 d_v_input,
                 d_out,
                 d_qk,
                 d_v,
                 dropout=0,
                 attn_dropout=0,
                 use_bias=True,
                 attn_scale=None,
                 n_kv_heads=None,
                 attention_backend="none",
                 pos_embedding_type="none",
                 lora_rank=None):

        """
        Args:
            n_heads (int): Number of query heads (>0)
            d_q_input (int): Query input dimension
            d_k_input (int): Key input dimension
            d_v_input (int): Value input dimension
            d_out (int): Output dimension
            d_qk (int): Projected dimension per head for Q/K (common practice: 64)
            d_v (int): Projected dimension per head for V (common practice: 64)
            dropout (float): Post-output dropout [0,1) (default=0)
            attn_dropout (float): Attention matrix dropout [0,1) (default=0)
            use_bias (bool): Enable projection biases (default=True)
            attn_scale (float): Manual scaling factor (default=1/√d_qk)
            n_kv_heads (int): Number of key/value heads (≤n_heads, default=n_heads)
            attention_backend (str): Specifies the backend implementation for attention computation.
                                     "none": Uses a naive, unoptimized implementation.
                                     "torch_native": Leverages PyTorch's built-in attention.
            pos_embedding_type (str): Position encoding type ('alibi','rope', "relative", "none")
            lora_rank(int): Specifies the rank of low-rank matrices for LoRA adaptation
        """
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dropout = None
        if dropout > 0:
            self.dropout = Dropout(dropout)
        self.attn_dropout = None
        if attn_dropout > 0:
            self.attn_dropout = Dropout(dropout)
        self.d_qk = d_qk
        self.d_v = d_v
        self.n_kv_heads = n_heads
        if n_kv_heads:
            self.n_kv_heads = n_kv_heads
       	self.attention_backend = attention_backend

        self.lora_rank = lora_rank
        self.q_proj = Linear(d_q_input, n_heads*d_qk, use_bias=use_bias, lora_rank=self.lora_rank)
        self.k_proj = Linear(d_k_input, self.n_kv_heads*d_qk, use_bias=use_bias, lora_rank=self.lora_rank)
        self.v_proj = Linear(d_v_input, self.n_kv_heads*d_v, use_bias=use_bias, lora_rank=self.lora_rank)
        self.o_proj = Linear(n_heads*d_v, d_out, use_bias=use_bias, lora_rank=self.lora_rank)

        self.attn_scale = attn_scale

        self.pos_embedding_type = pos_embedding_type


    def reset_parameters(self, initializer_range=0.02):
        """
        Initializes projection layers' parameters using modern init schemes
        """
        self.q_proj.reset_parameters(initializer_range)
        self.k_proj.reset_parameters(initializer_range)
        self.v_proj.reset_parameters(initializer_range)
        self.o_proj.reset_parameters(initializer_range)


    def forward(self,
                query,
                key,
                value,
                attn_mask=None,
                is_cached_kv=False,
                pos_embedding_query=None,
                pos_embedding_key=None,
                pos_embedding_value=None,
                pos_bias=None):
        """
        Compute multi-head attention with configurable positional encoding support.

        Args:
            query (Tensor): [B, L_q, d_model] query sequence
            key (Tensor): [B, L_kv, d_model] key sequence
            value (Tensor): [B, L_kv, d_model] value sequence
            attn_mask (Tensor): [B, L_q, L_kv] attention mask (True=keep)
            is_cached_kv (bool): Whether key/value are cached from previous steps
            pos_embedding_query (Tensor): Positional embeddings for queries
            pos_embedding_key (Tensor): Positional embeddings for keys
            pos_embedding_value (Tensor): Positional embeddings for values
            pos_bias (Tensor): Precomputed positional bias tensor

        Returns:
            tuple:
                - output (Tensor): [B, L_q, d_model] attention output
                - attn_weight (Tensor): [B, n_heads, L_q, L_kv] attention weights
        """
        #B x L x d_model -> B x l x (d*n_heads)
        query = self.q_proj(query)
        if not is_cached_kv:
            key = self.k_proj(key)
            value = self.v_proj(value)

        batch_size = query.size(0)
        #B x l x (d*n_heads) -> B x n_heads x L x d_qk
        query = query.view(batch_size, -1, self.n_heads, self.d_qk).transpose(1, 2)
        if not is_cached_kv:
            key = key.view(batch_size, -1, self.n_kv_heads, self.d_qk).transpose(1, 2)
            value = value.view(batch_size, -1, self.n_kv_heads, self.d_v).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        if self.pos_embedding_type == "rope" and is_cached_kv:
            pos_embedding_key = None

        output, attn_weight = scaled_dot_product_attention(query,
                                                           key,
                                                           value,
                                                           attn_mask,
                                                           self.attn_scale,
                                                           self.attn_dropout,
                                                           self.pos_embedding_type,
                                                           pos_embedding_query,
                                                           pos_embedding_key,
                                                           pos_embedding_value,
                                                           pos_bias,
                                                           self.attention_backend)

        #B x n_heads x L x d -> B x L x n_heads x d -> B x L x d_model
        output = output.transpose(1,2)
        output = output.contiguous().view(batch_size, -1, self.n_heads*self.d_v)

        output = self.o_proj(output)

        if self.attn_dropout:
            output = self.dropout(output)

        return output, attn_weight


    def cache_kv(self, key, value, pos_embedding_key=None):
        """
        Precomputes and formats KV pairs for autoregressive inference

        Args:
            key (Tensor): Raw key states [B, L_kv, d_k_input]
            value (Tensor): Raw value states [B, L_kv, d_v_input]
            pos_embedding_key (Tensor): Position indices for RoPE [B, L_kv] (optional)

        Returns:
            list: Processed KV cache [key, value] with shapes:
                - key: [B, n_kv_heads, L_kv, d_qk]
                - value: [B, n_kv_heads, L_kv, d_v]

        """
        batch_size = key.size(0)
        key = self.k_proj(key)
        value = self.v_proj(value)

        key = key.view(batch_size, -1, self.n_kv_heads, self.d_qk).transpose(1, 2)
        value = value.view(batch_size, -1, self.n_kv_heads, self.d_v).transpose(1, 2)

        if self.pos_embedding_type == "rope":
            key = apply_rope(key, pos_embedding_key)

        return [key, value]


class TransformerLayer(nn.Module):
    """
    A Transformer encoder layer consisting of multi-head self-attention and feed-forward network.
    Supports various configuration options including pre/post normalization, layer scaling, and
    different position embedding types.
    """
    def __init__(self, **kwargs):
        """
        Initialize the Transformer layer with configurable parameters.

        Keyword Args:
            n_heads (int): Number of attention heads. Default: 12
            d_model (int): Model dimension. Default: 768
            d_ff (int): Hidden dimension of feed-forward network. Default: 4*d_model
            d_qk (int): Dimension of query/key vectors. Default: d_model//n_heads
            d_v (int): Dimension of value vectors. Default: d_model//n_heads
            dropout (float): General dropout rate. Default: 0
            attn_dropout (float): Attention-specific dropout rate. Default: 0
            attn_scale (float): Scaling factor for attention scores. Default: sqrt(d_qk)
            ln_eps (float): Epsilon for layer normalization. Default: 1e-5
            use_ln_scale (bool): Use learnable scale in normalization. Default: True
            use_ln_bias (bool): Use learnable bias in normalization. Default: True
            use_pre_norm (bool): Use pre-normalization (else post-normalization). Default: True
            activation (str): Activation function for FFN. Default: 'swish'
            norm_type (str): Normalization type ('layer_norm' or 'rms_norm'). Default: 'rms_norm'
            use_attention_bias (bool): Use bias in attention projections. Default: False
            use_ffn_bias (bool): Use bias in FFN layers. Default: False
            use_glu (bool): Use Gated Linear Unit in FFN. Default: True
            n_kv_heads (int): Number of key/value heads (for grouped-query attention). Default: n_heads
            pos_embedding_type (str): Position embedding type ('rope' for RoPE). Default: 'rope'
            layer_scale (str): Layer scaling type ('learned' or 'dynamic'). Default: None
            init_layer_scale (float): Initial value for layer scaling. Default: 1
        """
        super(TransformerLayer, self).__init__()

        self.n_heads = kwargs.get("n_heads", 12)
        self.d_model = kwargs.get("d_model", 768)
        self.d_ff = kwargs.get("d_ff", 4 * self.d_model)
        self.d_qk = kwargs.get("d_qk", self.d_model//self.n_heads)
        self.d_v = kwargs.get("d_v", self.d_model//self.n_heads)
        self.dropout = kwargs.get("dropout", 0)
        self.attn_dropout = kwargs.get("attn_dropout", 0)
        self.attn_scale = kwargs.get("attn_scale", 1/np.sqrt(self.d_qk))
        self.ln_eps = kwargs.get("ln_eps", 1e-5)
        self.use_ln_scale = kwargs.get("use_ln_scale", True)
        self.use_ln_bias = kwargs.get("use_ln_bias", True)
        self.use_pre_norm = kwargs.get("use_pre_norm", True)
        self.activation = kwargs.get("activation", "swish")
        self.norm_type = kwargs.get("layer_norm_type", "rms_norm")
        self.use_attention_bias = kwargs.get("use_attention_bias", False)
        self.use_ffn_bias = kwargs.get("use_ffn_bias", False)
        self.use_glu = kwargs.get("use_glu", True)
        self.n_kv_heads = kwargs.get("n_kv_heads", self.n_heads)
        self.attention_backend = kwargs.get("attention_backend", "none")
        self.pos_embedding_type = kwargs.get("attn_pos_embedding_type", "rope")
        self.layer_scale = kwargs.get("layer_scale", None)
        self.lora_rank = kwargs.get("lora_rank", None)

        assert self.d_model % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0

        if self.norm_type == "layer_norm":
            norm_cls = partial(LayerNorm,
                               d_model=self.d_model,
                               eps=self.ln_eps,
                               use_scale=self.use_ln_scale,
                               use_bias=self.use_ln_bias)
        elif self.norm_type == "rms_norm":
            norm_cls = partial(RMSNorm,
                               d_model=self.d_model,
                               eps=self.ln_eps)

        if self.use_pre_norm:
            self.first_norm = norm_cls()
        self.self_attention = MultiHeadAttention(self.n_heads,
                                                 self.d_model,
                                                 self.d_model,
                                                 self.d_model,
                                                 self.d_model,
                                                 self.d_qk,
                                                 self.d_v,
                                                 self.dropout,
                                                 self.attn_dropout,
                                                 self.use_attention_bias,
                                                 self.attn_scale,
                                                 self.n_kv_heads,
                                                 self.attention_backend,
                                                 self.pos_embedding_type,
                                                 self.lora_rank)
        if not self.use_pre_norm:
            self.first_norm = norm_cls()

        if self.use_pre_norm:
            self.last_norm = norm_cls()

        ffn_cls = FeedForward
        if self.use_glu:
            ffn_cls = GatedFeedForward

        self.ffn = ffn_cls(d_model=self.d_model,
                           d_ff=self.d_ff,
                           activation=self.activation,
                           dropout=self.dropout,
                           use_bias=self.use_ffn_bias)

        if not self.use_pre_norm:
            self.last_norm = norm_cls()

        if self.layer_scale == "learned":
            init_layer_scale = kwargs.get("init_layer_scale", 1)
            self.first_gamma = nn.Parameter(init_layer_scale * torch.ones((self.d_model)))
            self.last_gamma = nn.Parameter(init_layer_scale * torch.ones((self.d_model)))
        elif self.layer_scale == "dynamic":
            init_layer_scale = kwargs.get("init_layer_scale", 1)
            self.first_gamma = init_layer_scale
            self.last_gamma = init_layer_scale


    def reset_parameters(self, initializer_range=0.02):
        """
        Reset all trainable parameters of the module.
        """
        self.self_attention.reset_parameters(initializer_range)
        self.ffn.reset_parameters(initializer_range)


    def forward(self,
                x,
                self_attn_mask=None,
                cached_kv=None,
                pos_embedding_query=None,
                pos_embedding_key=None,
                pos_embedding_value=None,
                pos_bias=None):
        """
        Transformer layer forward pass with optional cached key-value pairs and positional embeddings.

            Args:
                x: Input tensor of shape [batch_size, seq_len, d_model]
                self_attn_mask: Optional attention mask of shape
                    [batch_size, L_q, L_kv]
                    seq_len == L_q == L_kv when training(no cache kv)
                cached_kv: Optional tuple containing cached keys and values:
                    - cached_key: [batch_size, num_heads, past_seq_len, d_kv]
                    - cached_value: [batch_size, num_heads, past_seq_len, d_kv]
                pos_embedding_query: Positional embeddings for query
                    (used in rotary/multi-query attention) of shape [batch_size, seq_len, d_qk]
                pos_embedding_key: Positional embeddings for key
                    of shape [batch_size, seq_len, d_qk] or [batch_size, 1, kv_len, d_qk]
                pos_embedding_value: Positional embeddings for value
                    of shape [batch_size, seq_len, d_v] or [batch_size, 1, kv_len, d_qk]
                pos_bias: Positional bias tensor of shape
                    [batch_size, num_heads, q_len, k_len]

            Returns:
                outputs: Dictionary containing:
                    - output: [batch_size, seq_len, d_model]
                    - self_attn_weight: [batch_size, num_heads, L_q, L_kv]
                    - cached_kv: Tuple with updated keys/values (when caching enabled)
        """
        output = x

        residual = output
        if self.use_pre_norm:
            output = self.first_norm(output)
        query, key, value = output, output, output
        if cached_kv:
            key, value = self.cache_kv(output, pos_embedding_key)
            cached_key, cached_value = cached_kv
            if cached_key is not None:
                key = torch.cat([cached_key, key], 2)
            if cached_value is not None:
                value = torch.cat([cached_value, value], 2)
        output, self_attn_weight = self.self_attention(query,
                                                       key,
                                                       value,
                                                       self_attn_mask,
                                                       cached_kv is not None,
                                                       pos_embedding_query,
                                                       pos_embedding_key,
                                                       pos_embedding_value,
                                                       pos_bias)
        if self.layer_scale:
            output = residual + self.first_gamma * output
        else:
            output = residual + output

        if not self.use_pre_norm:
            output = self.first_norm(output)

        residual = output

        if self.use_pre_norm:
            output = self.last_norm(output)

        output = self.ffn(output)

        if self.layer_scale:
            output = residual + self.last_gamma * output
        else:
            output = residual + output

        if not self.use_pre_norm:
            output = self.last_norm(output)

        outputs = {}
        outputs["output"] = output
        outputs["self_attn_weight"] = self_attn_weight

        if cached_kv:
            outputs["cached_kv"] = [key, value]

        return outputs


    def cache_kv(self, dec_output, pos_embedding_key):
        """
        """
        return self.self_attention.cache_kv(dec_output, dec_output, pos_embedding_key)


class Transformer(nn.Module):
    """
    Transformer neural network architecture with configurable positional encodings.
    """
    def __init__(self, **kwargs):
        """
        Initialize transformer architecture components.

        **kwargs: Configuration parameters containing:
            Args:
                share_layer_params_type (str, optional): Layer parameter sharing strategy.
                    None for independent layers. Other values enable cross-layer sharing.
                n_share_cross_layers (int): Number of layer groups when using parameter sharing
                n_layers (int): Total number of transformer layers
                attn_pos_embedding_type (str): Position encoding type
                    ("rope", "relative", "relative_with_value", "alibi")
                attn_pos_embedding_max_len (int): Maximum sequence length for position tables
                attn_pos_embedding_init (str): Position encoding initialization method
                freeze_attn_pos_embedding (bool): Freeze position encoding parameters
                attn_pos_embedding_base (int): Base value for positional calculations
        """
        super(Transformer, self).__init__()
        self.share_layer_parmas_type = kwargs.get("share_layer_parmas_type", None)
        self.n_share_cross_layers = 1
        if self.share_layer_parmas_type:
            self.n_share_cross_layers = kwargs.get("n_share_cross_layers", 1)
        self.n_layers = kwargs.get("n_layers", 12)

        self.layers = nn.ModuleList([TransformerLayer(**kwargs) for i in range(self.n_layers//self.n_share_cross_layers)])
        self.d_qk = self.layers[0].d_qk

        self.pos_embedding_type = kwargs.get("attn_pos_embedding_type", "rope")
        self.pos_embedding_max_len = kwargs.get("attn_pos_embedding_max_len", kwargs.get("max_len", 4096))
        self.pos_embedding_init = kwargs.get("attn_pos_embedding_init", "sinusoidal")
        self.freeze_pos_embedding = kwargs.get("freeze_attn_pos_embedding", True)
        self.pos_embedding_base = kwargs.get("attn_pos_embedding_base", 500000)
        if self.pos_embedding_type == "rope":
            self.rope_embedding = RoPE(self.pos_embedding_max_len, self.d_qk, self.pos_embedding_base)
        elif self.pos_embedding_type and self.pos_embedding_type.startswith("relative"):
            self.relative_key_embedding = RelativePositionEmbedding(self.pos_embedding_max_len,
                                                                    self.d_qk,
                                                                    self.pos_embedding_init,
                                                                    self.freeze_pos_embedding,
                                                                    self.pos_embedding_base)
            self.relative_value_pe = None
            if self.pos_embedding_type == "relative_with_value":
                self.relative_value_embedding = RelativePositionEmbedding(self.pos_embedding_max_len,
                                                                          self.d_qk,
                                                                          self.pos_embedding_init,
                                                                          self.freeze_pos_embedding,
                                                                          self.pos_embedding_base)
        elif self.pos_embedding_type == "alibi":
            self.alibi = Alibi(self.n_heads)

        self.layers = nn.ModuleList([TransformerLayer(**kwargs) for i in range(self.n_layers//self.n_share_cross_layers)])


    def reset_parameters(self, initializer_range=0.02):
        """
        Initialize all learnable parameters in the transformer.
        """
        if self.pos_embedding_type == "rope":
            self.rope_embedding.reset_parameters(initializer_range)
        elif self.pos_embedding_type and self.pos_embedding_type.startswith("relative"):
            self.relative_key_embedding.reset_parameters(initializer_range)
            self.relative_value_pe = None
            if self.pos_embedding_type == "relative_with_value":
                self.relative_value_embedding.reset_parameters(initializer_range)
        for layer in self.layers:
            layer.reset_parameters(initializer_range)


    def forward(self,
                x,
                self_attn_mask=None,
                pos_ids=None,
                cache=None,
                use_checkpoint=False):
        """
        Process input through transformer layers with configurable position awareness.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, d_model]
            self_attn_mask (Tensor, optional): Attention mask of shape
                [batch_size, target_len, source_len]
            pos_ids (Tensor): Position indices of shape [batch_size, seq_len]
            cache (Dict, optional): Cache dictionary containing:
                - "cached_kv": List of previous key-value states per layer
                - "pos_ids": Accumulated position indices
            use_checkpoint (bool): Enable gradient checkpointing for memory optimization

        Returns:
            Dict: Contains:
                - "output": Final layer output [batch_size, seq_len, d_model]
                - "cache": Updated cache state (if provided)
                - "hidden_states": List of all layer outputs
                - "self_attn_weights": Attention weights from all layers
        """
        #process various position embeddings
        pos_embedding_query = None
        pos_embedding_key = None
        pos_embedding_value = None
        pos_bias = None

        q_pos_ids = pos_ids
        kv_pos_ids = pos_ids


        if self.pos_embedding_type == "rope":
            pos_embedding_query = self.rope_embedding(q_pos_ids)
            pos_embedding_key = self.rope_embedding(kv_pos_ids)
        elif self.pos_embedding_type and self.pos_embedding_type.startswith("relative"):
            if cache and cache["pos_ids"]:
                kv_pos_ids = torch.cat([cache["pos_ids"], q_pos_ids], 1)

            if cache:
                relative_dis = q_pos_ids[:,:,None] - kv_pos_ids[:,None,:]
            else:
                x_len = x.shape[1]
                q_pos_ids = torch.arange(x_len, dtype=torch.long, device=x.device).view(-1, 1)
                kv_pos_ids = torch.arange(x_len, dtype=torch.long, device=x.device).view(1, -1)
                relative_dis = q_pos_ids - kv_pos_ids

            pos_embedding_key = self.relative_key_embedding(relative_dis)
            pos_embedding_value = None
            if self.pos_embedding_type == "relative_with_shared_key_value":
                pos_embedding_value = pos_embedding_key
            elif self.pos_embedding_type == "relative_with_value":
                pos_embedding_value = self.relative_value_embedding(relative_dis)
        elif self.pos_embedding_type == "alibi":
            if cache and cache["pos_ids"]:
                kv_pos_ids = torch.cat([cache["pos_ids"], q_pos_ids], 1)
            relative_dis = torch.abs(q_pos_ids[:,:,None] - kv_pos_ids[:,None,:])
            pos_bias = self.alibi(relative_dis)

        output = x
        self_attn_weights = []
        hidden_states = []
        for i in range(self.n_layers):

            if self.share_layer_parmas_type:
                if self.share_layer_parmas_type == "repeat":
                    layer = self.layers[i // self.n_share_cross_layers]
                elif self.share_layer_parmas_type == "cycle":
                    layer = self.layers[i % self.n_share_cross_layers]
            else:
                layer = self.layers[i]

            if use_checkpoint and self.training:
                output.requires_grad_(True)
                outputs = torch.utils.checkpoint.checkpoint(layer,
                                                            output,
                                                            self_attn_mask,
                                                            cache["cached_kv"][i] if cache else None,
                                                            pos_embedding_query,
                                                            pos_embedding_key,
                                                            pos_embedding_value,
                                                            pos_bias,
                                                            use_reentrant=False)

            else:
                outputs = layer(output,
                                self_attn_mask,
                                cache["cached_kv"][i] if cache else None,
                                pos_embedding_query,
                                pos_embedding_key,
                                pos_embedding_value,
                                pos_bias)

            output = outputs["output"]

            self_attn_weights.append(outputs["self_attn_weight"])

            hidden_states.append(output)

            if cache:
                cache["cached_kv"][i] = outputs["cached_kv"]
        if cache and cache["pos_ids"] is not None:
            cache["pos_ids"] = torch.cat([cache["pos_ids"], pos_ids], 1)


        outputs = {
                      "output": output,
                      "cache": cache,
                      "hidden_states": hidden_states,
                      "self_attn_weights": self_attn_weights
                  }


        return outputs


    def init_cache(self):
        """
        Initialize a cache dictionary to store intermediate key-value pairs and related data during inference.

        Returns:
            dict: A dictionary containing initialized cache components.

            The cache structure includes:
                - cached_kv: A list where each element corresponds to a layer's cached key-value tensors
                            (initialized as [None, None]).
                - kv_mask: Placeholder for attention mask (initialized as None).
                - pos_ids: Placeholder for positional indices (initialized as None).
        """
        cache = {
                "cached_kv":[[None, None] for i in range(self.n_layers)],
                "kv_mask":None,
                "pos_ids":None,
                }
        return cache


    def cache_kv(self,
                 x=None,
                 self_attn_mask=None,
                 pos_ids=None,
                 cache=None,
                 block_size=None):
        """
        Cache key-value pairs for efficient autoregressive decoding.

        Processes input in chunks when block_size is specified to manage memory usage
        during long sequence processing. Returns updated key-value cache.

        Args:
            x (Tensor, optional): Input tensor of shape [batch_size, seq_len, d_model]
            self_attn_mask (Tensor, optional): Attention mask tensor of shape
                [batch_size, target_len, source_len]
            pos_ids (Tensor, optional): Position indices tensor of shape [batch_size, seq_len]
            cache (Dict, optional): Existing cache dictionary containing previous keys/values
            block_size (int, optional): Chunk size for processing long sequences. When None,
                processes entire sequence at once.

        Returns:
            Dict: Updated cache containing new keys/values. Structure:
                {
                    "cached_kv": [
                        [
                            key, #Tensor with shape [batch_size, num_heads, seq_len, d_head]
                            value #Tensor with shape [batch_size, num_heads, seq_len, d_head]
                        ],
                        ...
                    ],
                    kv_mask: Tensor with shape [batch_size, seq_len],
                    pos_ids: Tensor with shape [batch_size, seq_len]
                }
        """
        if block_size:
            x_len = x.size(1)
            start = 0
            while start < x_len:
                outputs = self.forward(x[:, start:start+block_size],
                                       self_attn_mask[:, start:start+block_size, :start+block_size],
                                       pos_ids[:, start:start+block_size],
                                       cache)
                start += block_size
        else:
            outputs = self.forward(x,
                                   self_attn_mask,
                                   pos_ids,
                                   cache)

        return outputs["cache"]


class TransformerLM(nn.Module):
    """
    Transformer-based Language Model supporting various architectural configurations.
    """
    def __init__(self, **kwargs):
        """
        Initialize Transformer Language Model
            Args:
                **kwargs: Configuration parameters containing:
                    d_model: Hidden dimension size of model embeddings
                    vocab_size: Size of vocabulary for input/output

                    max_len: Maximum sequence length (required if using absolute pos embeddings)
                    factorized_embedding_size: Intermediate size of factorized embedding.
                                               if None FactorizedEmbedding won't be used.
                    norm_type: Normalization type ("layer_norm" or "rms_norm")
                    use_ln_scale: Enable learnable scale parameter in normalization
                    use_ln_bias: Enable learnable bias parameter in normalization
                    norm_before_pred: Apply normalization before final projection
                    norm_after_embedding: Apply normalization after embedding layer

                    emb_dropout: Dropout probability for embeddings
                    embedding_scale: Optional scaling factor for embeddings (e.g. sqrt(d_model))

                    attn_pos_embedding_type: Position encoding for attention ("rope", "relative", "alibi", etc.)
                    use_absolute_pos_embedding: Add absolute position embeddings to inputs
                    pos_embedding_init: Position embedding init method ("sinusoidal", etc.)
                    freeze_pos_embedding: Freeze position embeddings during training
                    pos_embedding_base: Base value for sinusoidal position encoding

                    use_output_bias: Add bias term to output projection
                    share_emb_out_proj: Share weights between input embeddings and output projection

                    ln_eps: Epsilon for numerical stability in normalization
        """
        super(TransformerLM, self).__init__()
        self.d_model = kwargs["d_model"]
        self.vocab_size = kwargs["vocab_size"]
        self.max_len = kwargs.get("max_len", None)
        self.emb_dropout = kwargs.get("emb_dropout", 0)
        self.factorized_embedding_size = kwargs.get("factorized_embedding_size", None)
        self.norm_type = kwargs.get("layer_norm_type", "rms_norm")
        self.use_ln_scale = kwargs.get("use_ln_scale", True)
        self.use_ln_bias = kwargs.get("use_ln_bias", True)
        self.embedding_scale = kwargs.get("embedding_scale", None)
        self.norm_before_pred = kwargs.get("norm_before_pred", True)
        self.norm_after_embedding = kwargs.get("norm_after_embedding", False)
        self.attn_pos_embedding_type = kwargs.get("attn_pos_embedding_type", "rope")
        self.use_absolute_pos_embedding = kwargs.get("use_absolute_pos_embedding", False)
        self.pos_embedding_init = kwargs.get("pos_embedding_init", "sinusoidal")
        self.freeze_pos_embedding = kwargs.get("freeze_pos_embedding", False)
        self.pos_embedding_base = kwargs.get("pos_embedding_base", 500000)
        self.ln_eps = kwargs.get("ln_eps", 1e-5)
        self.use_output_bias = kwargs.get("use_output_bias", False)
        self.share_emb_out_proj = kwargs.get("share_emb_out_proj", False)

        if self.factorized_embedding_size:
            self.word_embedding = FactorizedEmbedding(self.vocab_size,
                                                      self.d_model,
                                                      self.factorized_embedding_size)
        else:
            self.word_embedding = Embedding(self.vocab_size, self.d_model)

        self.emb_dropout = Dropout(self.emb_dropout) if self.emb_dropout else None

        if self.use_absolute_pos_embedding:
            self.pos_embedding = PositionEmbedding(self.max_len,
                                                   self.d_model,
                                                   self.pos_embedding_init,
                                                   self.freeze_pos_embedding,
                                                   self.pos_embedding_base)

        if self.norm_type == "layer_norm":
            norm_cls = partial(LayerNorm,
                               d_model=self.d_model,
                               eps=self.ln_eps,
                               use_scale=self.use_ln_scale,
                               use_bias=self.use_ln_bias)
        elif self.norm_type == "rms_norm":
            norm_cls = partial(RMSNorm,
                               d_model=self.d_model,
                               eps=self.ln_eps)

        if self.norm_after_embedding:
            self.emb_norm = norm_cls()

        self.transformer = Transformer(**kwargs)

        if self.norm_before_pred:
            self.last_norm = norm_cls()

        self.out_proj = Linear(self.d_model, self.vocab_size, self.use_output_bias, self.share_emb_out_proj)


    def reset_parameters(self, initializer_range=0.02):
        """
        Initialize model parameters.
        """
        self.word_embedding.reset_parameters(initializer_range)
        if self.use_absolute_pos_embedding:
            self.pos_embedding.reset_parameters(initializer_range)
        self.transformer.reset_parameters(initializer_range)
        self.out_proj.reset_parameters(initializer_range)


    def forward(self,
                x,
                self_attn_mask=None,
                pos_ids=None,
                cache=None,
                prefix_embeded=None,
                return_logits=True,
                use_checkpoint=False):
        """
        Process input through the complete transformer language model architecture.

        Args:
            x (Tensor): Input token IDs of shape [batch_size, seq_len]
            self_attn_mask (Tensor, optional): Attention mask of shape
                [batch_size, target_len, source_len]. Default: None
            pos_ids (Tensor): Position indices of shape [batch_size, seq_len]
            cache (Dict, optional): Cache dictionary for autoregressive generation
            prefix_embeded (Tensor, optional): Pre-computed embeddings to prepend to input,
                shape [batch_size, prefix_len, d_model]
            return_logits (bool): Whether to compute output logits. Default: True
            use_checkpoint (bool): Use gradient checkpointing for memory optimization. Default: False

        Returns:
            Dict: Contains various model outputs with keys:
                - "output": Final hidden states [batch_size, seq_len, d_model]
                - "logits": Output predictions [batch_size, seq_len, vocab_size] (if return_logits=True)
                - "hidden_states": List of all layer outputs including embeddings
                - "cache": Updated cache dictionary (if cache was provided)
        """
        assert (x < self.vocab_size).all()
        embeded = self.word_embedding(x)

        if prefix_embeded is not None:
            embeded = torch.cat([prefix_embeded, embeded], 1)

        assert pos_ids.size(0) == embeded.size(0)
        assert pos_ids.size(1) == embeded.size(1)
        assert (pos_ids < self.max_len).all()

        if self.embedding_scale:
            embeded = embeded * self.embedding_scale

        if self.use_absolute_pos_embedding:
            embeded = embeded + self.pos_embedding(pos_ids)

        if self.norm_after_embedding:
            embeded = self.norm_emb(embeded)
        if self.emb_dropout:
            embeded = self.emb_dropout(embeded)

        output = embeded
        outputs = self.transformer(output,
                                   self_attn_mask=self_attn_mask,
                                   pos_ids=pos_ids,
                                   cache=cache,
                                   use_checkpoint=use_checkpoint)

        outputs["hidden_states"] = [embeded] + outputs["hidden_states"]

        output = outputs["output"]

        if self.norm_before_pred:
            output = self.last_norm(output)
            outputs["output"] = output

        if return_logits:
            if not self.share_emb_out_proj:
                logits = self.out_proj(output)
            else:
                logits = self.out_proj(output, self.word_embedding.get_embedding())

            outputs["logits"] = logits

        return outputs


    def init_cache(self):
        """
        Initialize an empty cache dictionary for autoregressive generation.

        Returns:
            Dict: Cache ready for incremental processing.
        """
        return self.transformer.init_cache()


    def cache_kv(self,
                 x,
                 cache,
                 self_attn_mask=None,
                 pos_ids=None,
                 prefix_embeded=None):
        """
        Update key-value cache with new input for autoregressive decoding.

        Args:
            x (Tensor): Input tokens of shape [batch_size, new_seq_len]
            cache (Dict): Existing cache dictionary to update
            self_attn_mask (Tensor, optional): Attention mask for new tokens
            pos_ids (Tensor): Position indices for new tokens
            prefix_embeded (Tensor, optional): Pre-computed prefix embeddings

        Returns:
            Dict: Updated cache containing both previous and new key-value pairs
        """
        return self.forward(x,
                            self_attn_mask,
                            pos_ids,
                            cache,
                            prefix_embeded,
                            return_logits=False)["cache"]


class TransformerEncoder(nn.Module):
    """
    Unified Transformer Encoder for multimodal inputs (vision/text)

    Supports both vision inputs (image patches) and text inputs (token embeddings)
    with configurable processing pipelines.
    """
    def __init__(self, **kwargs):
        """
        Initialize multimodal transformer encoder
            Args:
                **kwargs: Configuration parameters containing:
                    input_type (str): Modality type ["image", "text"]
                    patch_size (int): Spatial partitioning of input images (e.g., 14 for 224px images)
                    image_size (int or tuple[int]): The expected size of the input image
                        If an integer is provided, it defines a square size
                        (e.g., `224` for 224x224).
                        If a tuple is provided, it must contain two integers representing (height, width) dimensions
                        (e.g., `(256, 192)`for height=256, width=192).
                        Affects positional embedding initialization and sequence length
                    use_cls (bool): Add class embedding or not
                    d_model (int): Transformer embedding dimension
                    norm_type (str): Normalization layer type, "rms_norm" or "layer_norm"
                    use_ln_scale (bool): Enable scale parameter in normalization
                    use_ln_bias (bool): Enable bias parameter in normalization
                    norm_before_pred (bool): Apply final normalization before classifier head
                    norm_after_embedding (bool): Apply normalization after patch embedding + position embedding
                    pos_embedding_init (str): Position embedding initialization method ("random" or "sinusoidal")
                    freeze_pos_embedding (bool): Freeze position embeddings during training
                    pos_embedding_base (int): Base value for sincos position encoding
                    ln_eps (float): Epsilon value for normalization layers

        """
        super(TransformerEncoder, self).__init__()
        
        self.input_type = kwargs["input_type"]
        self.d_model = kwargs["d_model"]
        self.emb_dropout = kwargs.get("emb_dropout", 0)
        self.norm_type = kwargs.get("layer_norm_type", "rms_norm")
        self.use_ln_scale = kwargs.get("use_ln_scale", True)
        self.use_ln_bias = kwargs.get("use_ln_bias", True)
        self.norm_before_pred = kwargs.get("norm_before_pred", True)
        self.norm_after_embedding = kwargs.get("norm_after_embedding", False)
        self.pos_embedding_init = kwargs.get("pos_embedding_init", "random")
        self.freeze_pos_embedding = kwargs.get("freeze_pos_embedding", False)
        self.pos_embedding_base = kwargs.get("pos_embedding_base", 500000)
        self.ln_eps = kwargs.get("ln_eps", 1e-5)
        self.cls_emb_type = kwargs.get("cls_emb_type", "first")
        
        if self.input_type == "image":
            self.patch_size = kwargs["patch_size"]
            self.use_patch_emb_bias = kwargs.get("use_patch_emb_bias", False)

            image_size= kwargs["image_size"]
            if isinstance(image_size, int):
                self.grid_size = (image_size//self.patch_size, image_size//self.patch_size)
                self.image_size = (image_size, image_size)
            elif len(image_size) == 1:
                self.grid_size = (image_size[0]//self.patch_size, image_size[0]//self.patch_size)
                self.image_size = (image_size[0], image_size[0])
            else:
                self.grid_size = (image_size[0]//self.patch_size, image_size[1]//self.patch_size)
                self.image_size = image_size

            self.patch_embedding = nn.Conv2d(
                in_channels=3,
                out_channels=self.d_model,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                bias=self.use_patch_emb_bias)

            self.max_len = self.grid_size[0] * self.grid_size[1]
            self.use_cls_embedding = kwargs.get("use_cls_embedding", True)
            if self.use_cls_embedding:
                self.cls = nn.Parameter(torch.Tensor(self.d_model))
                self.max_len += 1

            self.use_absolute_pos_embedding = kwargs.get("use_absolute_pos_embedding", True)
            if self.use_absolute_pos_embedding:
                self.pos_embedding = PositionEmbedding(self.max_len,
                                                       self.d_model,
                                                       self.pos_embedding_init,
                                                       self.freeze_pos_embedding,
                                                       self.pos_embedding_base)
        elif self.input_type == "text":
            self.vocab_size = kwargs["vocab_size"]
            self.pad_id = kwargs.get("pad_id", 0)
            self.factorized_embedding_size = kwargs.get("factorized_embedding_size", None)
            self.attn_pos_embedding_type = kwargs.get("attn_pos_embedding_type", "rope")
            self.use_absolute_pos_embedding = kwargs.get("use_absolute_pos_embedding", False)

            if self.factorized_embedding_size:
                self.word_embedding = FactorizedEmbedding(self.vocab_size,
                                                          self.d_model,
                                                          self.factorized_embedding_size)
            else:
                self.word_embedding = Embedding(self.vocab_size, self.d_model)

            if self.use_absolute_pos_embedding:
                self.pos_embedding = PositionEmbedding(self.max_len,
                                                       self.d_model,
                                                       self.pos_embedding_init,
                                                       self.freeze_pos_embedding,
                                                       self.pos_embedding_base)

        if self.norm_type == "layer_norm":
            norm_cls = partial(LayerNorm,
                               d_model=self.d_model,
                               eps=self.ln_eps,
                               use_scale=self.use_ln_scale,
                               use_bias=self.use_ln_bias)
        elif self.norm_type == "rms_norm":
            norm_cls = partial(RMSNorm,
                               d_model=self.d_model,
                               eps=self.ln_eps)

        if self.norm_after_embedding:
            self.emb_norm = norm_cls()

        self.emb_dropout = Dropout(self.emb_dropout) if self.emb_dropout else None

        self.transformer = Transformer(**kwargs)

        if self.norm_before_pred:
            self.last_norm = norm_cls()
        

    def reset_parameters(self, initializer_range=0.02):
        """
        Initialize layer parameters.
        """
        if self.input_type == "image":
            if self.use_cls_embedding:
                nn.init.normal_(self.cls, mean=0.0, std=initializer_range)
        if self.use_absolute_pos_embedding:
            self.pos_embedding.reset_parameters(initializer_range)
        self.transformer.reset_parameters(initializer_range)

 
    def forward(self, x, self_attn_mask=None, return_cls_embedding=False, use_checkpoint=False):
        """
        Processes input images for vision-language alignment

        Args:
            x (Tensor): Input image tensor in shape
                [batch_size, channels, height, width]
            return_cls_embedding: Boolean flag to return [CLS] token embedding (aggregated sequence features).
            use_checkpoint (bool): Enable gradient checkpointing for memory efficiency

        Returns:
            dict: Contains:
                - output (Tensor): Final representations [batch_size, seq_len, d_model]
                - hidden_states (list): All layer outputs including initial embedding
                - cls (Tensor): aggregated features of the entire input sequence

        """
        if self.input_type == "image":
            x = self.patch_embedding(x).flatten(2).transpose(1, 2)
            if self.use_cls_embedding:
                cls = self.cls.repeat(x.shape[0], 1, 1)
                x = torch.cat([cls, x], 1)
            pos_ids = None
            self_attn_mask = None
        elif self.input_type == "text":
            seq_mask = x.ne(self.pad_id)
            x = self.word_embedding(x)
            seq_len = x.size(1)
            pos_ids = seq_mask.cumsum(-1) - 1
            self_attn_mask = None
            if self.cls_emb_type == "last": 
                self_attn_mask = get_attn_mask(seq_len, seq_mask)
                self_attn_mask &= get_subsequent_mask(seq_len).to(self_attn_mask.device)
           

        x_len = x.shape[1]
        embeded = x
        if self.use_absolute_pos_embedding:
            embeded = embeded + self.pos_embedding.get_embedding()[:x_len, :]

        if self.norm_after_embedding:
            embeded = self.emb_norm(embeded)
        if self.emb_dropout:
            embeded = self.emb_dropout(embeded)
        output = embeded

        outputs = self.transformer(output,
                                   pos_ids=pos_ids,
                                   self_attn_mask=self_attn_mask,
                                   use_checkpoint=use_checkpoint)

        outputs["hidden_states"] = [embeded] + outputs["hidden_states"]

        output = outputs["output"]
        if self.norm_before_pred:
            output = self.last_norm(output)
            outputs["output"] = output

        if return_cls_embedding:
            if self.cls_emb_type == "first":
                outputs["cls"] = output[:,0,:]
            elif self.cls_emb_type == "last":
                batch_idx = torch.arange(x.shape[0], device=seq_mask.device)
                step_idx = torch.arange(x.shape[1], device=seq_mask.device)
                last_non_pad_idx = (step_idx * seq_mask).argmax(-1, keepdims=True)
                outputs["cls"] = output[batch_idx, last_non_pad_idx].squeeze(1)

        return outputs


"""
Define All Models
"""

def get_default_generation_config():
    """
    """
    generation_config = {
            "top_p": None,
            "top_k": None,
            "temperature": None,
            "repetition_penalty": None, 
            "repetition_decay_factor": 1,         
            "repetition_window_size": None,         
            "max_repetition_penalty": None,
            "max_decode_steps": 512,
            "strategy": "greedy",
            "system_message": None,
            "generation_mode": "raw"
            }
    return generation_config


def get_attn_mask(query_len, kv_mask):
    """
    Generate attention mask for transformer self-attention mechanism.

    Args:
        query_len (int): Length of target sequence (Q in Q*K^T)
        kv_mask (Tensor): Original key/value mask of shape [batch_size, key_len]

    Returns:
        Tensor: Expanded attention mask of shape [batch_size, query_len, key_len]
            suitable for broadcasting in scaled dot-product attention

    """
    return kv_mask.unsqueeze(1).repeat(1, query_len, 1)


def get_subsequent_mask(seq_len):
    """
    Create causal mask for autoregressive sequence modeling.

    Args:
        seq_len (int): Length of target sequence

    Returns:
        Tensor: Lower triangular boolean mask of shape [seq_len, seq_len]
            where position (i,j) = 1 if j <= i (allows left-context only)

    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=0)


@Registry.register("model.gpt")
class GPT(nn.Module):

    def __init__(self, **kwargs):
        """
        Language Model container class with generation capabilities.

        Combines a Transformer-based language model with generation configuration
        for different text production scenarios (raw generation/chat completion).

        Args:
            **kwargs: Configuration dictionary containing:
                - model_config (Dict): Core model parameters forwarded to TransformerLM
                    - pad_id (int): Padding token ID (default: 0)
                    - eos_id (int): End-of-sequence token ID (default: 2)
                    - max_len (int): Maximum sequence length
                - generation_config (Dict, optional): Text generation settings
                    - system_message (str): Default system prompt for chat mode
                    - generation_mode (str): Operation mode ("raw"/"chat")
        """
        super(GPT, self).__init__()

        self.is_vl_model = ("vision_config" in kwargs["model_config"])
        self.vision_emb_len = 0
        self.image_size = None
        if self.is_vl_model:
            vision_model_config = kwargs["model_config"]["vision_config"]["model_config"]
            if "attention_backend" in kwargs["model_config"]:
                vision_model_config["attention_backend"] = kwargs["model_config"]["attention_backend"]
            self.vision_model = TransformerEncoder(**vision_model_config)
            self.vision_emb_len = self.vision_model.max_len
            self.image_size = self.vision_model.image_size

        self.lm_model = TransformerLM(**kwargs["model_config"])

        if self.is_vl_model:
            self.vision_text_proj = Linear(self.vision_model.d_model, self.lm_model.d_model)

        self.vocab_size = kwargs["model_config"]["vocab_size"]
        self.pad_id = kwargs["model_config"].get("pad_id", 0)
        self.eos_id = kwargs["model_config"].get("eos_id", 2)
        self.max_len = kwargs["model_config"]["max_len"]
        self.generation_config = get_default_generation_config()
        self.set_generation_config(kwargs.get("generation_config", {}))

        self.system_message = self.generation_config.get("system_message", None)
        self.generation_mode = self.generation_config.get("generation_mode", "raw")
        self.is_chat_model = ("chat" in self.generation_mode) if self.generation_mode else False

        self.precision = "float32"


    def reset_parameters(self, initializer_range=0.02):
        """
        Initialize layer parameters.
        """
        if self.is_vl_model:
            self.vision_model.reset_parameters(initializer_range)
        self.lm_model.reset_parameters(initializer_range)
        if self.is_vl_model:
            self.vision_text_proj.reset_parameters(initializer_range)


    def forward(self,
                x,
                x_vision=None,
                x_vision_mask=None,
                cache=None,
                targets=None,
                use_checkpoint=False):
        """
        Complete forward pass for transformer language model with caching support.

        Args:
            x (Tensor): Input token IDs of shape [batch_size, seq_len]
            x_vision (Tensor, optional): Raw image input of shape
                [batch_size, channels, height, width]. If provided, the image will be
                processed by a Vision Transformer (ViT) to generate visual prefix embeddings.
                **Note**: For samples with padded `x_vision` (e.g., no actual image),
                their corresponding `x_vision_mask` must be `False`.
            x_vision_mask (Tensor, optional): Boolean mask of shape [batch_size], indicating
                **whether each sample in the batch contains valid visual input**.
                - `True`: The sample's `x_vision` is a real image.
                - `False`: The sample's `x_vision` is a padded placeholder (visual features
                  for this sample will be ignored entirely).
            cache (Dict, optional): Cache dictionary for incremental decoding containing:
                - "cached_kv": Historical key/value
                - "kv_mask": Historical key/value mask
                - "pos_ids": Cumulative position indices
                - "previous_tokens": Accumulated token IDs
            targets (Tensor, optional): Target tokens for loss calculation of shape
                [batch_size, seq_len]
            use_checkpoint (bool): Enable gradient checkpointing for memory efficiency

        Returns:
            Dict: Model outputs containing:
                - "logits": Prediction scores [batch_size, seq_len, vocab_size]
                - "loss": Cross-entropy loss (if targets provided)
                - "cache": Updated cache state (if cache provided)
                - Other transformer intermediate states
        """
        prefix_embeded = None
        if self.is_vl_model and x_vision is not None:
            vision_encoded = self.vision_model(x_vision, use_checkpoint=use_checkpoint)["output"]
            prefix_embeded = self.vision_text_proj(vision_encoded)

        seq_mask, kv_mask = self.auto_infer_mask(x, prefix_embeded, x_vision_mask, cache)
        seq_len = seq_mask.size(1)
        
        #apply pad mask
        self_attn_mask = get_attn_mask(seq_len, kv_mask)
        
        #apply casual mask
        self_attn_mask[:,-seq_len:,-seq_len:] &= get_subsequent_mask(seq_len).to(self_attn_mask.device)

        pos_ids = kv_mask.cumsum(-1) - 1
        outputs = self.lm_model(x,
                                self_attn_mask=self_attn_mask,
                                pos_ids=pos_ids[:, -seq_len:],
                                cache=cache,
                                prefix_embeded=prefix_embeded,
                                return_logits=True,
                                use_checkpoint=use_checkpoint)

        if cache:
            cache["kv_mask"] = kv_mask
            cache["pos_ids"] = pos_ids
            if cache["previous_tokens"] is not None:
                cache["previous_tokens"] = torch.cat([cache["previous_tokens"], x], 1)
            else:
                cache["previous_tokens"] = x

        if targets is not None:
            logits = outputs["logits"]
            logits_len = logits.shape[1]
            target_len = targets.shape[1]
            if logits_len != target_len:
                logits = outputs["logits"][:,-target_len:,:].contiguous()
            outputs["loss"] = F.cross_entropy(logits.view(-1, self.lm_model.vocab_size),
                                              targets.view(-1),
                                              ignore_index=self.pad_id)

        return outputs


    def init_cache(self):
        """
        Initialize cache dictionary for autoregressive decoding.

        Returns:
            Dict: Cache structure containing:
                - Transformer layer key/value states (from lm_model)
                - "previous_tokens": Initialized as None to track generated tokens
        """
        cache = self.lm_model.init_cache()
        cache["previous_tokens"] = None
        return cache


    def auto_infer_mask(self, x, prefix_embeded=None, x_vision_mask=None, cache=None):
        """
        Automatically generate sequence masks considering prefix embeddings and cache state.

        Args:
            x (Tensor): Input token IDs of shape [batch_size, seq_len]
            prefix_embeded (Tensor, optional): Pre-computed prefix vision embeddings
                of shape [batch_size, prefix_len, d_model]
            x_vision_mask (Tensor, optional): Boolean mask of shape [batch_size], indicating
                **whether each sample in the batch contains valid visual input**.
                - `True`: The sample's `x_vision` is a real image.
                - `False`: The sample's `x_vision` is a padded placeholder (visual features
                  for this sample will be ignored entirely).
            cache (Dict, optional): Existing cache dictionary containing kv, kv_mask, pos_ids

        Returns:
            tuple: Contains two elements:
                - seq_mask: Padding mask for current sequence [batch_size, total_seq_len]
                - kv_mask: Combined key/value mask including historical context [batch_size, total_key_len]

        """
        bz = x.size(0)
        seq_mask = x.ne(self.pad_id).long()

        if prefix_embeded is not None:
            prefix_len = prefix_embeded.size(1)
            prefix_mask = torch.ones([bz, prefix_len], device=seq_mask.device).long()
            if x_vision_mask:
                prefix_mask = prefix_mask * x_vision_mask.unsqueeze(-1)
            seq_mask = torch.cat([prefix_mask, seq_mask], 1)

        kv_mask = seq_mask

        if cache and cache["kv_mask"] is not None:
            kv_mask = torch.cat([cache["kv_mask"], seq_mask], 1)


        return seq_mask, kv_mask


    def init_decoding(self,
                      x,
                      x_vision=None,
                      x_vision_mask=None,
                      cache=None):
        """
        Initialize search states for autoregressive sequence generation.

        Args:
            x (Tensor): Initial input tokens of shape [batch_size, init_len]
            x_vision (Tensor, optional): Raw image input of shape
                [batch_size, channels, height, width]. If provided, will be processed by ViT
                to initialize visual context for decoding.
            x_vision_mask (Tensor, optional): Boolean mask of shape [batch_size], indicating
                **whether each sample in the batch contains valid visual input**.
                - `True`: The sample's `x_vision` is a real image.
                - `False`: The sample's `x_vision` is a padded placeholder (visual features
                  for this sample will be ignored entirely).
            cache (Dict, optional): Partial cache for warm-start decoding

        Returns:
            Dict: Search state dictionary containing:
                - "hypothesis": Empty tensor to store generated sequences [batch_size, 0]
                - "finished": Completion flags [batch_size, 1]
                - "scores": Accumulated log probabilities [batch_size, 1]
                - "logits": Container for model predictions [batch_size, 0, vocab_size]
                - "cache": Initialized/updated cache
                - "inputs": Last token to start generation [batch_size, 1]
        """
        search_states = {}
        if not cache:
            cache = self.init_cache()

        bz,x_len = x.size(0), x.size(1)
        search_states["hypothesis"] = torch.zeros(bz, 0, dtype=torch.uint8, device=x.device)
        search_states["finished"] = torch.zeros(bz, 1, dtype=torch.uint8, device=x.device)
        search_states["scores"] = torch.zeros([bz, 1], dtype=torch.float, device=x.device)
        search_states["logits"] = torch.zeros([bz, 0, self.lm_model.vocab_size], dtype=torch.float, device=x.device)

        if x_len > 1 or x_vision is not None:
            cache = self.forward(x[:, :-1], x_vision, x_vision_mask, cache)["cache"]

        search_states["cache"] = cache
        search_states["inputs"] = x[:, -1:]

        return search_states


    def step(self, search_states):
        """
        Perform a single decoding step in autoregressive sequence generation.

        Args:
            search_states (Dict): Current search state containing:
                - "inputs": Current input tokens of shape [batch_size, 1]
                - "cache": Accumulated key-value cache from previous steps

        Returns:
            Tensor: Logits for next token predictions of shape
                [batch_size, 1, vocab_size]
        """
        x = search_states["inputs"]
        cache = search_states["cache"]
        outputs = self.forward(x, cache=cache)

        return outputs["logits"]


    def set_generation_config(self, generation_config):
        """
        """
        for k in generation_config:
            if k in self.generation_config:
                self.generation_config[k] = generation_config[k]
            else:
                logger.info(f"{k} not found in generation config.")


    def search(self,
               x,
               strategy=None,
               x_vision=None,
               x_vision_mask=None,
               cache=None,
               top_p=None,
               top_k=None,
               repetition_penalty=None,
               repetition_decay_factor=1,
               repetition_window_size=None,
               max_repetition_penalty=None,
               temperature=None,
               max_decode_steps=None,
               logits_processors=None):
        """
        Autoregressive text generation with configurable decoding strategies.

        Args:
            x (Tensor):               Initial input tokens [batch_size, seq_len]
            strategy (str):           Decoding strategy ("greedy"|"sample")
            x_vision (Tensor, optional): Raw image input [batch_size, C, H, W].
                If provided, processed by ViT to initialize visual context.
            x_vision_mask (Tensor, optional): Boolean mask of shape [batch_size], indicating
                **whether each sample in the batch contains valid visual input**.
                - `True`: The sample's `x_vision` is a real image.
                - `False`: The sample's `x_vision` is a padded placeholder (visual features
                  for this sample will be ignored entirely).
            cache (Dict):            Warm-start cache for incremental decoding
            top_p (float):           Nucleus sampling threshold (0-1)
            top_k (int):             Top-k filtering threshold
            repetition_penalty (float): Penalty factor for repeated tokens (>=1) 
            repetition_decay_factor (float): decay factor applied to repetition penalty based on distance.  
            repetition_window_size (int): Number of prior tokens to check for repetition.
            max_repetition_penalty (float): Limits the maximum penalty multiplier for repeated tokens.
            temperature (float):     Temperature for probability sharpening (>0)
            max_decode_steps (int):  Maximum generation steps per batch
            logits_processors: A list of functions
                sequentially process and modify language model's next-token prediction logits.
        Yields:
            Dict: Search states per generation step containing:
                - "hypothesis": Growing sequence tensor [batch_size, current_len]
                - "finished":   Completion flags [batch_size, 1]
                - "scores":     Accumulated sequence scores
                - "cache":      Updated key-value cache
                - "inputs":     Next input tokens
        """

        # Load parameters from generation config if not provided
        top_p = top_p or self.generation_config["top_p"]
        top_k = top_k or self.generation_config["top_k"]
        temperature = temperature or self.generation_config["temperature"]
        repetition_penalty = repetition_penalty or self.generation_config["repetition_penalty"]
        repetition_decay_factor = repetition_decay_factor or self.generation_config["repetition_decay_factor"]
        repetition_window_size = repetition_window_size or self.generation_config["repetition_window_size"]
        max_repetition_penalty = max_repetition_penalty or self.generation_config["max_repetition_penalty"]
        max_decode_steps = max_decode_steps or self.generation_config["max_decode_steps"]
        strategy = strategy or self.generation_config["strategy"]

        # Validate decoding strategy
        assert strategy in ["greedy", "sample"]
        
        if logits_processors is None:
            logits_processors = []
        if repetition_penalty:
            logits_processors = [
                    lambda logits, states:process_repetition_penalty(
                        logits,
                        states["cache"]["previous_tokens"],
                        #states["hypothesis"],
                        repetition_penalty,
                        repetition_decay_factor,
                        repetition_window_size)
                    ] + logits_processors

        #LLM decoding has hit the maximum allowed sequence length, inference terminate.
        x_len = x.shape[1]
        cache_len = 0
        if cache:
            cache_len = cache["pos_ids"].max().item() + 1
        if x_vision is not None:
            if cache_len + self.vision_emb_len + x_len >= self.max_len:
                return
        else:
            if cache_len + x_len >= self.max_len:
                return

        # Initialize search states with context prefill
        search_states = self.init_decoding(x,
                                           x_vision,
                                           x_vision_mask,
                                           cache)

        steps = 0
        while not search_states["finished"].all() and steps < max_decode_steps:

            # Get next-token logits from model
            logits = self.step(search_states).squeeze(1)

            for fn in logits_processors:
                logits = fn(logits, search_states)
            
            if "disallow_tokens_idxs" in search_states:
                for i,idxs in enumerate(search_states["disallow_token_idxs"]):
                    logits[i, idxs] = -torch.inf
    
            # Greedy decoding
            if strategy == "greedy":
                ids = logits.argmax(-1, keepdim=True)
            # Stochastic sampling
            elif strategy == "sample":
                # Temperature scaling
                if temperature:
                    logits /= temperature
                # Apply top-k/p filtering
                ids = top_k_top_p_sampling(logits, top_k=top_k, top_p=top_p)

            # Mask finished sequences with pad tokens
            ids = ids * (1 - search_states["finished"].long()) + self.pad_id * search_states["finished"].long()

            # Update search states
            search_states["inputs"] = ids
            search_states["hypothesis"] = torch.cat([search_states["hypothesis"], ids], 1)

            # Update completion flags (EOS detected)
            search_states["finished"] = search_states["finished"].logical_or(ids.eq(self.eos_id))
            steps += 1
            yield search_states # Stream intermediate states

            #LLM decoding has hit the maximum allowed sequence length, inference terminate.
            if search_states["cache"]:
                cache_len = search_states["cache"]["pos_ids"].max().item() + 1
            if cache_len >= self.max_len:
                break

        if steps > 0 and cache_len < self.max_len:
            # Final cache update after generation completes
            search_states["cache"] = self.forward(search_states["hypothesis"][:, -1:],
                                                  cache=search_states["cache"])["cache"]

 
def process_repetition_penalty(
        logits, 
        input_ids, 
        repetition_penalty, 
        repetition_decay_factor=1.0,
        repetition_window_size=None, 
        max_repetition_penalty=None,
        repetition_ignore_ids=None):
    """
    Apply repetition penalty to discourage repeated token generation.

    Args:
        logits (Tensor): Unnormalized model outputs of shape [batch_size, vocab_size]
        input_ids (Tensor): Previously generated token IDs of shape [batch_size, seq_len]
        repetition_penalty (float): Penalty factor where:
            >1 decreases probability of repeated tokens
            <1 increases probability of repeated tokens
            =1 no effect 
        repetition_window_size (int, optional): 
            Specifies the number of previous tokens to consider when applying repetition penalty. 
            If set to None, the penalty considers the entire preceding sequence.
        max_repetition_penalty (float, optional):
            Limits the maximum penalty multiplier for repeated tokens.
    Returns:
        Tensor: Modified logits with repetition penalty applied
    """
    if repetition_window_size:
        input_ids = input_ids[:, -repetition_window_size:]
     
    #score = torch.gather(logits, 1, input_ids)
    #score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
    #logits = logits.scatter(1, input_ids, score)
    
    if repetition_decay_factor is None:
        repetition_decay_factor = 1

    decays = torch.pow(
            torch.full_like(input_ids, repetition_decay_factor, dtype=logits.dtype),
            torch.arange(0, input_ids.shape[-1], device=logits.device, dtype=logits.dtype)
            ).flip(-1)
    mask = torch.zeros_like(logits)
    mask.scatter_add_(-1, input_ids, decays)
    
    if repetition_ignore_ids:
        mask[:,repetition_ignore_ids] = 0

    penalty_factor = torch.pow(torch.where(logits < 0, repetition_penalty, 1 / repetition_penalty), mask)
    
    if max_repetition_penalty:
        penalty_factor = torch.clamp(penalty_factor, 1.0/max_repetition_penalty, max_repetition_penalty)
    
    logits = logits * penalty_factor

    return logits


def top_k_top_p_sampling(logits, top_k=-1, top_p=-1):
    """
    Perform nucleus (top-p) and/or top-k sampling for diverse generation.

    Args:
        logits (Tensor): Unnormalized model outputs of shape [batch_size, vocab_size]
        top_k (int): Number of top candidates to keep (<=0 disables)
        top_p (float): Cumulative probability threshold (0-1, <=0 disables)

    Returns:
        Tensor: Sampled token IDs of shape [batch_size, 1]
    """
    probs = torch.softmax(logits, -1)

    if (top_k and top_k > 0) or (top_p and top_p > 0):
        _logits, _indices = torch.sort(logits, descending=True)

        if top_k and top_k > 0:
            probs[logits < _logits[:, top_k, None]] = 0

        if top_p and top_p > 0:
            cumulative_logits = torch.cumsum(torch.softmax(_logits, -1), dim=-1)
            need_filter =  (cumulative_logits > top_p)

            need_filter[:, 1:] = need_filter[:, :-1].clone()
            need_filter[:, 0] = 0

            filter_indice = need_filter.scatter(1, _indices, need_filter)
            probs[filter_indice] = 0

        probs /= torch.sum(probs, dim=-1, keepdim=True)

    ids = torch.multinomial(probs, 1)
    return ids


def build_generation_inputs(model,
                            inputs,
                            tokenizer,
                            device="cpu",
                            max_len=None,
                            image_processor=None,
                            images=None,
                            cache=None):

    """
    Streamingly generates text sequences using a language model with configurable decoding constraints.

    Args:
        model: Language model instance used for text generation
        inputs: Initial input text or messages to start generation
        tokenizer: Tokenizer for text <-> token conversion
        device: Hardware accelerator for computation ('cpu' or 'cuda')
        max_len: Maximum input sequence length
        image_processor: Image preprocessing function/object for processing input images
                         (e.g., resizing, normalization, format conversion).
        images: List[PIL.Image] Collection of input image data.
        cache: Key-value cache for accelerating autoregressive generation
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        - x: Input token IDs after text encoding and padding/truncation
        - x_vision: Processed image tensors stacked along batch dimension
        - x_vision_mask: Boolean mask indicating which text positions have associated images
    """
    model.eval()
    if model.is_chat_model:
        if cache:
            x = [tokenizer.encode_dialog_prompt(messages, bos=False) for messages in inputs]
        else:
            x = [tokenizer.encode_dialog_prompt(messages, bos=True) for messages in inputs]
        x = [[model.pad_id] * (max(len(t) for t in x) - len(t)) + t for t in x]
        x = torch.tensor(x, dtype=torch.long, device=device)
    else:
        x = [tokenizer.encode(text, bos=(cache is None), eos=False) for text in inputs]
        x = [[model.pad_id] * (max(len(t) for t in x) - len(t)) + t for t in x]
        x = torch.tensor(x, dtype=torch.long, device=device)
    max_len = max_len or model.max_len
    x = x[:, -max_len:]

    x_vision = None
    x_vision_mask = None
    
    if image_processor and images:
        x_vision = []
        x_vision_mask = []
        for image in images:
            if image:
                x_vision.append(image_processor(image))
                x_vision_mask.append(1)
            else:
                x_vision.append(torch.zeros(model.image_size))
                x_vision_mask.append(0)
        x_vision = torch.stack(x_vision).to(device)
        if model.precision in ["bf16", "bfloat16"]:
            x_vision = x_vision.bfloat16()
        if model.precision in ["fp16", "float16"]:
            x_vision = x_vision.half()
        x_vision_mask = torch.tensor(x_vision_mask, dtype=torch.long, device=device)

    return x, x_vision, x_vision_mask


def get_default_model_config(model_size="base", vocab_size=None):
    """
    """
    GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    config_mapping = {
            "small":
            {
                "n_layers": 6,
                "d_model": 512,
                "d_ff": 1024,
                "n_heads": 8,
                "n_kv_heads": 8,
                "vocab_size": vocab_size if vocab_size else 30256,
            },
            "base":
            {
                "n_layers": 12,
                "d_model": 768,
                "d_ff": 2304,
                "n_heads": 12,
                "n_kv_heads": 4,
                "vocab_size": vocab_size if vocab_size else 50256,
            },
            "large":
            {
                "n_layers": 24,
                "d_model": 1024,
                "d_ff": 3072,
                "n_heads": 8,
                "n_kv_heads": 4,
                "vocab_size": vocab_size if vocab_size else 50256,
            },
            "xl":
            {
                "n_layers": 36,
                "d_model": 1024,
                "d_ff": 3072,
                "n_heads": 8,
                "n_kv_heads": 4,
                "vocab_size": vocab_size if vocab_size else 50256,
            },
    }
    model_config = {
        "model_config": {
            "model_name": f"MimixLM-{model_size}",
            "vocab_size": config_mapping[model_size]["vocab_size"],
            "max_len": 1024,
            "share_emb_out_proj": True,
            "pad_id": config_mapping[model_size]["vocab_size"] - 1,
            "eos_id": config_mapping[model_size]["vocab_size"] - 5,
            "dtype": "bf16",
            "n_layers": config_mapping[model_size]["n_layers"],
            "d_model": config_mapping[model_size]["d_model"],
            "d_ff": config_mapping[model_size]["d_ff"],
            "n_heads": config_mapping[model_size]["n_heads"],
            "n_kv_heads": config_mapping[model_size]["n_kv_heads"],
            "attention_backend": "torch_native"
        },
        "tokenizer_config": {
            "tokenizer": "bpe",
            "pat_str": GPT2_PATTERN,
            "num_reserved_special_tokens": 256,
            "special_tokens": {
                "bos": [
                    "<|begin_of_text|>",
                    config_mapping[model_size]["vocab_size"] - 6
                ],
                "eos": [
                    "<|end_of_text|>",
                    config_mapping[model_size]["vocab_size"] - 5
                ],
                "eot": [
                    "<|eot_id|>",
                    config_mapping[model_size]["vocab_size"] - 4
                ],
                "start_header": [
                    "<|start_header_id|>",
                    config_mapping[model_size]["vocab_size"] - 3
                ],
                "end_header": [
                    "<|end_header_id|>",
                    config_mapping[model_size]["vocab_size"] - 2
                ],
                "pad": [
                    "<|pad_id|>",
                    config_mapping[model_size]["vocab_size"] - 1
                ]
            },
            "use_tiktoken": True
        },
        "generation_config": {
            "strategy": "sample",
            "top_p": 0.95,
            "top_k": None,
            "temperature": 0.95,
            "repetition_penalty": None,
            "max_decode_steps": 256,
            "generation_mode": "raw"
        }
    }

    return model_config


def init_model(args):
    """
    Initializes a language model from configuration files and saves initialized weights.

    Loads model configuration, creates model instance with randomized parameters,
    and persists both config and initial weights to specified paths.

    Args:
        args: Object containing initialization paths with attributes:
            - init_config_path: (str) Path to source configuration JSON file
            - init_model_path: (str) Directory path to save initialized model artifacts
            - init_weight_path: (List[str]) Path(s) to pretrained weight files for model parameter initialization
    Returns:
        LM: Initialized language model instance with fresh parameters
    """
    config_path, save_path = args.init_config_path, args.init_model_path

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    if config_path:
        with open(os.path.join(config_path), "rb") as f:
            config = json.load(f)
    else:
        local_rank0_log("Model config not found, use default config.")
        config = get_default_model_config(
                model_size=args.model_size, vocab_size=args.init_vocab_size)

    with open(os.path.join(save_path, "model_config.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(config, ensure_ascii=False, indent=4))
    
    model_type = config["model_config"].get("model_type", "gpt")
    model_cls = Registry.get_required(f"model.{model_type}")
    model = model_cls(**config)

    model.reset_parameters(initializer_range=args.initializer_range)  

    if args.init_weight_path:
        model = load_model_weights(model, collect_file_paths(args.init_weight_path))

    dtype = config["model_config"].get("dtype", "float")
    if dtype in ["bfloat16", "bf16"]:
        model = model.bfloat16()
    elif dtype in ["float16", "fp16"]:
        model = model.float16()

    torch.save(model.state_dict(), os.path.join(save_path, "model_weights"))
    show_model_info(model)
    return model


def load_model_weights(model, weight_path_list):
    """
    """
    reloaded_keys = set()
    for weight_path in weight_path_list:
        local_rank0_log(f"loading weights file {weight_path}.")

        state_dict = torch.load(weight_path, weights_only=True, map_location="cpu")

        model_weights = {}
        for k,v in model.named_parameters():
            if k in state_dict:
                model_weights[k] = state_dict[k]
                reloaded_keys.add(k)
            elif ("module." + k) in state_dict:
                model_weights[k] = state_dict["module." + k]
                reloaded_keys.add(k)
        if len(model_weights) > 0:
            model.load_state_dict(model_weights, False)

    missing_keys = []
    for k,v in model.named_parameters():
        if k not in reloaded_keys:
            missing_keys.append(k)
    if len(missing_keys) > 0:
        local_rank0_log(f"Weight {missing_keys} not found in model file", level=logging.WARNING)

    return model


def load_model(model_path, lora_rank=None):
    """
    Loads a pre-trained language model from disk, optionally injecting LoRA adapters.

    Expects model artifacts to follow standard structure:
    - Model config: <model_path>/model_config.json
    - Model weights: <model_path>/model_weights or model_weights_1, _2 ...

    Args:
        model_path (str): Directory path containing saved model configuration and weights
        lora_rank (int, optional): Rank parameter for LoRA adaptation. When specified:
            - Adds LoRA wegiht to linear modules
            If None, loads base model without modifications

    Returns:
        LM: Loaded language model instance.
    """
    local_rank0_log("Load model now.")

    with open(os.path.join(model_path, "model_config.json"), "rb") as f:
        config = json.load(f)

    formated_config = json.dumps(config, ensure_ascii=False, indent=4)
    local_rank0_log(f"List model config: {formated_config}")

    config["model_config"]["lora_rank"] = lora_rank

    model_type = config["model_config"].get("model_type", "gpt")
    model_cls = Registry.get_required(f"model.{model_type}")
    model = model_cls(**config)

    dtype = config["model_config"].get("dtype", "float32")
    if dtype in ["bfloat16", "bf16"]:
        model = model.bfloat16()
        model.precision = dtype
    elif dtype in ["float16", "fp16"]:
        model = model.float16()
        model.precision = dtype

    weight_path_list = []
    for f in os.listdir(model_path):
        if f.startswith("model_weights"):
            weight_path_list.append(os.path.join(model_path, f))

    model = load_model_weights(model, weight_path_list)

    show_model_info(model)
    local_rank0_log("Load model done.")
    return model


def load_tokenizer(model_path):
    """
    Loads a tokenizer from the specified model directory based on stored configuration.

        Requires the following files in model_path:
        - model_config.json: Main configuration file containing tokenizer settings
        - tokenizer.model OR vocab.txt: Vocabulary file depending on configuration

        Args:
            model_path (str): Path to directory containing model artifacts and tokenizer config

        Returns:
            Tokenizer: Initialized tokenizer instance (specific subclass depends on config)

    """
    with open(os.path.join(model_path, "model_config.json"), "rb") as f:
        config = json.load(f)
    
    vocab_path = os.path.join(model_path, "tokenizer.model")
    kwargs = config["tokenizer_config"]
    kwargs["vocab_path"] = vocab_path
    
    tokenizer_type = config["tokenizer_config"]["tokenizer"]
    tokenizer_cls = Registry.get_required(f"tokenizer.{tokenizer_type}")
    tokenizer = tokenizer_cls(**kwargs)

    return tokenizer


def load_image_processor(model_path):
    """
    Load and build the image processor from model configuration files.

    Reads the vision model configuration from 'model_config.json' in the specified model directory
    and constructs the corresponding image processing pipeline.

    Args:
        model_path (str/Path): Path to the directory containing model configuration files.

    Returns:
        Compose: A torchvision.transforms.Compose object containing the image processing pipeline.
    """
    with open(os.path.join(model_path, "model_config.json"), "rb") as f:
        config = json.load(f)
    return build_image_processor(config["model_config"]["vision_config"])


def build_image_processor(config):
    """
    Construct the image processing pipeline using specified configuration parameters.

    Creates a sequence of image transformations including:
    - RGB conversion
    - Resizing and center cropping
    - Tensor conversion
    - Value rescaling
    - Normalization

    Args:
        config (dict): Configuration dictionary containing image processing parameters.

    Returns:
        Compose: A torchvision.transforms.Compose object with the processing pipeline.

    Configuration Parameters:
        image_size (int): Target dimensions for resizing and cropping
        resample (PIL.Image.Resampling): Interpolation method for resizing (e.g., BILINEAR)
        rescale_factor (float): Scaling factor for tensor values (typically 1/255)
        norm_mean (list[float]): Mean values for normalization (per-channel)
        norm_std (list[float]): Standard deviations for normalization (per-channel)
    """
    image_size = config["model_config"]["image_size"]
    resample = config["resample"]
    rescale_factor = config["rescale_factor"]
    norm_mean = config["norm_mean"]
    norm_std = config["norm_std"]
    return Compose([
        Lambda(lambda img: img.convert('RGB')),
        Resize(image_size, interpolation=resample),
        CenterCrop(image_size),
        lambda x:x.convert("RGB"),
        ToTensor(),
        lambda x:x*(255*rescale_factor),
        Normalize(norm_mean, norm_std)
    ])


def load_model_with_processor(model_path, lora_rank=None):
    """
    Unified loader for AI models with their associated processing tools

    Loads core components for both standard language models (LM) and
    vision-language models (VLM) with automatic type detection

    Args:
        model_path (str): Directory containing model artifacts and tokenizer files
        lora_rank (int, optional): LoRA adaptation rank. See load_model() for details

    tuple: Components vary by model type:
            LM: (model, tokenizer, None)
            VLM: (model, tokenizer, image_preprocessor)

    """
    local_rank0_log("Load model and tokenizer now.")
    model,tokenizer = load_model(model_path, lora_rank=lora_rank), load_tokenizer(model_path)

    assert model.pad_id == tokenizer.pad_id
    assert model.vocab_size >= tokenizer.n_words

    image_processor = None
    if model.is_vl_model:
        image_processor = load_image_processor(model_path)

    local_rank0_log("Load model and tokenizer done.")
    return model,tokenizer,image_processor


def save_model(model_path, save_path, model):
    """
    Persists model artifacts with safe file handling and directory creation.

    Creates save directory if nonexistent. Copies all original model files except
    weight files, combined with newly saved weights from the trained model.

    Args:
        model_path (str): Source directory containing original model artifacts
        save_path (str): Destination directory for saving updated model
        model (LM): Model instance to save

    - Creates directory tree for save_path if not exists
    - Overwrites existing model_weights file in save_path
    - Copies non-weight files from model_path to save_path

    """
    local_rank0_log("Save model now.")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_path, "model_weights"))
    for f in os.listdir(model_path):
        if f not in ["model_config.json", "tokenizer.model"]:
            continue
        shutil.copyfile(os.path.join(model_path, f), os.path.join(save_path, f))
    local_rank0_log("Save model done.")


def stream_generate(model,
                    inputs,
                    tokenizer,
                    device,
                    image_processor=None,
                    images=None,
                    cache=None):
    """
    Streamingly generates text sequences using a language model with configurable decoding constraints.

    Args:
        model: Language model instance used for text generation
        inputs: Initial input text or messages to start generation
        tokenizer: Tokenizer for text <-> token conversion
        device: Hardware accelerator for computation ('cpu' or 'cuda')
        image_processor: Image preprocessing function/object for processing input images
                         (e.g., resizing, normalization, format conversion).
        images: List[PIL.Image] Collection of input image data.
        cache: Key-value cache for accelerating autoregressive generation

    Yields:
        dict: Search states containing intermediate generation results with keys
    """
    x,x_vision,x_vision_mask = build_generation_inputs(model,
                                                       inputs,
                                                       tokenizer,
                                                       device=device,
                                                       image_processor=image_processor,
                                                       images=images,
                                                       cache=cache)

    with torch.no_grad():
        for search_states in model.search(x,
                                             x_vision=x_vision,
                                             x_vision_mask=x_vision_mask,
                                             cache=cache):
            search_states["text_buffer"] = []
            for ids in search_states["hypothesis"].cpu().numpy():
                search_states["text_buffer"].append(tokenizer.decode(ids))
            yield search_states


def interact_with_cli(args):
    """
    Launches an interactive session with the language model, supporting both chat and completion modes.

    Implements conversation memory management and special command processing.
    Designed for terminal-based interaction with streaming generation.

    Args:
        args: Configuration object containing:

    Returns:
        None

    Chat Mode Features:
        - Maintains conversation history with role annotations
        - Supports system message initialization
        - Special commands:
            :restart  Reset conversation history and model cache
            :exit     Terminate the interactive session

    """
    model,tokenizer,image_processor = load_model_with_processor(args.model_path)
    cache = None
    messages = []
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else"cpu")
    model = model.to(device)

    if model.is_chat_model:
        print("\033[32mChat with MimixLM now.\033[0m")
        print("\033[32mUse ':restart' to clean memory and ':exit' to close dialog.\033[0m")

    cache = None
    image = None
    while True:
    
        while model.is_vl_model and not image:
            text = input("Input image path:\n").strip()
            path = text.strip()
            try:
                image = Image.open(path)
                print("load image success.")
            except:
                print("load image failed.")
                continue
    
        if model.is_chat_model:
            text = input("User:\n").strip()
        else:
            text = input("Input:").strip()

        if text == ":exit":
            break

        if (text == ":restart" and model.is_chat_model) or model.generation_mode == "single_turn_chat":
            messages = []
            image = None
            cache = None
            print("\033[32mClear All History.\033[0m")
            if text == ":restart":
                continue

        inputs = [text]
        if model.is_chat_model:
            if len(messages) == 0 and model.system_message:
                messages = [
                        {"role":"system", "content":model.system_message},
                        {"role":"user", "content":text}
                        ]
                inputs = [messages]
            else:
                messages += [{"role":"user", "content":text}]
                inputs = [messages[-1:]]

        if model.is_chat_model:
            print("Assistant:", end="\n", flush=True)
        else:
            print(text, end="", flush=True)


        printed_text = ""
        for search_states in stream_generate(model=model,
                                             inputs=inputs,
                                             tokenizer=tokenizer,
                                             device=device,
                                             image_processor=image_processor,
                                             images=[image] if cache is None else None,
                                             cache=cache):
            gen_text = search_states["text_buffer"][0]
            if gen_text.endswith("�"):
                continue
            print(gen_text[len(printed_text):], end="", flush=True)
            printed_text = gen_text
        print(gen_text[len(printed_text):].strip("�"))

        if model.is_chat_model:
            messages += [{"role":"assistant", "content":gen_text}]
            cache = search_states["cache"]


def create_gradio_app(generate_response_fn, enable_vision):
    """
    Instantiate a Gradio chat interface with dynamic layout configuration.

    Constructs a responsive chat interface supporting both text and optional image inputs.
    Implements state management for conversation history and supports responsive UI components.

    Args:
        generate_response_fn: Callable function that takes (message, chat_state, [image])
            and returns (message_input, chatbot, updated_state)
        enable_vision: Boolean flag to enable/disable image upload capabilities

    Returns:
        gr.Blocks: Configured Gradio interface object with chat components
    """

    css = """
    .gradio-container {max-width: 1000px !important}
    .right-panel {
        background: #f8f9fa;
        padding: 20px;
        border-left: 1px solid #dee2e6;
        height: 600px;
    }
    .upload-box {
        border: 2px dashed #ced4da;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        transition: border-color 0.3s;
    }
    .upload-box:hover {border-color: #339af0;}
    .preview-img {max-width: 220px;}
    
    
    """
    import gradio as gr
        
    with gr.Blocks(title="MimixAI Demo", css=css) as interface:
        chat_state = gr.State({"messages": [], "cache": None})

        with gr.Row():

            with gr.Column(scale=2):
                gr.Markdown("## 💬 Interact With MimixLM")

                chatbot = gr.Chatbot(
                    height=350,
                    #avatar_images=("user.png", "bot.png"),
                    show_copy_button=True
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Input text...",
                        lines=2,
                        max_lines=4,
                        show_label=False,
                        container=False,
                        scale=5
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("🧹 Clear History", variant="secondary")

            if enable_vision:
                with gr.Column(scale=1, elem_classes="right-panel"):
                    gr.Markdown("## 🖼 Upload Image")
                    with gr.Column(elem_classes="upload-box"):
                        image_upload = gr.Image(
                            type="filepath",
                            label="Drag Image Here",
                            height=200,
                            elem_classes="preview-img"
                        )
                        gr.Markdown("**Support**：JPG/PNG")
                        reset_btn = gr.Button("🔄 Reset", size="sm")

        inputs = [msg_input, chat_state] + ([image_upload] if enable_vision else [])
        
        def validate_inputs(message, state, *args):
            if enable_vision and not args[0]:  
                raise gr.Error("Please upload an image first before sending the message!")
            return message
        
        send_btn.click(
        validate_inputs,
        inputs=inputs,
        outputs=[msg_input],
        ).success(
            generate_response_fn,
            inputs=inputs,
            outputs=[msg_input, chatbot, chat_state]
        )
        clear_btn.click(
            lambda: ([], {"messages": [], "cache": None}),
            outputs=[chatbot, chat_state]
        )
        
        if enable_vision:
            reset_btn.click(
                lambda: ([], {"messages": [], "cache": None}, None),
                outputs=[chatbot, chat_state, image_upload]
            )
            image_upload.clear(
                lambda: ([], {"messages": [], "cache": None}, None),
                outputs=[chatbot, chat_state, image_upload]
    )

    return interface


def interact_with_gui(args):
    """
    """
    model,tokenizer,image_processor = load_model_with_processor(args.model_path)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)

    enable_vision = False
    if model.is_vl_model:
        enable_vision = True

    def generate_response(user_input,
                          chat_state,
                          image_path=None):
        """Handle both text and image inputs with streaming response."""
        image = None
        if model.is_vl_model and image_path:
            image = Image.open(image_path)
                
        inputs = [user_input]
        if model.is_chat_model:
            inputs = [[{"role":"user", "content":user_input}]]
        else:
            chat_state["cache"] = None
        partial_response = ""
        for search_states in stream_generate(model=model,
                                             inputs=inputs,
                                             tokenizer=tokenizer,
                                             device=device,
                                             image_processor=image_processor,
                                             images=[image] if chat_state["cache"] is None else None,
                                             cache=chat_state["cache"]):
            partial_response = search_states["text_buffer"][0]
            if partial_response.endswith("�"):
                continue

            partial_response = re.sub("(^<[a-zA-Z]*>)|(</[a-zA-Z]*>$)", "", partial_response)
            
            updated_messages = chat_state["messages"].copy()
            updated_entry = (user_input, partial_response)

            updated_messages.append(updated_entry)

            new_state = {
                "messages": updated_messages,
                "cache": search_states["cache"]
            }
            yield "", new_state["messages"], new_state
        
        final_entry = (user_input, partial_response)
        final_messages = chat_state["messages"].copy()
        final_messages.append(final_entry)
        new_state = {
            "messages": final_messages,
            "cache": search_states["cache"]
        }
        yield "", new_state["messages"], new_state

    app = create_gradio_app(generate_response, enable_vision)
    app.launch()


def interact(args):
    """
    """
    if args.enable_gui:
        interact_with_gui(args)
    else:
        interact_with_cli(args)


def ppl(args):
    """
    Computes perplexity scores for text files to evaluate language model performance.

    Processes all files in specified directory, calculating word-level perplexity
    using cross-entropy loss. Supports text truncation for efficient batch processing.

    Args:
        args: Configuration object containing:
            - model_path (str): Directory containing model artifacts
            - ppl_data_path (str): Directory with text files for evaluation
            - ppl_eval_len (int): Maximum character length for text truncation
            - ppl_batch_size (int): Batch size for evaluation
    """
    model = load_model(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    model = model.to(device)
    sum_n_tokens, sum_nll = 0, 0

    for f in os.listdir(args.ppl_data_path):
        dataset = DistributedStreamingJSONLDataset(
            os.path.join(args.ppl_data_path, f),
            args.ppl_batch_size,
            args.ppl_eval_len,
            model.pad_id,
            args.buffer_size,
            args.disable_gzip_compression,
            loop_once=True,
            raw_image_path=args.raw_image_path,
            model_path=args.model_path,
            world_size=1,
            rank=0)

        dataloader = DataLoader(
            dataset,
            batch_size=args.ppl_batch_size,
            collate_fn=lambda x:collate_fn(
                x, args.stage, model.pad_id, args.max_len, False, model.precision),
            num_workers=args.n_dataloader_workers,
            pin_memory=True,
            prefetch_factor=3,
            drop_last=False
        )

        n_tokens, nll = 0, 0
        for batch_data in dataloader:

            model.eval()
            with torch.no_grad():
                inputs = {k:batch_data[k].to(device) for k in batch_data}
                logits = model(**inputs)["logits"]
                n_tokens += inputs["targets"].ne(model.pad_id).sum().item()
                nll += F.cross_entropy(
                        logits.flatten(0, 1),
                        inputs["targets"].flatten(),
                        reduction="sum",
                        ignore_index=model.pad_id).item()

        loss = nll/n_tokens
        ppl = np.exp(loss)
        local_rank0_log(f"{f}: num_tokens: {n_tokens}, loss: {loss:.2f}, ppl: {ppl:.2f}")
        sum_n_tokens += n_tokens
        sum_nll += nll

    loss = sum_nll/sum_n_tokens
    ppl = np.exp(loss)
    local_rank0_log(f"num_tokens: {sum_n_tokens}, loss: {loss:.2f}, ppl: {ppl:.2f}")


def get_data_from_dict(data_dict, keys, mode="first", connector=''):
    """
    Retrieve data from a dictionary based on a list of keys with two modes.
    Supports a special case where keys=None returns the original dictionary.

    Args:
        data_dict (dict): The source dictionary to extract data from.
        keys (list | None): List of keys to check, or None to return the entire dictionary.
        mode (str): Operation mode: 'first' (return first found value) or 'join' (join valid string values).
        connector (str, optional): String to concatenate values in 'join' mode. Defaults to empty string.

    Returns:
        any: Depending on input:
            - If keys=None: Returns data_dict directly.
            - In 'first' mode: Value of the first existing key, or None.
            - In 'join' mode: String of valid values joined by connector, or empty string.
    """

    # Special case: Return entire dictionary if keys is None
    if keys is None:
        return data_dict

    def get_value(key):
        """Helper function to get value from a key (either direct or nested path)."""
        # Check as a direct key first
        if key in data_dict:
            return data_dict[key]
        # If not found, try nested path
        path_parts = [part.strip() for part in key.split(',')]
        current = data_dict
        for part in path_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    # Mode 1: Return the value of the first existing key in the list
    if mode == 'first':
        for key in keys:
            value = get_value(key)
            if value is not None:
                return value
        return None
    # Mode 2: Join all valid string values of existing keys using the connector
    elif mode == 'join':
        valid_values = []
        for key in keys:
            value = get_value(key)
            if isinstance(value, str):
                valid_values.append(value)
        return connector.join(valid_values)
    else:
        raise ValueError("Invalid mode. Use 'first' or 'join'.")


def normalize_dialog_format(
        dialog,
        system_field_keys=None,
        user_field_keys=None,
        assistant_field_keys=None):
    """
    Normalizes various conversation formats into a standardized list format.

    Processes 3 input types:
    1. String input: Returns None (unsupported type)
    2. List input: Converts between different role-content formats
    3. Dict input: Extracts values from different key naming conventions

    Args:
        dialog: Input dialog data (str/list/dict)
        system_field_keys: Custom keys for system messages in dict inputs
        user_field_keys: Custom keys for user messages in dict inputs
        assistant_field_keys: Custom keys for assistant messages in dict inputs

    Returns:
        list: Standardized dialog format with role/content pairs, or
        None: For invalid/mismatched input formats

    Standard Format:
        [{"role": "system", "content": ...},
         {"role": "user", "content": ...},
         {"role": "assistant", "content": ...}]
    """
    if isinstance(dialog, str):
        return None
    elif isinstance(dialog, list):
        formated = [] 
        role_mapping = {
                "human": "user", 
                "gpt": "assistant", 
                "system": "system",
                "user": "user",
                "assistant": "assistant"
                }
        for data in dialog:
            if "role" in data and "content" in data:
                formated.append(data)
            elif "from" in data and "value" in data:
                formated.append({"role":role_mapping[data["from"]], "content":data["value"]})
            else:
                return None

        return formated
    elif isinstance(dialog, dict):
        formated = [
                {"role": "system", "content":""},
                {"role": "user", "content":""},
                {"role": "assistant", "content":""}
                ]

        field_keys = {
                "system": system_field_keys or ["system"],
                "user": user_field_keys or ["instruction", "instruct", "input"],
                "assistant": assistant_field_keys or ["output"],
                }

        for data in formated:
            data["content"] = get_data_from_dict(dialog, field_keys[data["role"]], model="join")
            if data["role"] in ["user", "assistant"] and len(data["content"]) == 0:
                return None

        if len(formated[0]["content"]) == 0:
            formated = formated[1:]
        return formated

    return None


def convert_train_data(data,
                       stage,
                       tokenizer,
                       max_len=1024,
                       text_fields=["title", "text", "content"],
                       conversation_fields=["messages", "conversations"],
                       image_path_fields=None,
                       ):
    """
    Convert raw data dict into training-ready format based on training stage.

    Args:
        data (Dict): Raw input data dictionary
        stage (str): Training stage type: "pretrain", "sft", "dpo"
        tokenizer (Tokenizer): Text tokenizer instance
        max_len (int): Maximum sequence length (default: 1024)
        text_fields (List[str]): Field names containing text data (default: ["title", "text", "content"])
        conversation_fields (List[str]): Fields containing conversation data (default: ["messages", "conversations"])
        image_path_fields (List[str]): Field name for image paths (default: None)

    Returns:
        Optional[Dict]: Processed data dictionary with structure:
            - pretrain: {"x": List[int]}
            - sft: {"x": List[int], "targets": List[int]}
            - dpo: {"x_chosen": List[int], "targets_chosen": List[int],
                       "x_reject": List[int], "targets_reject": List[int]}
            With optional image fields if configured
    """
    converted_data = None

    if stage == "pretrain":
        text = get_data_from_dict(data, text_fields, mode="join", connector='\n')

        ids = tokenizer.encode(text, bos=True, eos=True)

        converted_data = {"x": ids}

    elif stage == "sft":

        messages = get_data_from_dict(data, conversation_fields)

        messages = normalize_dialog_format(messages)

        if messages is None:
            local_rank0_log("conversation data process failed", level=logging.WARNING)
            return None

        x,targets = tokenizer.encode_dialog_and_targets(messages)

        converted_data = {"x": x, "targets": targets}

    elif stage == "dpo":

        chosen = get_data_from_dict(data, keys=["chosen"])
        reject = get_data_from_dict(data, keys=["reject", "rejected"])

        chosen = get_data_from_dict(chosen, conversation_fields)
        reject = get_data_from_dict(reject, conversation_fields)

        chosen = normalize_dialog_format(chosen)
        reject = normalize_dialog_format(reject)

        if chosen is None or reject is None:
            local_rank0_log("conversation data process failed", level=logging.WARNING)
            return None

        x,targets = tokenizer.encode_dialog_and_targets(chosen)
        x2,targets2 = tokenizer.encode_dialog_and_targets(reject)
        converted_data = {"x_chosen": x, "targets_chosen": targets, "x_reject": x2, "targets_reject": targets2}

    if converted_data and image_path_fields:
        for image_path_field in image_path_fields:
            if image_path_field in data:
                converted_data["image_path"] = data[image_path_field]
                break

    return converted_data


def preprocess_one_file(input_path,
                        output_path,
                        stage,
                        model_path,
                        max_len,
                        auto_concat,
                        system_message,
                        text_fields,
                        conversation_fields,
                        image_path_fields,
                        disable_gzip_compression):
    """
    Processes and shuffles training data in a single file using stage-specific conversion rules.

    Args:
        input_path (str): Path to JSONL data file
        output_path (str): Path to processed JSONL data file
        stage (str): Processing stage - "pretrain"/"sft"/"dpo"
        model_path: Path to the tokenizer/image preprocesor config for text conversion/image preprocess
        max_len (int): Max sequence length for concatenation
        auto_concat (bool): Enable automatic sequence concatenation
        system_message (str): System prompt for dialog formatting
        text_fields (List[str]): Keys for pretraining text in data entries
        conversation_fields (List[str]): Keys for chat data in "sft"/"rm" modes
        image_path_fields (List[str]): Key for path to image data
        disable_gzip_compression: If True, writes uncompressed .jsonl;
                                  If False, writes .jsonl.gz with GZIP compression
    """
    tokenizer = load_tokenizer(model_path)

    cache = None
    if auto_concat:
        if stage == "pretrain":
            cache = {"x": []}
        elif stage == "sft":
            cache = {"x": [], "targets": []}

    system_message_tokens = []
    if system_message:
        system_message_tokens = tokenizer.encode_message({"role": "system", "content":system_message})

    alldata = [data for data in read_data_shards(input_path)]
    random.shuffle(alldata)

    processed_list = []
    for idx,data in enumerate(alldata):
        if idx > 0 and idx % 10000 == 0:
            logger.info(f"Processed {input_path} {idx} lines, total: {len(alldata)}")

        converted_data = convert_train_data(data,
                                            stage,
                                            tokenizer,
                                            max_len=max_len,
                                            text_fields=text_fields,
                                            conversation_fields=conversation_fields,
                                            image_path_fields=image_path_fields)
        if not converted_data:
            continue

        if auto_concat and stage == "pretrain":
            ids = cache["x"] + converted_data["x"]
            while len(ids) > max_len:
                processed_list.append({"x":ids[:max_len+1]})
                ids = ids[max_len:]
            cache["x"] = ids
        elif auto_concat and stage == "sft":
            if len(cache["x"]) + len(converted_data["x"]) > max_len - len(system_message_tokens):
                cache["x"] = cache["x"][:1] + system_message_tokens + cache["x"][1:]
                cache["targets"] = [tokenizer.pad_id] * len(system_message_tokens) + cache["targets"]
                processed_list.append(cache)
                cache = {"x": converted_data["x"], "targets": converted_data["targets"]}
            else:
                cache["x"] = cache["x"] + converted_data["x"]
                cache["targets"] = cache["targets"] + converted_data["targets"]
        else:
            processed_list.append(converted_data)

    random.shuffle(processed_list)

    logger.info(f"{output_path} size:{len(processed_list)}")
    logger.info(f"Writing to {output_path}")

    if disable_gzip_compression:
        with open(output_path, "w", encoding="utf-8") as f:
            for data in processed_list:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
    else:
        with gzip.open(output_path, "wt", compresslevel=1) as f:
            for data in processed_list:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    logger.info(f"Process {output_path} done.")

    return len(processed_list)


def split_data(
        raw_data_path,
        processed_data_path_list,
        disable_gzip_compression,
        sample_ratio,
        keep_fields,
        discard_fields,
        worker_id,
        n_workers,
        image_data_path_list,
        image_fields):
    """
    Split and process raw data into multiple output files with parallel processing support.

    Distributes data processing across multiple workers, allows field filtering, and supports
    optional gzip compression. Output is rotated between multiple files in processed_data_path_list.

    Args:
        raw_data_path: Path to the input raw data file(s)
        processed_data_path_list: List of output file paths to write processed data
        disable_gzip_compression: If True, disable gzip compression for output files
        sample_ratio: Ratio of input data to process (0.0-1.0)
        keep_fields: Specific fields to preserve in output (mutually exclusive with discard_fields)
        discard_fields: Specific fields to remove from output (mutually exclusive with keep_fields)
        worker_id: Identifier for current worker process (0 <= worker_id < n_workers)
        n_workers: Total number of parallel worker processes
    """
    if disable_gzip_compression:
        fo_list = [open(f, "w", encoding="utf-8") for f in processed_data_path_list]
    else:
        fo_list = [gzip.open(f, "wt", compresslevel=1) for f in processed_data_path_list]
    if image_fields:
        image_fo_list = [zipfile.ZipFile(f, "w") for f in image_data_path_list]

    idx = 0
    local_rank0_log(f"Split {raw_data_path}.")
    for i,data in enumerate(read_data_shards(raw_data_path, sample_ratio=sample_ratio)):
        if (i + 1) % 1000 == 0:
            local_rank0_log(f"worker-{worker_id} split {i+1} lines.", level=logging.DEBUG)
        if i % n_workers != worker_id:
            continue
        
        image_path = None
        if image_fields:
            image_data = get_data_from_dict(data, image_fields)
            if image_data:
                image_path = data["seq_id"]
                image_fo_list[idx].writestr(image_path, image_data)   

        if discard_fields:
            for key in discard_fields:
                if key in data:
                    del data[key]

        pruned_data = data
        if image_path:
            pruned_data["image_path"] = image_path

        if keep_fields:
            pruned_data = {}
            for k in keep_fields:
                if k in data:
                    pruned_data[k] = data[k]
        
        out = (json.dumps(pruned_data, ensure_ascii=False) + "\n")
        fo_list[idx].write(out)

        idx = (idx + 1) % len(fo_list)

    for fo in fo_list:
        fo.close()

    if image_fields:
        for fo in image_fo_list:
            fo.close()


def preprocess(args):
    """
    Executes distributed preprocessing pipeline for training data preparation.

    Args:
        args: Configuration object requiring:
            - model_path (str): Source directory for tokenizer/config
            - raw_data_path (list[str]): Input directory with .jsonl files
            - processed_data_path (str): Output directory for shards
            - n_split_shards (int): Number of output file shards
            - n_preprocess_workers (int): Parallel worker processes
            - stage (str): Data processing stage (see convert_train_data)
            - max_len (int): Sequence length limit
            - auto_concat (bool): Text concatenation flag
            - disable_gzip_compression: If True, writes uncompressed .jsonl;
                                        If False, writes .jsonl.gz with GZIP compression (default);
            - sample_ratio: Controls the ratio of random sampling from the original text
    """
    start = time.time()

    if not os.path.exists(args.processed_data_path):
        os.makedirs(args.processed_data_path, exist_ok=True)
    
    default_image_path_key = "image_path"
    if args.image_fields:
        args.image_path_fields = [default_image_path_key]

    mp.set_start_method('spawn')

    processed_data_path_list = [
            os.path.join(args.processed_data_path,
                f"{args.task_name}-{args.stage}-{i:05d}-of-{args.n_split_shards:05d}.jsonl")
            for i in range(args.n_split_shards)
    ]
    if not args.disable_gzip_compression:
        processed_data_path_list = [f"{path}.gz" for path in processed_data_path_list]

    image_data_path_list = None
    if args.image_fields:
        image_data_path_list = [
                os.path.join(args.processed_data_path,
                    f"{args.task_name}-{args.stage}-{i:05d}-of-{args.n_split_shards:05d}.zip")
                for i in range(args.n_split_shards)
            ]

    keep_fields = []
    if args.stage == "pretrain":
        keep_fields += args.text_fields
    elif args.stage == "sft":
        keep_fields += args.conversation_fields
    elif args.stage == "dpo":
        keep_fields += ["chosen", "reject", "rejected"]
    if args.image_path_fields:
        keep_fields += args.image_path_fields

    discard_fields = []
    if args.image_fields:
        discard_fields += args.image_fields
        keep_fields += [default_image_path_key]

    n_preprocess_workers = min(len(processed_data_path_list), args.n_preprocess_workers)
    process_args = []

    for i in range(n_preprocess_workers):
        process_args.append(
                (args.raw_data_path,
                 processed_data_path_list[i::n_preprocess_workers],
                 args.disable_gzip_compression,
                 args.sample_ratio,
                 keep_fields,
                 discard_fields,
                 i,
                 n_preprocess_workers,
                 image_data_path_list[i::n_preprocess_workers] if image_data_path_list else None,
                 args.image_fields
                 )
                )

    with mp.Pool(n_preprocess_workers) as pool:
        results = pool.starmap(split_data, process_args)

    cost = time.time() - start
    local_rank0_log(f"Split done, cost {cost:.2f}s. Start preprocess now.")

    input_path_list = processed_data_path_list
    output_path_list = processed_data_path_list

    process_args = []
    for input_path, output_path in zip(input_path_list, output_path_list):
        process_args.append(
                (input_path,
                 output_path,
                 args.stage,
                 args.model_path,
                 args.max_len,
                 args.auto_concat,
                 args.system_message_in_preprocess,
                 args.text_fields,
                 args.conversation_fields,
                 args.image_path_fields,
                 args.disable_gzip_compression
                 )
                )

    n_preprocess_workers = min(len(process_args), args.n_preprocess_workers)
    with mp.Pool(n_preprocess_workers) as pool:
        results = pool.starmap(preprocess_one_file, process_args)
        total_samples = sum(results)

    cost = time.time() - start
    local_rank0_log(f"Preprocess done, total {total_samples} samples, total cost {cost:.2f}s")


def collect_file_paths(path_list, extensions=None):
    """
    Collects all file paths from the given list of paths, which can include directories,
    individual files, or glob patterns (matching files only).

    Args:
        path_list (list): List of paths (directories, files, or glob patterns).
        extensions (list, optional): List of file extensions to include
                                     (e.g., ['.txt', '.jsonl']).
                                     If None, all files are included.
    Returns:
        list: A list of all file paths collected from the input paths.
    """
    if isinstance(path_list, str):
        path_list = [path_list]

    result = []
    for path in path_list:
        if glob.has_magic(path):
            # Expand glob pattern and filter out directories
            matched_paths = glob.glob(path)
            for matched_path in matched_paths:
                if os.path.isfile(matched_path):
                    abs_path = os.path.abspath(matched_path)
                    file_ext = os.path.splitext(abs_path)[1]
                    if extensions is None or file_ext in extensions:
                        result.append(abs_path)
        else:
            # Handle individual directory or file path
            if os.path.isdir(path):
                # Traverse directory and subdirectories for all files
                for root, _, files in os.walk(path):
                    for f in files:
                        file_path = os.path.join(root, f)
                        abs_path = os.path.abspath(file_path)
                        file_ext = os.path.splitext(abs_path)[1]
                        if extensions is None or file_ext in extensions:
                            result.append(abs_path)
            elif os.path.isfile(path):
                # Add the file directly
                abs_path = os.path.abspath(path)
                file_ext = os.path.splitext(abs_path)[1]
                if extensions is None or file_ext in extensions:
                    result.append(abs_path)
            # Note: Invalid paths (non-existent) are skipped as per problem assumptions


    return result


def read_data_shards(data_paths, sample_ratio=None, max_sample_size=None):
    """
    Iteratively reads data samples from multiple files (JSONL/CSV/Parquet/JSON/etc).

    Processes a list of file paths and/or directories, recursively reading all supported
    data files. When sampling constraints are applied, each file's maximum sample count
    is proportionally limited based on total file count.

    Args:
        data_paths (Union[str, List[str]]): Path(s) to data files/directories.
            Accepts either a single path string or list of paths. All directories are
            recursively scanned for supported data file extensions.
        sample_ratio (float, optional): Ratio of samples to retain from each file.
        max_sample_size (int, optional): Global maximum for total samples across all files.
            Individual file limits are calculated as max_sample_size // total_file_count.

    Yields:
        dict: Individual data samples as dictionaries.
    """

    def check_encoding(file_path):
        """
        Performs a deterministic check for UTF-8 encoding variants in file headers.
        """
        with open(file_path, 'rb') as file:
            start = file.read(3)
            if start == b'\xef\xbb\xbf':
                return "UTF-8-SIG"
            else:
                return "UTF-8"

    if isinstance(data_paths, str):
        data_paths = [data_paths]

    files = collect_file_paths(data_paths)
    files.sort()

    cnt = 0
    seq_id = -1
    for f in files:
        if f.endswith(".jsonl.gz"):
            data_iter = gzip.open(f, "rt")
        elif f.endswith(".jsonl"):
            data_iter = open(f, "r", encoding="utf-8")
        elif f.endswith(".csv"):
            with open(f, "r", encoding=check_encoding(f)) as fi:
                data_iter = [line for line in csv.DictReader(fi)]
        elif f.endswith(".json"):
            with open(f, "r", encoding="utf-8") as fi:
                data_iter = json.load(fi)
        elif f.endswith(".parquet"):
            data_iter = pq.read_table(f).to_pylist()
        elif f.endswith(".txt"):
            with open(f, "r", encoding=check_encoding(f)) as fi:
                data_iter = [
                        {"text": text.strip()}
                        for text in re.split("<|endoftext|>", fi.read())
                        if len(text) > 0
                        ]
        else:
            local_rank0_log(f"Skip: {f}")
            continue

        for i,data in enumerate(data_iter):
            if f.endswith(".jsonl") or f.endswith(".jsonl.gz"):
                data = json.loads(data)
            seq_id += 1
            if sample_ratio and random.random() > sample_ratio:
                continue
            data["seq_id"] = str(seq_id)
            yield data
            cnt += 1
            if max_sample_size and cnt >= max_sample_size:
                break

        if max_sample_size and cnt >= max_sample_size:
            break


class CompressedImages():
    """
    """
    def __init__(self, image_data_path_list):
        """
        """
        self.path_or_name2file = {}
        self.basename2path = {}
        for f in collect_file_paths(image_data_path_list):
            if f.endswith('.zip'):
                zf = zipfile.ZipFile(f, 'r')
                for name in zf.namelist():
                    self.path_or_name2file[name] = zf
                    f = os.path.basename(name)
                    self.basename2path[f] = name


    def get_image(self, path_or_name):
        """
        """
        if path_or_name in self.path_or_name2file:
            image_data = self.path_or_name2file[path_or_name].read(path_or_name)
            image = Image.open(io.BytesIO(image_data))
            return image

        basename = os.path.basename(path_or_name)
        if basename in self.basename2path:
            path_name = self.basename2path[basename]
            image_data = self.path_or_name2file[path_name].read(path_name)                            
            image = Image.open(io.BytesIO(image_data))                                                      
            return image 

        return None


class DistributedStreamingJSONLDataset(IterableDataset):
    """
    A memory-efficient streaming dataset for distributed training
    loading tokenized text-target pairs from sharded JSONL/gz files
    """
    def __init__(self,
            data_path,
            batch_size,
            max_len,
            pad_id,
            buffer_size,
            disable_gzip_compression,
            loop_once=False,
            raw_image_path=None,
            model_path=None,
            world_size=1,
            rank=0
            ):
        """
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_len = max_len
        self.pad_id = pad_id
        self.buffer_size = 8 * batch_size
        if buffer_size:
            self.buffer_size = buffer_size
        self.loop_once = loop_once
        
        extensions = [".gz"]
        if disable_gzip_compression:
            extensions = [".jsonl"]
        self.files = collect_file_paths(data_path, extensions)

        self.world_size = world_size
        self.rank = rank

        self.raw_image_path = raw_image_path
        self.model_path = model_path
        self.compressed_images = {}
        self.image_processor = {}


    def check_valid(self, data):
        """
        """
        for k in data:
            if "targets" in k and all(x==self.pad_id for x in data[k][:self.max_len]):
                return False

        return True


    def __iter__(self):
        """
        """
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        total_workers = world_size * num_workers
        global_worker_id = rank * num_workers + worker_id

        if self.raw_image_path and global_worker_id not in self.compressed_images:
            self.image_processor = load_image_processor(self.model_path)
            self.compressed_images[global_worker_id] = CompressedImages(self.raw_image_path)

        idx = -1
        batch_data = []
        while True:
            for data in read_data_shards(self.files):
                idx += 1
                if idx % total_workers == global_worker_id:
                    if not self.check_valid(data):
                        continue
                    if "image_path" in data:
                        try:
                            image = self.compressed_images[global_worker_id].get_image(data["image_path"])
                            data["x_vision"] = self.image_processor(image)
                        except:
                            logger.warning(f'convert {data["image_path"]} failed')
                            traceback.print_exc()
                            continue
                    batch_data.append(data)
                    if len(batch_data) % self.buffer_size == 0:
                        random.shuffle(batch_data)
                        for data in batch_data[:self.batch_size]:
                            yield data
                        batch_data = batch_data[self.batch_size:]
            if self.loop_once:
                break

        for data in batch_data:
            yield data


class LRScheduler():
    """
    Implements hybrid learning rate scheduling with warmup and decay strategies
    """
    def __init__(self, optimizer, args):
        """
        Initializes scheduler with control parameters

        Args:
            optimizer (torch.optim): Optimizer to schedule
            args (Namespace): Configuration containing:
                - lr: Maximum learning rate (peak after warmup)
                - total_steps: Total training steps
                - warmup_steps: Linear warmup duration
                - lr_scheduler: Decay strategy ['cosine', 'linear']
        """
        self.optimizer = optimizer
        self.steps = 0
        self.max_lr = args.lr
        self.total_steps = args.total_steps
        self.warmup_steps = args.warmup_steps
        self.lr_scheduler = args.lr_scheduler


    def update_lr(self, steps):
        """
        Computes current learning rate using hybrid warmup-decay schedule

        Mathematical Formulations:
            - Warmup: lr = max_lr * (steps / warmup_steps)
            - Cosine: 0.5 * max_lr * (1 + cos(π * progress))
            - Linear: max_lr * (1 - progress)

        Returns:
            float: Current learning rate
        """
        if steps < self.warmup_steps:
            return self.max_lr / self.warmup_steps * steps
        elif self.lr_scheduler == "cosine":
            ratio = (1 + math.cos((steps-self.warmup_steps)/(self.total_steps-self.warmup_steps)*math.pi)) / 2
            return self.max_lr * ratio
        elif self.lr_scheduler == "linear":
            ratio = 1 - (steps-self.warmup_steps)/(self.total_steps-self.warmup_steps)
            return self.max_lr * ratio
        return self.lr

    def step(self):
        """
        Executes learning rate update.
        """
        self.steps += 1
        self.lr = self.update_lr(self.steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr


def show_model_info(model):
    """
    Logs model architecture and parameter statistics to rank0 process

    Args:
        model (nn.Module): PyTorch model to analyze
    """
    local_rank0_log(f"{model}")
    total_params = sum(p.numel() for p in model.parameters())
    local_rank0_log(f"Total Model Params:{total_params}")


def get_log_probs(logits, targets, pad_id):
    """
    Computes masked log probabilities for sequence

    Args:
        logits (Tensor): [batch_size, seq_len, vocab_size]
        targets (Tensor): [batch_size, seq_len]
        pad_id (int): Padding token identifier

    Returns:
        tuple: (avg_log_probs, sum_log_probs)
            - avg_log_probs: Per-sequence mean log prob (excluding padding)
            - sum_log_probs: Total log prob

    """
    log_probs = torch.gather(F.log_softmax(logits, dim=-1), -1, targets.unsqueeze(-1))
    log_probs = log_probs * (targets.unsqueeze(-1).ne(pad_id))

    sum_log_probs = log_probs.squeeze(-1).sum(-1)

    avg_log_probs = sum_log_probs / (targets.ne(pad_id).sum(-1))

    return avg_log_probs, sum_log_probs
 

@Registry.register("loss.pretrain")
@Registry.register("loss.sft")
def compute_model_loss(model, inputs, ref_model, args):
    """
    """
    return model(**inputs)


@Registry.register("loss.dpo")
def compute_dpo_loss(model, inputs, ref_model, args):
    """ 
    Implements DPO preference optimization algorithm.

    Args:
        model (LM): Main language model to optimize
        inputs (dict): Contains:
            - x_chosen: Token IDs for preferred responses
            - targets_chosen: Target masks for preferred responses
            - x_reject: Token IDs for rejected responses
            - targets_reject: Target masks for rejected responses
        ref_model (LM): Reference model for RL methods (DPO/DPO variants)
        args:
            - beta (float): KL divergence regularization strength
            - use_sft_loss (bool): Enable SFT gradient mixing for some methods
            - dpo_lambda (float): Weighting factor for SFT loss component

    Returns:
        dict: Contains:
            - loss (Tensor): Computed loss value for backward()
            - chosen_rewards (float): Mean reward of preferred responses
            - rejected_rewards (float): Mean reward of rejected responses
            - reward_accuracies (float): % of samples where chosen > rejected
    """
    chosen_inputs = {"x": inputs["x_chosen"], "use_checkpoint": inputs.get("use_checkpoint", False)}
    reject_inputs = {"x": inputs["x_reject"], "use_checkpoint": inputs.get("use_checkpoint", False)}

    chosen_outputs = model(**chosen_inputs)
    reject_outputs = model(**reject_inputs)
    chosen_log_probs = get_log_probs(chosen_outputs["logits"], inputs["targets_chosen"], model.pad_id)
    reject_log_probs = get_log_probs(reject_outputs["logits"], inputs["targets_reject"], model.pad_id)

    with torch.no_grad():
        chosen_inputs = {"x": inputs["x_chosen"], "use_checkpoint": False}
        reject_inputs = {"x": inputs["x_reject"], "use_checkpoint": False} 
        ref_chosen_outputs = ref_model(**chosen_inputs)
        ref_reject_outputs = ref_model(**reject_inputs)
        ref_chosen_log_probs = get_log_probs(ref_chosen_outputs["logits"], inputs["targets_chosen"], model.pad_id)
        ref_reject_log_probs = get_log_probs(ref_reject_outputs["logits"], inputs["targets_reject"], model.pad_id)

    logits = chosen_log_probs[1] - ref_chosen_log_probs[1] - reject_log_probs[1] + ref_reject_log_probs[1]

    loss = -F.logsigmoid(args.dpo_beta * logits).mean()

    if args.use_sft_loss:
        loss = -chosen_log_probs[0].mean() + args.dpo_lambda * loss

    chosen_rewards = (chosen_log_probs[1] - ref_chosen_log_probs[1]).detach()
    rejected_rewards = (reject_log_probs[1] - ref_reject_log_probs[1]).detach()

    reward_accuracies = (chosen_rewards > rejected_rewards).float().mean()                                 
    chosen_rewards = chosen_rewards.mean()                                                                 
    rejected_rewards = rejected_rewards.mean()   
 
    outputs = {
            "loss": loss,
            "chosen_rewards": chosen_rewards.item(),
            "rejected_rewards": rejected_rewards.item(),
            "reward_accuracies": reward_accuracies.item(),
    }

    return outputs


def get_default_ds_config(args):
    """
    Generates a DeepSpeed configuration dictionary for distributed training setup.

    Args:
        args: Configuration object containing training hyperparameters:
            - lr (float): Learning rate for optimizer
            - weight_decay (float): Weight decay coefficient
            - lr_scheduler (str): Learning rate scheduler type ("linear" or others)
            - warmup_steps (int): Number of warmup steps for scheduler
            - total_steps (int): Total training steps
            - accumulate_steps (int): Gradient accumulation steps
            - batch_size (int): Global batch size

    Returns:
        dict: DeepSpeed configuration with the following structure:
            - Precision: Disables FP16, enables BF16 mixed-precision training
            - Optimizer: AdamW with specified learning rate and weight decay
            - Scheduler: Warmup-based LR scheduler (linear decay or cosine)
            - ZeRO Optimization: Stage 2 with communication optimizations
            - Training Params: Batch sizes, gradient accumulation, logging
    """
    ds_config = {
        "fp16": {
            "enabled": args.training_precision in ["fp16", "float16"]
        },
        "bf16": {
            "enabled": args.training_precision in ["bf16", "bfloat16"]
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": args.weight_decay
            }
        },
        "scheduler": {
              "type": "WarmupDecayLR",
              "params": {
                  "warmup_type": "linear",
                  "warmup_min_lr": 0,
                  "warmup_max_lr": args.lr,
                  "warmup_num_steps": args.warmup_steps,
                  "total_num_steps": args.total_steps
            }
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "none",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },

        "gradient_accumulation_steps": args.accumulate_steps,
        "steps_per_print": 100,
        "train_batch_size": args.batch_size*args.accumulate_steps,
        "train_micro_batch_size_per_gpu": args.batch_size//world_size
    }

    if args.lr_scheduler=="cosine":
        ds_config["scheduler"] = {
                "type": "WarmupCosineLR",
                "params": {
                    "warmup_type": "linear",
                    "warmup_num_steps": args.warmup_steps,
                    "total_num_steps": args.total_steps
                    }
                }

    local_rank0_log(f"deepspeed config: {ds_config}")
    return ds_config


def plot_log_loss(log_path, output_dir=None):
    """
    Generate loss curve from training log with auto-output path

    Args:
        log_path (str): Path to input .log file
        output_dir (str, optional): Custom output directory

    Example:
        Input:  log/train.log
        Output: log/train_loss.png
    """
    # Parse log file
    def extract_data(fpath):
        pattern = r'(\d+) steps.*loss: (\d+\.\d+)'
        data = []
        with open(fpath, 'r') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    data.append([int(match.group(1)), float(match.group(2))])
        if not data:
            raise ValueError(f"No loss data found in {fpath}")
        return sorted(data, key=lambda x: x[0])

    # Generate output path
    log_dir = os.path.dirname(os.path.abspath(log_path))
    base_name = os.path.splitext(os.path.basename(log_path))[0]
    output_fname = f"{base_name}_loss.png"

    output_path = os.path.join(
        output_dir if output_dir else log_dir,
        output_fname
    )

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Plot configuration
    data = extract_data(log_path)
    steps, losses = zip(*data)

    plt.figure(figsize=(12, 6))
    plt.plot(steps, losses, 'r-', linewidth=1.5)
    plt.title(f'Loss Curve: {base_name}', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.grid(alpha=0.3)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    local_rank0_log(f"Plot loss curve to: {os.path.abspath(output_path)}")


def pad_to_max_len(x, pad_id, max_len, dynamic_padding):
    """
    Standardize sequence length through truncation/padding.

    Args:
        x (list[list[int]]): Input token IDs sequence
        pad_id (int): ID of the padding token used to fill sequences.
        max_len (int): Target maximum length for sequences after padding/truncation.
        dynamic_padding (bool): Whether to dynamically adjust `max_len` to the minimum of
            the predefined `max_len` and the maximum sequence length in the current batch.

    Returns:
        list[int]: Padded/truncated sequence of exactly `max_len` length:
        - First `len(x)` elements: original tokens (if len(x) > max_len)
        - Remaining elements: pad_id tokens (if len(x) < max_len)

    """
    if dynamic_padding:
        max_len = min(max_len, max(len(xx) for xx in x))
        return [xx[:max_len] + [pad_id] * (max_len - len(xx)) for xx in x]
    return [xx[:max_len] + [pad_id] * (max_len - len(xx)) for xx in x]


def collate_fn(batch_data, stage, pad_id, max_len, dynamic_padding, precision):
    """
    """
    if stage == "pretrain":
        for data in batch_data:
            data["targets"] = data["x"][1:]
            data["x"] = data["x"][:-1]

    concat = {}
    for k in batch_data[0]:
        if k in ["x_vision"]:
            concat[k] = torch.stack([x[k] for x in batch_data])
            if precision in ["bf16", "bfloat16"]:
                concat[k] = concat[k].bfloat16()
            if precision in ["fp16", "float16"]:
                concat[k] = concat[k].half()
        elif k in ["x_vision_mask"]:
            dtype = torch.bool
            concat[k] = torch.tensor([x[k] for x in batch_data], dtype=dtype)
        elif k in ["label"]:
            dtype = torch.long
            concat[k] = torch.tensor([x[k] for x in batch_data], dtype=dtype)
        elif k not in ["seq_id", "image_path"]:
            dtype = torch.long
            ids = pad_to_max_len([x[k] for x in batch_data], pad_id, max_len, dynamic_padding)
            concat[k] = torch.tensor(ids, dtype=dtype)
    return concat


def train(args):
    """
    Orchestrates distributed model training with advanced configuration support
    """  
    global local_rank,world_size,rank    
    if args.use_cuda:
        init_process_group(backend="nccl")    
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
    
    IMPORT_DEEPSPEED_SUCCESS = False
    try:
        import deepspeed
        IMPORT_DEEPSPEED_SUCCESS = True
    except:
        pass

    if not IMPORT_MATPLOTLIB_SUCCESS:
        local_rank0_log(
                        "Matplotlib not found. Loss plots are unavailable. "
                        "(Install with `pip install matplotlib` for visualization support)"
                        "Falling back to text-only output."
                        )

    logger.info(f"start local_rank: {local_rank} , rank: {rank}, world_size: {world_size}")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    model = load_model(args.model_path)
    ref_model = None
    if args.stage == "dpo":
        ref_model = load_model(args.model_path)
        
    if args.lora_rank:
        for name,parameter in model.named_parameters():
            if "lora" in name:
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False
            if "lora_B" in name:
                parameter.data.fill_(0)
            if "lora_A" in name:
                stdv = 1 / np.sqrt(parameter.shape[0])
                parameter.data.uniform_(-stdv, stdv)

    total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad==True)
    local_rank0_log(f"Trainable Model Params: {total_train_params}")

    device = "cpu"
    if args.use_cuda:
        device = f"cuda:{local_rank}"
        model = model.to(device)
        if args.stage == "dpo":
            ref_model = ref_model.to(device)

    # batch_size must be divisible by world_size to ensure equal splits across processes
    args.batch_size = (args.batch_size // world_size) * world_size

    if args.total_steps is None:
        args.total_steps = args.epochs * args.total_size // args.batch_size // args.accumulate_steps + 1

    lr_scheduler = None
    if args.use_deepspeed and not IMPORT_DEEPSPEED_SUCCESS:
        local_rank0_log("IMPORT DEEPSPEED FAILED, USE NATIVE PYTORCH NOW!",
                level=logging.WARNING)
        args.use_deepspeed = False

    if args.use_deepspeed:
        ds_config = get_default_ds_config(args)
        model, optimizer, _, __ = deepspeed.initialize(
            model=model,
            model_parameters=filter(lambda p: p.requires_grad, model.parameters()),
            args=args,
            config=ds_config
        )
        local_rank0_log("deepspeed initialize done.")
    else:
        optimizer = optim.AdamW(
                    params=filter(lambda p: p.requires_grad, model.parameters()),
                    lr=args.lr,
                    weight_decay=args.weight_decay
                    )
        lr_scheduler = LRScheduler(optimizer, args)

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

    dataset = DistributedStreamingJSONLDataset(
            args.train_data_path,
            args.batch_size//world_size,
            args.max_len,
            model.pad_id,
            args.buffer_size,
            args.disable_gzip_compression,
            loop_once=False,
            raw_image_path=args.raw_image_path,
            model_path=args.model_path,
            world_size=world_size,
            rank=rank)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size//world_size,
        collate_fn=lambda x:collate_fn(
            x, args.stage, model.pad_id, args.max_len, False, args.training_precision),
        num_workers=args.n_dataloader_workers,
        pin_memory=True,
        prefetch_factor=2
    )
    
    loss_fn = Registry.get_required(f"loss.{args.stage}")

    smooth_loss = None
    history_loss = []

    dtype = torch.float
    if args.use_amp:
        if args.training_precision in ["bfloat16", "bf16"]:
            dtype = torch.bfloat16
        elif args.training_precision in ["float16", "fp16"]:
            dtype = torch.float16

    log_path = os.path.join(LOG_DIR, args.task_name + ".log")
    local_rank0_log("Train Start.")
    local_rank0_log(f"Total Trainging steps: {args.total_steps}")
    start = time.time()
    for batch_idx, batch_data in enumerate(dataloader):
        steps = batch_idx // args.accumulate_steps

        inputs = {k:batch_data[k].to(device) for k in batch_data}
        inputs["use_checkpoint"] = False
        if args.use_checkpoint:
            inputs["use_checkpoint"] = True

        if not args.use_deepspeed:
            if args.use_amp:
                with torch.cuda.amp.autocast(dtype=dtype):
                    outputs = loss_fn(model, inputs, ref_model, args)
                    loss = outputs["loss"]
            else:
                outputs = loss_fn(model, inputs, ref_model, args)
                loss = outputs["loss"]
        else:
            outputs = loss_fn(model, inputs, ref_model, args)
            loss = outputs["loss"]

        if args.logger_smooth_loss == "ema":
            if smooth_loss is None:
                smooth_loss = loss.item()
            else:
                smooth_loss = 0.6 * smooth_loss + 0.4 * loss.item()
        elif args.logger_smooth_loss == "ma":
            if smooth_loss is None:
                smooth_loss = loss.item()
                history_loss.append(loss.item())
            elif len(history_loss) >= 100:
                smooth_loss = smooth_loss + (loss.item() - history_loss[0]) / 100
                history_loss = history_loss[1:] + [loss.item()]
            else:
                smooth_loss = (smooth_loss*len(history_loss) + loss.item()) / (len(history_loss) + 1)
                history_loss.append(loss.item())

        if not args.use_deepspeed:
            loss = loss / args.accumulate_steps
            if args.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip
                    )
        else:
            model.backward(loss)
            model.step()


        if (batch_idx + 1) % args.accumulate_steps == 0:
            if not args.use_deepspeed:
                lr_scheduler.step()
                if args.use_amp:
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                    scaler.update()
                else:
                    optimizer.step()
                    optimizer.zero_grad()

            if rank == 0 and (steps + 1) % args.save_every_steps == 0:
                save_model(args.model_path, os.path.join(args.save_path, f"{steps+1}-steps"), model)
                if IMPORT_MATPLOTLIB_SUCCESS:
                    plot_log_loss(log_path)

            if (steps + 1) % args.print_every_steps == 0:
                cost = (time.time() - start) / (steps + 1)
                local_rank0_log(
                        f"{steps + 1} steps smooth_loss:{smooth_loss:.2f} loss: {loss.item():.2f} cost: {cost:.2f}s/step")
                if lr_scheduler:
                    local_rank0_log("learning rate: {lr_scheduler.lr:.6f}")

                if args.stage == "dpo":
                    cr,rr,rc = outputs["chosen_rewards"], outputs["rejected_rewards"], outputs["reward_accuracies"]
                    local_rank0_log(f"chosen_rewards: {cr:.2f}, rejected_rewards: {rr:.2f}, reward_accuracies: {rc:.2f}")

        if steps >= args.total_steps:
            break

    if rank == 0:
        save_model(args.model_path, args.save_path, model)
        if IMPORT_MATPLOTLIB_SUCCESS:
            plot_log_loss(log_path)

    cost = time.time() - start
    local_rank0_log(f"Train Done, total cost {cost:.2f}s.")

    destroy_process_group()

def main():
    """
    Main entry point for language model pipeline with multi-mode capabilities.

    Handles command dispatch for:
    - Data preprocessing
    - Model initialization
    - Training bpe tokenizer
    - Training workflows
    - Interactive inference
    - Evaluation metrics

    Args:
        Command-line arguments via argparse. Required parameters vary by mode.

    Returns:
        None: Writes outputs to files/stderr based on operation mode.

    """
    logo_str = get_logo(concat=True)
    description=(
            f"\n{logo_str}\n"
    )
    usage = (
    '\nTo perform simple inference via the command line, run "python mimixlm.py '
    '--model_path <model_path>".\n'
    "For preprocessing, training, and evaluation, please refer to the respective "
    "shell scripts."
    )
    parser = argparse.ArgumentParser(
            usage=usage,
            description=description,
            formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--mode",
                        type=str,
                        choices=["preprocess", "init", "train", "interact", "ppl", "train_bpe"],
                        default="interact",
                        help="specify in which running mode.")

    parser.add_argument('--stage',
                        type=str,
                        choices=["pretrain", "sft", "dpo"],
                        default="pretrain",
                        help="specify in which training stage, only use in train or preprocess mode.")

    parser.add_argument('--init_config_path',
                        type=str,
                        default=None,
                        help="config for model initialization, only use in init model stage")

    parser.add_argument('--init_vocab_size',
                        type=int,
                        default=None,
                        help="config for default model initialization, only use in init model stage")

    parser.add_argument('--init_model_path',
                        type=str,
                        default=None,
                        help="where to save model when in init model stage.")

    parser.add_argument('--init_weight_path',
                        type=str,
                        nargs='+',
                        default=None,
                        help="pretrained weight path list for model initialization")

    parser.add_argument('--initializer_range',
                        type=float,
                        default=0.02,
                        help="weight initialization range for model initialization")

    parser.add_argument('--model_size',
                        type=str,
                        choices=["small", "base", "large", "xl"],
                        default="base",
                        help="default config for model initialization when config is not specified")

    parser.add_argument('--task_name',
                        type=str,
                        default=datetime.datetime.today().strftime('run-%Y-%m-%d-%H-%M-%S'),
                        help="training log file name will be starts with task name in logger dir.")

    parser.add_argument('--use_cuda',
                        default=False,
                        action='store_true',
                        help="use cuda or not.")

    parser.add_argument('--epochs',
                        type=int,
                        default=1,
                        help="train how many epochs.(ignored if --total_steps is specified)")

    parser.add_argument('--total_steps',
                        type=int,
                        default=None,
                        help="train how many steps.")

    parser.add_argument('--save_every_steps',
                        type=int,
                        default=1000,
                        help="save model after how many steps from last save.")

    parser.add_argument('--print_every_steps',
                        type=int,
                        default=1,
                        help="print loss info how many steps from last print.")

    parser.add_argument('--logger_smooth_loss',
                        type=str,
                        choices=["ema", "ma"],
                        default="ema",
                        help="which variation to calculate printed loss when in the training mode.")

    parser.add_argument('--use_deepspeed',
                        default=False,
                        action='store_true',
                        help="use deepspeed or not."
                        )

    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help="learning rate, default: 1e-4.")

    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-2,
                        help="weight Decay, default: 1e-2.")

    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help="batch sized default: 16.")

    parser.add_argument('--warmup_steps',
                        type=int,
                        default=2000,
                        help="warmup steps in training.")

    parser.add_argument('--lr_scheduler',
                        type=str,
                        choices=["constant", "cosine", "linear"],
                        default="cosine",
                        help="learning rate scheduler, default: cosine.")

    parser.add_argument('--use_amp',
                        default=False,
                        action='store_true',
                        help="use amp or not.")

    parser.add_argument('--training_precision',
                        type=str,
                        choices=["float16", "bfloat16", "fp16", "bf16"],
                        default="bf16",
                        help="training precision, default: bf16.")

    parser.add_argument('--accumulate_steps',
                        type=int,
                        default=1,
                        help="gradient accumulate steps, default 1.")

    parser.add_argument('--grad_clip',
                        type=float,
                        default=1,
                        help="gradient clip, default 1.")

    parser.add_argument('--lora_rank',
                        type=int,
                        default=None,
                        help="LoRA rank (enables LoRA training when specified")

    parser.add_argument('--use_checkpoint', default=False, action='store_true')

    parser.add_argument('--dpo_beta',
                        type=float,
                        default=None,
                        help="DPO or SimPO or IPO beta.")

    parser.add_argument('--use_sft_loss',
                        action='store_true',
                        help="use sft loss in dpo stage.")

    parser.add_argument('--dpo_lambda',
                        type=float,
                        default=None,
                        help="use in dpo when conbine with sft loss")

    parser.add_argument('--raw_data_path',
                        type=str,
                        nargs='+',
                        default=None,
                        help="raw data dir list in preprocess.")

    parser.add_argument('--disable_gzip_compression',
                        default=False,
                        action='store_true',
                        help="disable gzip compression or not when save preprocessed data")

    parser.add_argument('--processed_data_path',
                        type=str,
                        default=None,
                        help="processed data dir in preprocess")

    parser.add_argument('--train_data_path',
                        type=str,
                        default=None,
                        help="train data dir in training")

    parser.add_argument('--buffer_size',
                        type=int,
                        default=None,
                        help="Buffer size for data reading. Default: 8*batch_size")

    parser.add_argument('--total_size',
                        type=int,
                        default=None,
                        help="number of samples in training dataset")

    parser.add_argument('--max_len',
                        type=int,
                        default=1024,
                        help="max length of sequence in training dataset")

    parser.add_argument('--n_dataloader_workers',
                        type=int,
                        default=4,
                        help="how many workers when preprocess data")

    parser.add_argument('--n_split_shards',
                        type=int,
                        default=None,
                        help="number of shards when dump processed data")

    parser.add_argument('--n_preprocess_workers',
                        type=int,
                        default=4,
                        help="how many workers when preprocess data")

    parser.add_argument('--sample_ratio',
                        type=float,
                        default=None,
                        help="controls the ratio of random sampling from the original data")

    parser.add_argument('--auto_concat',
                        default=False,
                        action='store_true',
                        help="auto concat samples to max length or not when preprocess data")

    parser.add_argument('--system_message_in_preprocess',
                        type=str,
                        default=None,
                        help="Prepend a system message to each dialogue during preprocessing.")

    parser.add_argument('--text_fields',
                        type=str,
                        nargs='+',
                        default=None,
                        help="keys of pretrain data")

    parser.add_argument('--conversation_fields',
                        type=str,
                        nargs='+',
                        default=None,
                        help="keys of conversation data")

    parser.add_argument('--raw_image_path',
                        type=str,
                        nargs='+',
                        default=None,
                        help="image data path")

    parser.add_argument('--image_path_fields',
                        type=str,
                        nargs='+',
                        default=None,
                        help="possible keys of path for cross-modal alignment if image is stored separately")

    parser.add_argument('--image_fields',
                        type=str,
                        nargs='+',
                        default=None,
                        help="possible keys of image data if original data stores images and text together")

    parser.add_argument('--model_path',
                        type=str,
                        default=None,
                        help="load model path")

    parser.add_argument('--ppl_eval_len',
                        type=int,
                        default=1024,
                        help="max length of text when evaluate ppl. Defalut 1024")

    parser.add_argument('--ppl_data_path',
                        type=str,
                        default=None,
                        help="data dir used in ppl evaluation")

    parser.add_argument('--ppl_batch_size',
                        type=int,
                        default=1,
                        help="batch size used in ppl evaluation")

    parser.add_argument('--save_path',
                        type=str,
                        default=None,
                        help="save model path"
                        )

    parser.add_argument('--max_train_bpe_lines',
                        type=int,
                        default=50000,
                        help="limit bpe training size to reduce memory usage"
                        )

    parser.add_argument('--min_bpe_pairs_occurrence',
                        type=int,
                        default=1,
                        help="Minimum occurrence threshold for BPE token pairs"
                        )

    parser.add_argument('--enable_gui',
                        default=False,
                        action='store_true',
                        help="if true, use GUI interface for interaction.")

    args = parser.parse_args(sys.argv[1:])

    log_path = os.path.join(LOG_DIR, args.task_name + ".log")
    if args.mode not in ["interact"]:
        add_file_handlers(log_path)

    show_logo(use_logger=True)

    print_formated_args(args)

    if args.mode == "preprocess":
        preprocess(args)
    elif args.mode == "init":
        init_model(args)
    elif args.mode == "train_bpe":
        train_bpe_from_config(args)
    elif args.mode == "train":
        train(args)
    elif args.mode == "interact":
        interact(args)
    elif args.mode == "ppl":
        ppl(args)

if __name__ == "__main__":

    main()
