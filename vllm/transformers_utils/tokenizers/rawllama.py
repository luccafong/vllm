# SPDX-License-Identifier: Apache-2.0

import base64
from collections.abc import Collection, Iterator, Set
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast

import tiktoken

from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer_base import TokenizerBase
from vllm.utils import is_list_of

if TYPE_CHECKING:
    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

logger = init_logger(__name__)


@dataclass
class Encoding:
    input_ids: Union[list[int], list[list[int]]]


TIKTOKEN_MAX_ENCODE_CHARS = 400_000
MAX_NO_WHITESPACES_CHARS = 25_000


def load_bpe_file(model_path: Path) -> dict[bytes, int]:
    """
    Load BPE file directly and return mergeable ranks.

    Args:
        model_path (Path): Path to the BPE model file.

    Returns:
        dict[bytes, int]: Dictionary mapping byte sequences to their ranks.
    """
    mergeable_ranks = {}

    with open(model_path, encoding="utf-8") as f:
        content = f.read()

    for line in content.splitlines():
        if not line.strip():  # Skip empty lines
            continue
        try:
            token, rank = line.split()
            mergeable_ranks[base64.b64decode(token)] = int(rank)
        except Exception as e:
            logger.warning("Failed to parse line '%s': %s", line, e)
            continue

    return mergeable_ranks


def get_reserved_special_tokens(name, count, start_index=0):
    return [
        f"<|{name}_reserved_special_token_{i}|>"
        for i in range(start_index, start_index + count)
    ]


# 200005, ..., 200079
LLAMA4_TEXT_POST_TRAIN_SPECIAL_TOKENS = [
    "<|header_start|>",
    "<|header_end|>",
    "<|eom|>",
    "<|eot|>",
    "<|step|>",
    "<|text_post_train_reserved_special_token_0|>",
    "<|text_post_train_reserved_special_token_1|>",
    "<|text_post_train_reserved_special_token_2|>",
    "<|text_post_train_reserved_special_token_3|>",
    "<|text_post_train_reserved_special_token_4|>",
    "<|text_post_train_reserved_special_token_5|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|finetune_right_pad|>",
] + get_reserved_special_tokens("text_post_train", 61, 8)
# <|text_post_train_reserved_special_token_6|>, ...,
# <|text_post_train_reserved_special_token_66|>

# 200080, ..., 201133
LLAMA4_VISION_SPECIAL_TOKENS = [
    "<|image_start|>",
    "<|image_end|>",
    "<|vision_reserved_special_token_0|>",
    "<|vision_reserved_special_token_1|>",
    "<|tile_x_separator|>",
    "<|tile_y_separator|>",
    "<|vision_reserved_special_token_2|>",
    "<|vision_reserved_special_token_3|>",
    "<|vision_reserved_special_token_4|>",
    "<|vision_reserved_special_token_5|>",
    "<|image|>",
    "<|vision_reserved_special_token_6|>",
    "<|patch|>",
] + get_reserved_special_tokens("vision", 1041, 7)
# <|vision_reserved_special_token_7|>, ...,
# <|vision_reserved_special_token_1047|>

# 201134, ..., 201143
LLAMA4_REASONING_SPECIAL_TOKENS = [
    "<|reasoning_reserved_special_token_0|>",
    "<|reasoning_reserved_special_token_1|>",
    "<|reasoning_reserved_special_token_2|>",
    "<|reasoning_reserved_special_token_3|>",
    "<|reasoning_reserved_special_token_4|>",
    "<|reasoning_reserved_special_token_5|>",
    "<|reasoning_reserved_special_token_6|>",
    "<|reasoning_reserved_special_token_7|>",
    "<|reasoning_thinking_start|>",
    "<|reasoning_thinking_end|>",
]

LLAMA4_SPECIAL_TOKENS = (LLAMA4_TEXT_POST_TRAIN_SPECIAL_TOKENS +
                         LLAMA4_VISION_SPECIAL_TOKENS +
                         LLAMA4_REASONING_SPECIAL_TOKENS)

BASIC_SPECIAL_TOKENS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
]

O200K_PATTERN = r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""  # noqa: E501


class RawLlamaTokenizer(TokenizerBase):

    special_tokens: dict[str, int]
    num_reserved_special_tokens = 2048

    def __init__(self, model_path: Path):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (Path): The path to the Tiktoken model file.
        """
        if not model_path.exists():
            raise FileNotFoundError(
                f"Tokenizer model file not found: {model_path}")

        mergeable_ranks = load_bpe_file(model_path)
        num_base_tokens = len(mergeable_ranks)

        special_tokens = BASIC_SPECIAL_TOKENS + LLAMA4_SPECIAL_TOKENS
        assert len(set(special_tokens)) == len(special_tokens)
        assert len(special_tokens) <= self.num_reserved_special_tokens

        reserved_tokens = [
            f"<|reserved_special_token_{i}|>"
            for i in range(self.num_reserved_special_tokens -
                           len(special_tokens))
        ]
        special_tokens = special_tokens + reserved_tokens

        self.special_tokens = {
            token: num_base_tokens + i
            for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=model_path.name,
            pat_str=O200K_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.n_words: int = num_base_tokens + len(special_tokens)

        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]

        self.pad_id: int = self.special_tokens["<|finetune_right_pad|>"]
        self.eot_id: int = self.special_tokens["<|eot|>"]
        self.eom_id: int = self.special_tokens["<|eom|>"]

        self.thinking_start_id: int = self.special_tokens[
            "<|reasoning_thinking_start|>"]
        self.thinking_end_id: int = self.special_tokens[
            "<|reasoning_thinking_end|>"]

        self.stop_tokens = [
            self.eos_id,
            self.special_tokens["<|eom|>"],
            self.special_tokens["<|eot|>"],
        ]

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "RawLlamaTokenizer":
        logger.info("[qqzz] Init RawLlamaTokenizer with %s, %s", args, kwargs)
        return cls(Path("/data/local/models/arpg_1b/l4_200k_base"))

    @property
    def all_special_tokens_extended(self) -> list[str]:
        return list(self.special_tokens.keys())

    @property
    def all_special_tokens(self) -> list[str]:
        return list(self.special_tokens.keys())

    @property
    def all_special_ids(self) -> list[int]:
        return list(self.special_tokens.values())

    @property
    def bos_token_id(self) -> int:
        return self.bos_id

    @property
    def eos_token_id(self) -> int:
        return self.eos_id

    @property
    def sep_token(self) -> str:
        raise NotImplementedError()

    @property
    def pad_token(self) -> str:
        return "<|finetune_right_pad|>"

    @property
    def is_fast(self) -> bool:
        raise NotImplementedError()

    @property
    def vocab_size(self) -> int:
        return self.n_words

    @property
    def max_token_id(self) -> int:
        return self.vocab_size - 1

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        text: Union[str, list[str], list[int]],
        text_pair: Optional[str] = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ):
        input_ids: Union[list[int], list[list[int]]]
        # For list[str], original prompt text
        if is_list_of(text, str):
            input_ids_: list[list[int]] = []
            for p in text:
                each_input_ids = self.encode_one(p, truncation, max_length)
                input_ids_.append(each_input_ids)
            input_ids = input_ids_
        # For list[int], apply chat template output, already tokens.
        elif is_list_of(text, int):
            input_ids = text
        # For str, single prompt text
        else:
            input_ids = self.encode_one(text, truncation, max_length)
        return Encoding(input_ids=input_ids)

    def get_vocab(self) -> dict[str, int]:
        raise NotImplementedError()

    def get_added_vocab(self) -> dict[str, int]:
        raise NotImplementedError()

    def encode_one(
        self,
        text: str,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ) -> list[int]:
        return self.encode(text, truncation=truncation, max_length=max_length)

    def encode(
        self,
        text: str,
        truncation: Optional[bool] = None,
        max_length: Optional[int] = None,
        add_special_tokens: Optional[bool] = None,
        allowed_special: Literal["all"] | Set[str] | None = None,
        disallowed_special: Literal["all"] | Collection[str] = (),
    ) -> list[int]:
        if allowed_special is None:
            allowed_special = set()
        substrs = (
            substr for i in range(0, len(text), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                text[i:i +
                     TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS))
        t: list[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                ))
        if add_special_tokens:
            t.insert(0, self.bos_id)
            t.append(self.eos_id)
        if truncation:
            t = t[:max_length]
        return t

    def apply_chat_template(self,
                            messages: list["ChatCompletionMessageParam"],
                            tools: Optional[list[dict[str, Any]]] = None,
                            **kwargs) -> list[int]:
        raise NotImplementedError()

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        raise NotImplementedError()

    def decode(self,
               ids: Union[list[int], int],
               skip_special_tokens: bool = True) -> str:
        if isinstance(ids, int):
            ids = [ids]
        return self.model.decode(cast(list[int], ids))

    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        raise NotImplementedError()

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
            s: str, max_consecutive_slice_len: int) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than
        `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
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
