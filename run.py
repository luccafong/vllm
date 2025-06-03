# SPDX-License-Identifier: Apache-2.0
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer_base import TokenizerRegistry

TokenizerRegistry.register("rawllama_tokenizer",
                           "vllm.transformers_utils.tokenizers.rawllama",
                           "RawLlamaTokenizer")

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(
    model="/data/local/models/arpg_1b/unified",
    # "/data/local/models/arpg_1b/l4_200k_base"
    tokenizer="rawllama_tokenizer",
    tokenizer_mode="custom",
)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
