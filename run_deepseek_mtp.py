from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry
from vllm.model_executor.models.deepseek_v3 import DeepseekV3ForCausalLM
prompts = [
    "The future of AI is",
    "What is the meaning of life?",
    "How to create a super AI?",
    "What is the best way to learn a new language?",
    "How to make friends with a new person?",
    "What is the best way to learn a new skill?",
    "How to improve your memory?",
    "Can you help me with my math homework?",
    "Is there a way to improve my writing skills?",
    "How to improve my public speaking skills?",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# ModelRegistry.register_model("DeepseekV3ForCausalLM", DeepseekV3ForCausalLM)
llm1 = LLM(
    # model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    model="../hg-cache/deepseekv3",
    tensor_parallel_size=1,
    max_model_len=8192,
    speculative_model="../hg-cache/deepseekv3-draft",
    speculative_draft_tensor_parallel_size=1,
    num_speculative_tokens=1,
    trust_remote_code=True,
    enforce_eager=True,
)

outputs = llm1.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
