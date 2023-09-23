import time, torch
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
### CREDITS: Tatsu-lab @ github for Original Alpaca Model/Dataset/Inference Code. @Main for much of inference code - https://twitter.com/main_horse - @Teknium1 for guide - https://twitter.com/Teknium1
### Requires: Nvidia GPU with at least 11GB vram (in 8bit) or 20GB without 8bit
### Download Latest Files from https://huggingface.co/chavinlo/alpaca-native/tree/main - Whichever checkpoint-xxx is the highest (this is full fine tuned model, not LORA)
### Your folder structure before running should include config.json, pytorch_model.bin.index.json, pytorch_model-00001-3.bin, tokenizer.model, and tokenizer_config.json
### Change ./checkpoint-800/ to the directory of your HF-Format Model Files Directory
### Requires CUDA Enabled Pytorch. Installation guide here: https://pytorch.org/get-started/locally/
### Currently Requires transformers install from GitHub (not pypackage) - use pip install git+https://github.com/huggingface/transformers.git
### You need at least 24GB of VRAM to run the model in fp16 (for the 7B Alpaca). You need to install bitsandbytes and set load_in_8bit=true to run in 8bit,
 ### which can allow running on 12GB VRAM. BitsandBytes does not have native support on Windows, so be advised. 
### Here is a guide to get BitsandBytes setup on Windows for 8bit: https://rentry.org/llama-tard-v2#install-bitsandbytes-for-8bit-support-skip-this-on-linux

tokenizer = LlamaTokenizer.from_pretrained("./checkpoint-800/")

# Leave this generate_prompt in tact - the fine tune requires prompts to be in this format
def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ### Instruction:
    {instruction}
    ### Input:
    {input}
    ### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
    ### Instruction:
    {instruction}
    ### Response:"""

model = LlamaForCausalLM.from_pretrained(
    "checkpoint-800",
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto"
)

while True:
    text = generate_prompt(input("User: "))
    time.sleep(1)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    generated_ids = model.generate(input_ids, max_new_tokens=250, do_sample=True, repetition_penalty=1.0, temperature=0.8, top_p=0.75, top_k=40)
    print(tokenizer.decode(generated_ids[0]))
