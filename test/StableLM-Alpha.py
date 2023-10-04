#@title Setup

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList,  TextStreamer

from IPython.display import Markdown, display
def hr(): display(Markdown('---'))
def cprint(msg: str, color: str = "blue", **kwargs) -> None:
    color_codes = {
        "blue": "\033[34m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "purple": "\033[35m",
        "cyan": "\033[36m",
    }
    
    if color not in color_codes:
        raise ValueError(f"Invalid info color: `{color}`")
    
    print(color_codes[color] + msg + "\033[0m", **kwargs)


#@title Pick Your Model
#@markdown Refer to Hugging Face docs for more information the parameters below: https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained

# Choose model name
model_name = "stabilityai/stablelm-tuned-alpha-7b" #@param ["stabilityai/stablelm-tuned-alpha-7b", "stabilityai/stablelm-base-alpha-7b", "stabilityai/stablelm-tuned-alpha-3b", "stabilityai/stablelm-base-alpha-3b"]

cprint(f"Using `{model_name}`", color="blue")

# Select "big model inference" parameters
torch_dtype = "float16" #@param ["float16", "bfloat16", "float"]
load_in_8bit = False #@param {type:"boolean"}
device_map = "auto"

cprint(f"Loading with: `{torch_dtype=}, {load_in_8bit=}, {device_map=}`")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=getattr(torch, torch_dtype),
    load_in_8bit=load_in_8bit,
    device_map=device_map,
    offload_folder="./offload",
)
streamer = TextStreamer(tokenizer)


#@title Generate Text
#@markdown Note: The model response is colored in green

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

# Process the user prompt
user_prompt = "Can you write a song about a pirate at sea?" #@param {type:"string"}
if "tuned" in model_name:
    # Add system prompt for chat tuned models
    system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
    - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
    - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
    - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
    - StableLM will refuse to participate in anything that could harm a human.
    """
    prompt = f"{system_prompt}<|USER|>{user_prompt}<|ASSISTANT|>"
else:
    prompt = user_prompt

# Sampling args
max_new_tokens = 128 #@param {type:"slider", min:32.0, max:3072.0, step:32}
temperature = 0.7 #@param {type:"slider", min:0.0, max:1.25, step:0.05}
top_k = 0 #@param {type:"slider", min:0.0, max:1.0, step:0.05}
top_p = 0.9 #@param {type:"slider", min:0.0, max:1.0, step:0.05}
do_sample = True #@param {type:"boolean"}

cprint(f"Sampling with: `{max_new_tokens=}, {temperature=}, {top_k=}, {top_p=}, {do_sample=}`")
hr()

# Create `generate` inputs
inputs = tokenizer(prompt, return_tensors="pt")
inputs.to(model.device)

# Generate
tokens = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    do_sample=do_sample,
    pad_token_id=tokenizer.eos_token_id,
    stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
    streamer=streamer
)

# Extract out only the completion tokens
completion_tokens = tokens[0][inputs['input_ids'].size(1):]
completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)

# Display
print(user_prompt + " ", end="")
cprint(completion, color="green")