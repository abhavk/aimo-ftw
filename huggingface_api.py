import requests
from prompts import code_prompt
import transformers
import torch
import gc
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# torch.cuda.set_device(0)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.cuda.empty_cache()

torch.cuda.empty_cache()
gc.collect()

API_URL = "https://li1t6zzkr7cn9h8f.us-east-1.aws.endpoints.huggingface.cloud"
RL_API_URL = "https://tcou0ujy480brhb5.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
	"Accept" : "application/json",
	"Authorization": "Bearer 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",
	"Content-Type": "application/json" 
}

# load MODEL_PATH from .env file
from dotenv import load_dotenv
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")

# MODEL_PATH = "/kaggle/input/deepseek-math"
# MODEL_PATH = "deepseek-math-7b-rl"
# MODEL_PATH = "/opt/dlami/nvme/deepseek-math-7b-rl"

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    set_seed
)

config = AutoConfig.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def create_device_map(num_layers):
    num_gpus = torch.cuda.device_count()
    device_map = []
    layers_per_gpu = num_layers // num_gpus
    extra_layers = num_layers % num_gpus
    
    # Assign layers to GPUs
    current_layer = 0
    for gpu_id in range(num_gpus):
        for _ in range(layers_per_gpu):
            device_map.append((f'model.layers.{current_layer}', gpu_id))
            current_layer += 1
        # Distribute extra layers
        if extra_layers > 0:
            device_map.append((f'model.layers.{current_layer}', gpu_id))
            current_layer += 1
            extra_layers -= 1
    
    # Assign remaining parts of the model
    device_map.append(('model.embed_tokens', 0))
    device_map.append(('model.norm', num_gpus - 1))
    device_map.append(('lm_head', num_gpus - 1))
    device_map = {ii:jj for (ii,jj) in device_map}
    return device_map

# Example usage:
num_layers = 30  # Total layers in the model
device_map = create_device_map(num_layers)

# device_map = {ii:jj for (ii,jj) in device_map}

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map=device_map,
    torch_dtype="auto",
    trust_remote_code=True,
    #quantization_config=quantization_config,
    config=config
)

model.eval()

# model.to(device)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype='auto',
    device_map=device_map,
)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda:0") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            last_token = input_ids[0][-len(stop):]
            if torch.all(torch.eq(stop,last_token)):
                return True
        return False

def query(payload):
	response = requests.post(RL_API_URL, headers=headers, json=payload)
	return response.json()

def generate_response(text, type, max_new_tokens, old_key_values=None, local=False):
    if local:
        return generate_response_local(text, type, max_new_tokens, old_key_values)
    else:
        return generate_response_api(text, type, max_new_tokens)
    
APPROACH_STOP_WORDS = ["```output", "```python", "```\nOutput" , ")\n```" , "``````output", "``````python"] #,  
CODE_STOP_WORDS = ["```output", "```", "```\nOutput" , "``````output"]
    
def generate_response_local(text, type, max_new_tokens, old_key_values=None):
    stop_words = None
    if type == "approach":
        stop_words = APPROACH_STOP_WORDS
    else:
        stop_words = CODE_STOP_WORDS

    for stop_word in stop_words:
        result = tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)
        # print(stop_word, result)  # This will show if any stop word causes an issue

    stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    temperature = 0.9
    top_p = 1.0

    model_inputs = tokenizer(text, return_tensors='pt')
    print("Tokenized text input:", model_inputs)
    model_inputs = model_inputs.to(model.device)
    print("Mapped model input", model_inputs)
    input_len = len(model_inputs['input_ids'][0])
    print("Input length:", input_len)
    with torch.no_grad():
        generation_output = model.generate(**model_inputs, 
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            do_sample = True,
            temperature = temperature,
            top_p = top_p,
            num_return_sequences=1, stopping_criteria = stopping_criteria
        )
    print("Generation output:")
    print(generation_output)
    output_ids = generation_output.sequences[0]
    old_key_values = generation_output.past_key_values

    decoded_output = tokenizer.decode(output_ids, skip_special_tokens=True)

    stop_word_cond = False
    for stop_word in stop_words:
        stop_word_cond = stop_word_cond or (decoded_output[-len(stop_word):]==stop_word)

    return decoded_output, stop_word_cond, old_key_values

def generate_response_api(problem, type, max_new_tokens=1042):
    if type == "approach":
        template = """User: Below is a math problem you must solve (non-negative numerical answer):\n{}\nWrite the entire script covering all the steps (use comments and document it well) and print the result. After solving the problem, output the final numerical answer within \\boxed{{}}.\n\nApproach:"""
    elif type == "code":
        template = code_prompt
    else:
        print("Invalid type")
    inputs = template.format(problem)
    print(f"Input: {inputs}")
    output = query({
        "inputs": inputs,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "use_cache": False,
            "return_full_text": False,
            "temperature": 0.69
        }
    })
    return output

if __name__ == "__main__":
    # read problem from input
    problem = input()
    response = generate_response(problem)
    print(response)
