import requests
from prompts import code_prompt
import transformers
import torch
import gc
import os
import torch.nn as nn
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

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype='auto',
#     device_map=device_map,
# )


class ValueModel(nn.Module):
    def __init__(self, base_model, num_attention_heads, dropout, fc):
        super(ValueModel, self).__init__()
        self.base_model = base_model
        self.dropout = dropout
        self.fc = fc
        self.hidden_size = base_model.config.hidden_size
        self.num_attention_heads = num_attention_heads

        # multi-head attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=num_attention_heads).to(model.dtype).to("cuda:3")
        self.dropout = dropout
        self.fc = fc

    def forward(self, input_ids):
        # Get outputs from the base model
        print(f"Input id dtype: {input_ids.dtype}")
        print(f"Model dtype: {model.dtype}")
        input_ids = input_ids.to("cuda:0")
        outputs = self.base_model(input_ids=input_ids, output_hidden_states=True)
        print(f"Outputs: {outputs.hidden_states[-1]}")

        # Extract hidden states of all tokens from the final layer
        hidden_states = outputs.hidden_states[-1].to("cuda:3")

        # shape: (batch_size, sequence_length, hidden_size)
        # print shape
        print(f"Shape of hidden states: {hidden_states.shape}")

        hidden_states = hidden_states.permute(1, 0, 2)  # (sequence_length, batch_size, hidden_size)
        # Apply multi-head attention
        attn_output, attn_weights = self.multihead_attn(hidden_states, hidden_states, hidden_states)

        # Apply dropout
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, sequence_length, hidden_size)
        pooled_hidden_states = torch.mean(attn_output, dim=1)  # (batch_size, hidden_size)
        dropped_out = self.dropout(pooled_hidden_states)

        # Pass through fully connected layer
        prediction = self.fc(dropped_out)
        
        return prediction
    
# get the value model
value_model = ValueModel(
    base_model=model,
    dropout=nn.Dropout(0.1).to(model.dtype).to("cuda:3"),
    num_attention_heads=8,
    fc=nn.Linear(4096, 1).to(model.dtype).to("cuda:3")
)
# value_model = nn.DataParallel(value_model, device_ids=[0, 1])

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
    
def get_value(input_ids):
    # Get the value of the input
    prediction = value_model(input_ids)
    return prediction

def query(payload):
	response = requests.post(RL_API_URL, headers=headers, json=payload)
	return response.json()

def generate_response(text, in_code_block, max_new_tokens, old_key_values=None, local=False):
    if local:
        return generate_response_local(text, in_code_block, max_new_tokens, old_key_values)
    else:
        return generate_response_api(text, in_code_block, max_new_tokens)
    
APPROACH_STOP_WORDS = ["```output", "```python", "```\nOutput" , ")\n```" , "``````output", "``````python"] #,  
CODE_STOP_WORDS = ["```output", "\n```", "```\nOutput" , "``````output"]
    
def generate_response_local(text, in_code_block, max_new_tokens, old_key_values=None):
    stop_words = None
    stopping_criteria = None
    if in_code_block:
        stop_words = CODE_STOP_WORDS
        stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    # for stop_word in stop_words:
    #     result = tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)
        # print(stop_word, result)  # This will show if any stop word causes an issue

    temperature = 0.9
    top_p = 1.0

    model_inputs = tokenizer(text, return_tensors='pt')
    # print("Tokenized text input:", model_inputs)
    model_inputs = model_inputs.to(model.device)
    # print("Mapped model input", model_inputs)
    input_len = len(model_inputs['input_ids'][0])
    # print("Input length:", input_len)
    with torch.no_grad():
        if stopping_criteria:
            generation_output = model.generate(**model_inputs, 
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                do_sample = True,
                temperature = temperature,
                top_p = top_p,
                num_return_sequences=1, stopping_criteria = stopping_criteria
            )
        else:
            generation_output = model.generate(**model_inputs, 
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                do_sample = True,
                temperature = temperature,
                top_p = top_p,
                num_return_sequences=1
            )
    # print("Generation output:")
    # print(generation_output)
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
