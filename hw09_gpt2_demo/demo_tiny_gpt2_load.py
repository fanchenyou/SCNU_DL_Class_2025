import torch
from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM

# this is a demo code of running a pre-trained GPT-2
set_seed(123)
tokenizer = AutoTokenizer.from_pretrained('/data/LLM_MODEL/tiny-gpt2')
model = AutoModelForCausalLM.from_pretrained('/data/LLM_MODEL/tiny-gpt2')
tokenizer.pad_token_id = tokenizer.eos_token_id

# You could try a full gpt-2 model https://huggingface.co/openai-community/gpt2/tree/main
# tokenizer = AutoTokenizer.from_pretrained('gpt2_model')
# model = AutoModelForCausalLM.from_pretrained('gpt2_model')

device = "cuda:0"
model = model.to(device)
print(model)  # check the layers 

max_len=10 # max output token nums
model_inputs = tokenizer('To be or not to be,', return_tensors='pt').to(device)
out = model.generate(**model_inputs, use_cache=True, max_new_tokens=max_len,
                        num_beams=5, num_return_sequences=5, output_scores=True,
                        return_dict_in_generate=True)

sentences = out.sequences
# sent_scores = out.sequences_scores 

# decode output to words
responses = tokenizer.batch_decode(sentences)
print("responses:",responses)




# TODO: try to write your dataloader and train the model on Shakespeare dataset