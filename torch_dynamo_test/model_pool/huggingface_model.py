from transformers import AutoModelForCausalLM, AutoTokenizer

model_fn_list = {}


def model_sst2():
    model_name = "philschmid/MiniLM-L6-H384-uncased-sst2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, num_hidden_layers=2)
    example_input = tokenizer("Hello, my dog is cute", return_tensors="pt")
    return model, example_input


def model_bloom():
    model_name = 'bigscience/bloom-560m'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, n_layer=2)
    example_input = tokenizer("Hello, my dog is cute", return_tensors="pt")
    return model, example_input


def model_gpt():
    model_name = 'MBZUAI/LaMini-GPT-124M'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, n_layer=2)
    example_input = tokenizer("Hello, my dog is cute", return_tensors="pt")
    return model, example_input


def model_gpt2():
    model_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, n_layer=2)
    example_input = tokenizer("Hello, my dog is cute", return_tensors="pt")
    return model, example_input


def model_opt():
    model_name = 'facebook/opt-125m'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, num_hidden_layers=2)
    example_input = tokenizer("Hello, my dog is cute", return_tensors="pt")
    return model, example_input


def model_llama():
    model_name = 'JackFram/llama-68m'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, num_hidden_layers=2)
    example_input = tokenizer("Hello, my dog is cute", return_tensors="pt")
    return model, example_input


def model_falcon():
    model_name = 'tiiuae/falcon-rw-1b'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, num_hidden_layers=2)
    example_input = tokenizer("Hello, my dog is cute", return_tensors="pt")
    return model, example_input


def register_models():
    model_fn_list['sst2'] = model_sst2
    model_fn_list['bloom'] = model_bloom
    model_fn_list['gpt'] = model_gpt
    model_fn_list['gpt2'] = model_gpt2
    model_fn_list['opt'] = model_opt
    # model_fn_list['llama'] = model_llama  # FIXME
    model_fn_list['falcon'] = model_falcon


register_models()
