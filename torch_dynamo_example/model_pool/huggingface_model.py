from transformers import AutoModelForCausalLM, AutoTokenizer

model_list = {}


def model_gpt2():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", n_layer=2)
    example_input = tokenizer("Hello, my dog is cute", return_tensors="pt")
    return model, example_input


def register_models():
    model_list["gpt2"] = model_gpt2


register_models()
