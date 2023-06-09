from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_code(prompt):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True)

    # Encode the prompt
    x = tokenizer.encode(prompt, return_tensors='pt')

    # Generate code
    y = model.generate(x, max_length=100, do_sample=True, top_p=0.95, top_k=4, temperature=0.2, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)

    # Decode the generated code
    generated_code = tokenizer.decode(y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return generated_code

prompt = 'def fibonacci(n): ' # Replace this with your prompt
print(generate_code(prompt))
