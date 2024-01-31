def generate_prompt(user_query):
    instruction = user_query["instruction"]
    input = user_query["input"]
    output = user_query["output"]
    prompt = f"""
        Below is an instruction that describes a task, paired with an input that
        provides further context. Write a response that appropriately completes the
        request.

        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
        {output}
    """
    return prompt

def tokenize(tokenizer, prompt, cutoff_len):
    return tokenizer(
        prompt + tokenizer.eos_token,
        truncation=True,
        max_length=cutoff_len ,
        padding="max_length"
    )

