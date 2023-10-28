#Import librairies:
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

#Create generation function:
def generate_response(model, tokenizer, prompt, max_length=250):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Create the attention mask and pad token id
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = tokenizer.eos_token_id

    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

model_path = r"C:\Users\Output trained model"

# Determine the available device (CPU or GPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the fine-tuned model and tokenizer
meducol_chatbot = GPT2LMHeadModel.from_pretrained(model_path)
meducol_tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Move the model to the selected device
meducol_chatbot.to(device)

#Test the model:
prompt = "Hi doctor, can you help me ?"
response = generate_response(
    model=meducol_chatbot, 
    tokenizer=meducol_tokenizer, 
    prompt=prompt, 
    max_length=100)  

print("Generated response:", response)
