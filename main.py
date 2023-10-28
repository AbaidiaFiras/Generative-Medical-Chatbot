#Import librairies:
import docx
import numpy as np 
import os
import re
import torch
from PyPDF2 import PdfReader  
from transformers import Trainer, TrainingArguments
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from torch.utils.data import Dataset

# Functions to read different file types
def read_pdf(file_path):
    with open(file_path, "rb") as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def read_word(file_path):
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def read_txt(file_path):
    with open(file_path, "r", encoding="ISO-8859-1") as file:
        text = file.read()
    return text

def read_documents_from_directory(directory):
    combined_text = ""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            combined_text += read_pdf(file_path)
        elif filename.endswith(".docx"):
            combined_text += read_word(file_path)
        elif filename.endswith(".txt"):
            combined_text += read_txt(file_path)
    return combined_text

#Load data:
data_directory = r"C:\Users\edito\OneDrive\Documents\MEDUCOL Stroke Dataset"
data_files = read_documents_from_directory(data_directory)
#data_files = re.sub(r'\n+', '\n', data_files).strip()
with open("text_data", "w") as f:
        f.write(data_files)
        
# Initialize lists to store conversations
conversations = []
# Define role-specific tokens
role_tokens = {
    "Nurse": "[Nurse]",
    "Intern": "[Intern]",
    "Patient": "[Patient]",
    "Senior": "[Senior]"
}
# Process each data file
with open("text_data", "r") as file:
    lines = file.readlines()
    conversation = []
    current_role = None
    for line in lines:
      # Determine the role and text
      if ': ' in line:
        # Determine the role and text
        role, text = line.strip().split(': ', 1)
        current_role = role_tokens.get(role, current_role)
        conversation.append(current_role + " " + text)
      else:
        pass
    current_role = role_tokens.get(role, current_role)
    conversation.append(current_role + " " + text)
conversations.append('\n'.join(conversation))

# Assuming 'conversations' contains a single conversation
single_conversation = conversations[0]

# Insert a marker to indicate the start of a new conversation
split_marker = "[NEW_CONVERSATION]"
single_conversation_with_markers = single_conversation.replace("\n", f"\n{split_marker}")

# Split the text into individual conversations based on the marker
individual_conversations = single_conversation_with_markers.split(split_marker)

# Remove any empty conversations
individual_conversations = [conv.strip() for conv in individual_conversations if conv.strip()]

# Define the split ratio (80% for training, 20% for validation)
split_ratio = 0.8
split_index = int(len(individual_conversations) * split_ratio)

# Split the individual conversations into training and validation parts
train_conversations = individual_conversations[:split_index]
val_conversations = individual_conversations[split_index:]

# Save the training and validation data as text files
with open("train.txt", "w") as f:
    f.write("\n".join(train_conversations))
with open("val.txt", "w") as f:
    f.write("\n".join(val_conversations))


#Load dataset:
def load_dataset(file_path, tokenizer, block_size=128):
    dataset= TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size        
    )
    return dataset

#Processing dataset:
def load_data_collator(tokenizer, mlm=False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer= tokenizer,
        mlm=mlm
    )
    return data_collator

#Define train function:
def train_chatbot(train_file_path, val_file_path, model_name, output_dir, 
          overwrite_output_dir, per_device_train_batch_size, per_device_val_batch_size, 
          num_train_epochs, save_steps
          ):   
    
    #Initialize tokenizer and the model:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Add the following lines to set the device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    #Load and process train dataset:
    train_dataset = load_dataset(train_file_path, tokenizer)
    val_dataset = load_dataset(val_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)
    
    #Save tokenizer and the model:
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    
    #Set up the training arguments:
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_val_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps,
        save_total_limit=2,
        logging_dir='./logs'
    )
    
    #Train the model:
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset= val_dataset
    )
    trainer.train()
    trainer.save_model(output_dir)
    

#The main function:
def main():
    #Define the parameters:
    train_file_path="train.txt"
    val_file_path="val.txt"
    model_name="gpt2"
    output_dir=r"C:\Users\edito\OneDrive\Documents\Output trained model"
    overwrite_output_dir= True
    per_device_train_batch_size=4
    per_device_val_batch_size=4
    num_train_epochs=150
    save_steps=10_000
    
    #Train the chatbot:
    train_chatbot(train_file_path=train_file_path, 
                  val_file_path=val_file_path, 
                  model_name=model_name, 
                  output_dir=output_dir,
                  overwrite_output_dir=overwrite_output_dir,
                  per_device_train_batch_size=per_device_train_batch_size,
                  per_device_val_batch_size=per_device_val_batch_size,
                  num_train_epochs=num_train_epochs,
                  save_steps=save_steps)
    
    # Load the fine-tuned model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(output_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
    
if __name__ == "__main__":
    main()
