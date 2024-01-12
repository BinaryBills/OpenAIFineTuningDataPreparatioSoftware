# Author: Larnell Moore
# Creation Date: August 23, 2023
# Date Modified: October 23, 2023
# Purpose: Modularized helper functions to ensure high level of abstraction of program functionalities.
import os
import streamlit as st
import pandas as pd
import json
from langchain.prompts.prompt import PromptTemplate
import tiktoken

def read_data(EXCEL_FILE):
    """Precondition: Functionality triggered that requires program to read excel file
    Postcondition: Returns the excel file or an empty dataframe that only consists of expected columns."""
    if os.path.exists(EXCEL_FILE):
        return pd.read_excel(EXCEL_FILE)
    else:
        return pd.DataFrame(columns=["id", "user_message", "assistant_message"])

def write_data(df, EXCEL_FILE):
    """Writes the dataframe to the excel file."""
    df.to_excel(EXCEL_FILE, index=False)

def add_dialogue(user_msg, assistant_msg,tokens, EXCEL_FILE):
    """Adds the question-answer pair to the finetuning dataset."""
    df = read_data(EXCEL_FILE)
    max_id = df["id"].max() if not df.empty else 0
    new_id = max_id + 1
    df.loc[len(df)] = [new_id, user_msg, assistant_msg, tokens]
    write_data(df, EXCEL_FILE)

def delete_entry(entry_id, EXCEL_FILE):
    """Searches for the ID associated with a row and deletes the corresponding row from the excel file."""
    df = read_data(EXCEL_FILE)
    df = df[df['id'] != entry_id]
    write_data(df, EXCEL_FILE)
    
def get_system_prompt(AI_PROMPT_FILE):
    """Returns the user-specified system prompt or the default system prompt"""
    if os.path.exists(AI_PROMPT_FILE):
        with open(AI_PROMPT_FILE, 'r') as file:
            return file.read().strip()
    else:
        return "You are a wise and polite AI assistant, speaking like Master Yoda, and knowledgeable about the University of Michigan-Dearborn, ending every response with 'my young padawan.'"

def set_system_prompt(prompt, AI_PROMPT_FILE):
    """Given a user-specified system prompt, creates a file that contains the prompt for future usage."""
    with open(AI_PROMPT_FILE, 'w') as file:
        file.write(prompt)

def generate_jsonl(AI_PROMPT_FILE, EXCEL_FILE):
    """Transforms the data stored as an excel file into JSONL"""
    dialogues = []
    system_prompt = get_system_prompt(AI_PROMPT_FILE)
    
    system_message = {
        "role": "system",
        "content": system_prompt
    }
    
    df = read_data(EXCEL_FILE)
    for _, row in df.iterrows():
        dialogues.append([
            system_message,
            {"role": "user", "content": row["user_message"]},
            {"role": "assistant", "content": row["assistant_message"]}
        ])
    with open("um_dearborn_data.jsonl", 'w') as f:
        for dialogue in dialogues:
            formatted_data = json.dumps({"messages": dialogue})
            f.write(formatted_data + '\n')
    st.write("JSONL file generated successfully!")
    
    
def generate_ai_QA_with_no_context(user_query, conversation):
    template = """The following is an AI agent designed to help the human prepare their question-and-answer pairs dataset. To each response made by the user, the AI agent should return a question about the information entered and an answer using the format Q: and A: "
    {history}
    Human: {input} 
    AI Assistant:"""
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    conversation.prompt = PROMPT
    return conversation.predict(input=user_query)

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def generate_ai_QA_with_vector_database(user_query):
    pass