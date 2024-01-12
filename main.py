# Author: Larnell Moore
# Creation Date: August 23, 2023
# Date Modified: October 23, 2023
# Purpose: Generates training data into the proper JSONL format that can get passed into the OpenAI API.
import streamlit as st
import pandas as pd
import json
import os
from AIhelper import read_data, generate_jsonl,get_system_prompt, write_data, add_dialogue, delete_entry, set_system_prompt, generate_ai_QA_with_no_context, num_tokens_from_string
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


# The EXCEL allows the the user to easily prepare and share their dataset.
EXCEL_FILE = "um_dearborn_data.xlsx"

# The AI Prompt file allows the user to set a custom system prompt.
AI_PROMPT_FILE = "ai_system_prompt.txt"

# Check if Excel exists, if not, create it
if not os.path.exists(EXCEL_FILE):
    df = pd.DataFrame(columns=["id", "user_message", "assistant_message", "tokens"])
    df.to_excel(EXCEL_FILE, index=False)
    
# Sets the environment variables and creates a conversation AI Agent with buffer memory.
load_dotenv()   
chat = ChatOpenAI(temperature=0.9)
llm = ChatOpenAI()
conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory(), verbose=True)

# Main Menu options
st.title("Training Data for OPENAI Finetuning")
choice = st.selectbox("Choose a page", ["Home", "Edit Data", "Delete Entries", "Ask AI"])

if choice == "Home":
    st.header("Add Dialogue!")
    user_msg = st.text_input("User Message")
    assistant_msg = st.text_input("Assistant Response")
    tokens = num_tokens_from_string(user_msg, "cl100k_base") +  num_tokens_from_string(assistant_msg, "cl100k_base")
    
    if st.button("Add Dialogue to Dataset"):
        add_dialogue(user_msg, assistant_msg, tokens, EXCEL_FILE)
        st.success("Dialogue added successfully!")
        
    if st.button("Generate JSONL"):
        generate_jsonl(AI_PROMPT_FILE,EXCEL_FILE)
        st.success("JSONL file generated successfully!")
        
    st.write("### Dialogues Table")
    df = read_data(EXCEL_FILE)
    st.write(df)
    
elif choice == "Delete Entries":
    df = read_data(EXCEL_FILE)
    for _, row in df.iterrows():
        col1, col2, col3, col4 = st.columns(4)
        col1.write(f"User: {row['user_message']}")
        col2.write(f"Assistant: {row['assistant_message']}")
        if col4.button(f"Delete ID {row['id']}"):
            delete_entry(row['id'], EXCEL_FILE)
            st.success(f"Deleted entry {row['id']} successfully!")

elif choice == "Edit Data":
    df = read_data(EXCEL_FILE)
    st.header("Edit Data!")

    # Allow the user to manually edit data in the table
    for index, row in df.iterrows():
        userMSG = st.text_input(f"User Message (ID: {row['id']})", row["user_message"])
        assistantMSG = st.text_input(f"Assistant Message (ID: {row['id']})", row["assistant_message"])
        tokenMSG =  num_tokens_from_string(userMSG, "cl100k_base") +  num_tokens_from_string(assistantMSG, "cl100k_base")
        
        df.loc[index, "user_message"] = userMSG
        df.loc[index, "assistant_message"] = assistantMSG
        df.loc[index, "tokens"] = tokenMSG
        

    # Commit changes button
    if st.button("Commit Changes"):
        write_data(df, EXCEL_FILE)
        st.success("Changes committed to Excel file!")
elif choice == "Ask AI":
    st.write("### Set AI System Prompt")
    prompt = get_system_prompt(AI_PROMPT_FILE)
    new_prompt = st.text_area("AI System Prompt:", prompt)

    if new_prompt != prompt:
        set_system_prompt(new_prompt, AI_PROMPT_FILE)
        st.write("AI System Prompt updated successfully!")

    AI_QA_DEFAULT = "Provide additional context about why you are making this request and what you hope to obtain!"
    AI_QA = st.text_area("Ask AI for Question-Answer Pair: ", AI_QA_DEFAULT)
    
    if AI_QA != AI_QA_DEFAULT:
        st.write(generate_ai_QA_with_no_context(AI_QA, conversation))

    
   
    
