import streamlit as st
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import os
from dotenv import load_dotenv
load_dotenv()

st.subheader("Testing")

# Load model and tokenizer
hf_token = os.getenv("hf_token")
model = BertForSequenceClassification.from_pretrained(r"research\Bank_chat_bot\fine_tuned_bert_model")  
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', token=hf_token)  

def preprocess_input(input_text, tokenizer):
    # Tokenize the input
    inputs = tokenizer(input_text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    return inputs

def save_feedback_to_csv(query, correct_label, feedback_file):
    # Attempt to read the existing feedback file, if not found, create a new one
    try:
        feedback_data = pd.read_csv(feedback_file)
    except FileNotFoundError:
        feedback_data = pd.DataFrame(columns=["query", "label"])
    
    # Add new feedback entry and save it
    new_feedback = pd.DataFrame({"query": [query], "label": [correct_label]})
    feedback_data = pd.concat([feedback_data, new_feedback], ignore_index=True)
    feedback_data.to_csv(feedback_file, index=False)

def get_feedback(query, model, tokenizer, feedback_file, confidence_threshold=0.8): 
    # Preprocess the query and get model prediction
    inputs = preprocess_input(query, tokenizer=tokenizer)
    model.eval()
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=-1)
    predicted_class = torch.argmax(logits, dim=-1)
    
    # Display model prediction
    st.write(f"Model Prediction: {'informative' if predicted_class == 0 else 'transactional'} (Confidence: {probs[0][predicted_class].item():.4f})")
    
    # Define possible classes and select the correct one
    prediction_list = ['informative', 'transactional'] if predicted_class == 0 else ['transactional', 'informative']
    
    # Only ask for feedback if confidence is below threshold
    if probs[0][predicted_class].item() < confidence_threshold:
        # Only ask for feedback if not already submitted
        if 'feedback_submitted' not in st.session_state:
            st.session_state.feedback_submitted = False  # Track feedback status
        
        if not st.session_state.feedback_submitted:
            st.session_state.user_feedback = None  # Reset feedback field
            
            # Ask for feedback input
            st.session_state.user_feedback = st.text_input("Is this prediction correct? (Y/N): ")
        
        else:
            st.write("Feedback has already been submitted for this prediction.")

query = st.text_input("Enter Text")

# Ensure only valid, non-empty query is entered
if st.button("Predict") and query.strip():
    # Check if feedback has been submitted already to prevent multiple predictions without feedback
    if 'feedback_submitted' not in st.session_state or not st.session_state.feedback_submitted:
        get_feedback(query=query, 
                     model=model, tokenizer=tokenizer, 
                     feedback_file="intent_data.csv", 
                     confidence_threshold=0.8)
    else:
        st.write("You have already provided feedback for this query. Please enter a new query to make another prediction.")

elif query.strip() == "":
    st.write("Please enter a query to make a prediction.")

# Store feedback when user clicks submit button
if st.button("Submit Feedback"):
    if st.session_state.user_feedback:
        user_feedback = st.session_state.user_feedback.lower()
        
        # Check feedback and save to CSV
        if user_feedback == 'y':
            save_feedback_to_csv(query=query, correct_label='informative' if 'informative' in query else 'transactional', feedback_file="intent_data.csv")
            st.write(f"Thank you for the feedback! As per feedback, 'informative' intent has been saved.")
            st.session_state.feedback_submitted = True  # Mark feedback as submitted
            
        elif user_feedback == 'n':
            save_feedback_to_csv(query, 'transactional' if 'informative' in query else 'informative', feedback_file="intent_data.csv")
            st.write(f"Thank you for the feedback! As per feedback, 'transactional' intent has been saved.")
            st.session_state.feedback_submitted = True  # Mark feedback as submitted
        
        else:
            st.write("Invalid input. Please enter 'Y' or 'N'.")
    else:
        st.write("Please enter your feedback (Y/N) before submitting.")
