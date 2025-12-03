# app.py

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LEN = 10 

@st.cache_resource 
def load_resources():
    """Loads the heavy Keras model and the tokenizer only once."""
    try:
        model = load_model('model.h5')
        
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        return model, tokenizer
    except FileNotFoundError:
        st.error("Model or Tokenizer files not found. Make sure 'model.h5' and 'tokenizer.pickle' are in the same directory.")
        return None, None

model, tokenizer = load_resources()

if model is None:
    st.stop()

index_word = tokenizer.index_word

st.set_page_config(layout="wide")
st.title("GhostWriter ✍️: Context-Aware AI Autocomplete")
st.caption("LSTM Model Trained on Technical Articles | Latency optimized with Streamlit Caching.")


def generate_suggestions(input_text, model, tokenizer, max_len, top_n=3):
    """
    Takes user input, preprocesses it, and generates the top N word suggestions.
    """
    text_to_process = input_text.lower()
    text_to_process = re.sub(r'[^a-z0-9\s]', '', text_to_process)
    
    token_list = tokenizer.texts_to_sequences([text_to_process])[0]
    
    if not token_list:
        return [], []
        
    token_list_padded = pad_sequences([token_list], maxlen=max_len, padding='pre')
    
    predicted_probs = model.predict(token_list_padded, verbose=0)[0]
    
    top_n_indices = predicted_probs.argsort()[-top_n:][::-1]
    
    suggestions = []
    probabilities = []
    
    for idx in top_n_indices:
        word = index_word.get(idx, "[UNKNOWN]")
        prob = predicted_probs[idx]
        suggestions.append(word)
        probabilities.append(prob)
        
    return suggestions, probabilities



input_text = st.text_input("Start typing a tech sentence...", "The future of artificial intelligence is")

if input_text and model is not None:
    suggestions, probabilities = generate_suggestions(
        input_text, model, tokenizer, MAX_SEQUENCE_LEN
    )
    
    st.subheader("Contextual Suggestions")
    
    col_metrics, col_chart = st.columns([2, 1])
    
    with col_metrics:

        for word, prob in zip(suggestions, probabilities):
            st.metric(label=f"Suggested Word: **{word}**", 
                      value=f"{prob*100:.2f}%", 
                      delta="Confidence")
            
    with col_chart:

        chart_data = pd.DataFrame({
            'Word': suggestions, 
            'Confidence': probabilities
        })

        st.bar_chart(chart_data, x='Word', y='Confidence', height=300)