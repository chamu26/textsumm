import streamlit as st
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

# Load the pre-trained PEGASUS model and tokenizer
model_name = "google/pegasus-cnn_dailymail"
model = PegasusForConditionalGeneration.from_pretrained(model_name)
tokenizer = PegasusTokenizer.from_pretrained(model_name)

# Streamlit UI
st.title("Text Summarizer App")

# Input text area
input_text = st.text_area("Enter the text to summarize:")

if st.button("Summarize"):
    if input_text:
        # Tokenize and summarize the input text
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(input_ids, max_length=150, num_beams=4, length_penalty=2.0, min_length=30, no_repeat_ngram_size=3)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Display the summary
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")


