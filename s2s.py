import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Load pre-trained model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-de"  # English to German model
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Streamlit interface
st.title("Language Translator")
st.write("Using pre-trained model for translation")

# Translate function
def translate(sentence):
    inputs = tokenizer.encode(sentence, return_tensors="pt")
    outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# Input and output in Streamlit
input_text = st.text_input("Enter english text to translate to german:")
if st.button("Translate"):
    if input_text:
        translation = translate(input_text)
        st.write("Translation:", translation)
    else:
        st.write("Please enter some text.")
