import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the model
model = tf.keras.models.load_model('shakespeare_text_generator.h5')

# Load the tokenizer from training (you need to save this tokenizer when you train your model)
# For demonstration, we'll recreate the tokenizer as in the training script
from tensorflow.keras.preprocessing.text import Tokenizer

# Initialize the tokenizer
tokenizer = Tokenizer()

# Load the Shakespeare dataset to fit the tokenizer again (if not saved separately)
data = open('/tmp/shakespeare.txt').read()
corpus = data.lower().split("\n")
tokenizer.fit_on_texts(corpus)

total_words = len(tokenizer.word_index) + 1

# Max sequence length from training
max_sequence_len = 10  # Replace with your actual max_sequence_len from training

# Function to generate text
def generate_text(seed_text, next_words=50):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Streamlit app interface
st.title('Shakespeare Text Generator')
st.write("This app generates text based on a trained model using Shakespeare's works.")

# User input
seed_text = st.text_input('Enter the starting text:', 'To be or not to be')
next_words = st.slider('Number of words to generate:', min_value=1, max_value=100, value=20)

# Generate text on button click
if st.button('Generate Text'):
    generated_text = generate_text(seed_text, next_words)
    st.write(generated_text)
