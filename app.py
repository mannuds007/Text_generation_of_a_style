import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle


model = tf.keras.models.load_model('iris_poem_generator_reduced.h5')


with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)


max_sequence_len = 10


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
st.write("Generate text based on Shakespeare's works using a machine learning model.")

seed_text = st.text_input('Enter the starting text:', 'To be or not to be')
next_words = st.slider('Number of words to generate:', min_value=1, max_value=100, value=20)


if st.button('Generate Text'):
    generated_text = generate_text(seed_text, next_words)
    st.write(generated_text)
