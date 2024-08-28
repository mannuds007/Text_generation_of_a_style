# Shakespeare Text Generator

This Streamlit app generates text in the style of Shakespeare using a machine learning model trained on Shakespeare's works. The app takes a user-provided seed text and generates a continuation based on the patterns it learned during training.

## Techniques Used

1. **Text Preprocessing**: 
   - The model was trained on a corpus of Shakespeare’s text, which was cleaned and preprocessed to lower case and split into sequences.
   - A tokenizer was used to convert words into numerical representations, which are easier for the model to understand.

2. **Neural Network Architecture**:
   - The model uses a **Bidirectional Long Short-Term Memory (LSTM)** network, which can learn from both the past and future context of a word, making it effective for generating coherent text.
   - **Dropout layers** were added to the model to prevent overfitting and help it generalize better to new data.

3. **Pre-trained Word Embeddings**:
   - **GloVe word embeddings** were used to initialize the model’s word representations. These embeddings capture the semantic meaning of words based on their usage in a large corpus of text, providing a richer understanding of language.

4. **Callbacks**:
   - **Early Stopping** was used to halt training once the model's performance stopped improving, helping to avoid overfitting.
   - **Model Checkpoints** were used to save the best version of the model during training.

## How to Use the App

1. Enter a starting text (seed text) in the input box.
2. Choose the number of words you want to generate.
3. Click the "Generate Text" button to see the generated Shakespeare-like text.
