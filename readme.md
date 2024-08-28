# Irish Poem Text Generator

This project is a text generation model trained on a dataset of Irish poems. The model generates text that mimics the style and content of Irish poetry.

## Overview

The Irish Poem Text Generator is a machine learning application that uses a Long Short-Term Memory (LSTM) neural network to generate text based on Irish poetry. This project demonstrates the capabilities of deep learning in natural language processing (NLP), specifically in generating text that follows the patterns and styles of a given corpus.

## Dataset

The dataset used for training this model consists of a collection of Irish poems. The text data has been preprocessed to lower case and split into sequences for model training.

## Model Architecture

The model uses a sequential neural network architecture with the following layers:

1. **Embedding Layer**: Converts integer-encoded words into dense vectors of fixed size. This layer helps the model to learn word embeddings during training.

2. **LSTM Layer**: A single-direction Long Short-Term Memory (LSTM) layer with 100 units, which is well-suited for sequential data and helps the model remember long-term dependencies in the text.

3. **Dropout Layer**: Adds dropout to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time.

4. **Dense Layer**: A fully connected layer with a softmax activation function to predict the next word in the sequence.

## Training

The model was trained using the following configurations:

- **Loss Function**: Categorical cross-entropy was used as the loss function since this is a multi-class classification problem where the model predicts the next word in a sequence.
  
- **Optimizer**: Adam optimizer was used for training, which is an adaptive learning rate optimization algorithm designed specifically for training deep neural networks.

- **Batch Size**: Training was conducted with a suitable batch size to optimize memory usage and performance.

- **Epochs**: The model was trained for a sufficient number of epochs to ensure convergence without overfitting.

## Usage

To generate new text based on Irish poetry, the model takes a seed text as input and predicts the next word in the sequence. This process is repeated to generate a sequence of text.

### Streamlit Application

A Streamlit application has been created to provide an easy-to-use interface for generating text with the trained model. Users can input a seed phrase and specify the number of words they want to generate.
