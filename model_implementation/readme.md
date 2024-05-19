This project demonstrates how to implement Named Entity Recognition (NER) for Arabic text using the HuggingFace Transformers library and the BERT architecture. The implementation involves loading a dataset, tokenizing and aligning labels, training a BERT model for token classification, evaluating the model, and deploying it using Gradio.

## Overview

- **Dataset**: Utilizes the `conllpp-ner-ar` dataset from HuggingFace datasets.
- **Model**: Employs `aubmindlab/bert-base-arabertv02` for token classification.
- **Training**: Includes tokenization, label alignment, and model training using the Trainer API.
- **Evaluation**: Evaluates the model on a test dataset to measure performance metrics.
- **Deployment**: Uses Gradio to create an interactive web interface for the NER model, allowing users to input Arabic text and receive NER predictions.
