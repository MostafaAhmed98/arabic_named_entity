# Arabic Named Entity Recognition with HuggingFace and AraBERT

This project demonstrates how to implement Named Entity Recognition (NER) for Arabic text using the HuggingFace Transformers library and the BERT architecture. The implementation involves loading a dataset, tokenizing and aligning labels, training a AraBERT model for token classification, evaluating the model, and deploying it using Gradio.

## Overview

- **Dataset**: Utilizes the `conllpp-ner-ar` dataset from HuggingFace datasets.
- **Model**: Employs `aubmindlab/bert-base-arabertv02` for token classification.
- **Training**: Includes tokenization, label alignment, and model training using the Trainer API.
- **Evaluation**: Evaluates the model on a test dataset to measure performance metrics.
- **Deployment**: Uses Gradio to create an interactive web interface for the NER model, allowing users to input Arabic text and receive NER predictions.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MostafaAhmed98/arabic_named_entity
   cd arabic_named_entity
   ```

2. Install the required libraries:
   ```bash
   pip install transformers datasets evaluate gradio
   ```

## Usage

To use the Gradio app, run the following command:

```bash
python app.py
```

This will launch a Gradio interface in your web browser where you can input Arabic text and get Named Entity Recognition predictions.

## Acknowledgements

- HuggingFace for the `transformers` and `datasets` libraries.
- `e-hossam96` for the `conllpp-ner-ar` dataset.
- `aubmindlab` for the `bert-base-arabertv02` model.

---

## Gradio App Usage

Here's how you can use the `app.py` file to run the Gradio interface:

```python
import gradio as gr
from pathlib import Path
from transformers import pipeline

# Define base path and default text
base_path = str(Path(__file__).parent)
default_text = "اجتياح رفح الفلسطينية أكبر جريمة إبادة فى التاريخ المعاصر"

def loading_model_and_prediction(ner_text):
    # Load the model checkpoint
    model_checkpoint = base_path + "/checkpoint-3846/"
    token_classifier = pipeline("token-classification", model=model_checkpoint, aggregation_strategy="simple")
    predictions = token_classifier(ner_text)
    formated_preds = [f"the word {i['word']} is labeled as {i['entity_group']}" for i in predictions]
    return formated_preds

def predict(user_text):
    model_preds = loading_model_and_prediction(user_text)
    if len(model_preds) == 0:
        return "No Named Entity Found"
    return "\n".join(model_preds)

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Text(value=default_text, placeholder="Arabic Text", label="Arabic Text"),
    outputs=gr.Text(label="Named Entity Predictions"),
    title="Arabic Named Entity",
    allow_flagging=False
)

# Launch the interface
demo.launch(share=True)
```

Save this code as `app.py` and run it using:
```bash
python app.py
```

This will start a Gradio web interface where you can input Arabic text and receive NER predictions.
