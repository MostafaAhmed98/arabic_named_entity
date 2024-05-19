import gradio as gr
from pathlib import Path
from transformers import pipeline


base_path = str(Path(__file__).parent)
default_text = "اجتياح رفح الفلسطينية أكبر جريمة إبادة فى التاريخ المعاصر"

def loading_model_and_prediction(ner_text):
    # Replace this with your own checkpoint
    model_checkpoint = base_path + "/assets/checkpoint-3846/"
    token_classifier = pipeline("token-classification", model=model_checkpoint, aggregation_strategy="simple")
    predictions = token_classifier(ner_text)
    formated_preds = [f"the word {i['word']} is labeled as {i['entity_group']}" for i in predictions]
    return formated_preds

def predict(user_text):
  model_preds = loading_model_and_prediction(user_text)
  if len(model_preds) == 0:
     return "No Named Entity Found"
  return "\n".join(model_preds)


demo = gr.Interface(fn=predict,inputs=gr.Text(value= default_text,
                                              placeholder="Arabic Text", label="Arabic Text"),
                                              outputs=gr.Text(label="Named Entity Predictions"),
                                              title="Arabic Named Entity",
                                              allow_flagging=False
                                              )
demo.launch(share=True)