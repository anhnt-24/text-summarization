import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from pyvi import ViTokenizer

class AbstractiveSummarizer:
    def __init__(self, model_path, base_model_name="vinai/bartpho-word"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def generate_summary(self, text):
        text = " ".join(text.split())
        text_segmented = ViTokenizer.tokenize(text)
        inputs = self.tokenizer(text_segmented, return_tensors="pt", max_length=512, truncation=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=128,
                num_beams=4
            )
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_summary = summary.replace("_", " ").strip()
        return final_summary