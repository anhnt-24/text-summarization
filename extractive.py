import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from pyvi import ViTokenizer
import re

class ExtractiveSummarizer:
    def __init__(self, adapter_path, model_name="vinai/phobert-base-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, return_dict=True)
        
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.to(self.device)
        self.model.eval()
        
        self._abbrs = ["TP.", "PGS.", "TS.", "GS.", "ThS.", "BS.", "KS.", "CN.", "Mr.", "Mrs.", "Ms.", "Dr.", "Prof."]

    def _split_sentences(self, text):
        for i, abbr in enumerate(self._abbrs):
            text = text.replace(abbr, f"__A{i}__")
        sents = re.split(r'(?<=[.!?])\s+(?=[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ"\(])', text)
        if len(sents) <= 1:
            sents = re.split(r'(?<=[.!?])\s+', text)
            
        result = []
        for s in sents:
            for i, abbr in enumerate(self._abbrs):
                s = s.replace(f"__A{i}__", abbr)
            s = s.strip()
            if len(s) > 5: result.append(s)
        return result

    @torch.no_grad()
    def summarize(self, text, top_k=3, max_length=256, threshold=0.7, max_summary_words=60):
        text = " ".join(text.split())
        raw_sentences = self._split_sentences(text)
        if not raw_sentences: return ""

        segmented_sentences = [ViTokenizer.tokenize(s) for s in raw_sentences]

        inputs = self.tokenizer(
            segmented_sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()

        candidate_indices = np.where(probs >= threshold)[0]
        
        if len(candidate_indices) == 0:
            candidate_indices = np.array([np.argmax(probs)])

        sorted_candidates = candidate_indices[np.argsort(probs[candidate_indices])[::-1]]

        selected_indices = []
        current_word_count = 0
        
        for idx in sorted_candidates:
            sentence = raw_sentences[idx]
            words = sentence.split()
            
            if current_word_count + len(words) <= max_summary_words:
                selected_indices.append(idx)
                current_word_count += len(words)
            
            if len(selected_indices) >= top_k:
                break
        
        if not selected_indices:
            selected_indices = [sorted_candidates[0]]

        selected_indices.sort() 

        summary = " ".join([raw_sentences[i] for i in selected_indices])
        return summary.strip()