import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

import os
HF_TOKEN = os.environ.get("HF_TOKEN")

class LlamaSummarizer:
    def __init__(self, adapter_path="./final-adapter", base_model_name="meta-llama/Llama-3.2-1B-Instruct"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, token=HF_TOKEN)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            token=HF_TOKEN,
            low_cpu_mem_usage=True
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_path, token=HF_TOKEN)
        self.model.to(self.device)
        self.model.eval()

    def summarize(self, text, max_new_tokens=128, temperature=0.7, top_p=0.9):
        prompt = f"""Bạn là một trợ lý tóm tắt văn bản tiếng Việt. Hãy tóm tắt văn bản sau một cách ngắn gọn và chính xác.

Văn bản: {text}

Tóm tắt:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip()

if __name__ == "__main__":
    print("Loading Llama 3.2 Summarizer...")
    llama = LlamaSummarizer(adapter_path="./final-adapter")
    print("Model loaded!")

    test_text = "Trí tuệ nhân tạo (AI) là một lĩnh vực của khoa học máy tính, liên quan đến việc xây dựng các máy móc thông minh có khả năng thực hiện các tác vụ đòi hỏi trí thông minh của con người. Các ứng dụng của AI bao gồm nhận dạng giọng nói, xử lý ngôn ngữ tự nhiên, thị giác máy tính, và nhiều lĩnh vực khác. Gần đây, các mô hình ngôn ngữ lớn (LLM) đã đạt được nhiều tiến bộ vượt bậc."

    print(f"\nInput text:\n{test_text}\n")
    print("Summarizing...")
    summary = llama.summarize(test_text)
    print(f"\nSummary:\n{summary}")
