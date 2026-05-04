# Text Summarization - Vietnamese

Ứng dụng tóm tắt văn bản tiếng Việt với 3 phương pháp: Extractive, Abstractive và LLaMA.

## Yêu cầu

- Python 3.8+
- GPU (khuyến nghị CUDA cho hiệu suất tốt)

## Cài đặt

### 1. Clone project

```bash
git clone https://github.com/anhnt-24/text-summarization.git
cd text-summarization
```

### 2. Tạo môi trường ảo và cài đặt dependencies

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Cài đặt các thư viện
pip install -r requirements.txt
```

### 3. Cài đặt thư viện bổ sung (nếu cần)

```bash
pip install bitsandbytes accelerate
```

### 4. Thiết lập Hugging Face Token

Ứng dụng sử dụng mô hình LLaMA từ Hugging Face. Bạn cần:

1. Tạo tài khoản tại [Hugging Face](https://huggingface.co/)
2. Xin quyền truy cập [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
3. Tạo [Access Token](https://huggingface.co/settings/tokens)
4. Đặt token vào biến môi trường:

```bash
# Windows:
set HF_TOKEN=your_huggingface_token_here

# Linux/Mac:
export HF_TOKEN=your_huggingface_token_here
```

Hoặc tạo file `.env` trong thư mục gốc (đảm bảo không commit file này):

```
HF_TOKEN=your_huggingface_token_here
```

## Chạy ứng dụng

```bash
python main.py
```

Server sẽ khởi động tại: **http://127.0.0.1:8000**

Mở trình duyệt và truy cập địa chỉ trên để sử dụng.

## Các phương pháp tóm tắt

| Phương pháp | Mô tả |
|-------------|-------|
| **Extractive** | Chọn các câu quan trọng nhất từ văn bản gốc (PhoBERT) |
| **Abstractive** | Tạo tóm tắt mới từ văn bản gốc (BARTpho) |
| **LLaMA** | Tóm tắt bằng mô hình LLaMA 3.2 1B |

## Cấu trúc project

```
.
├── main.py                  # FastAPI server
├── abstractive.py          # Abstractive summarizer (BARTpho)
├── extractive.py            # Extractive summarizer (PhoBERT)
├── llama_summarizer.py      # LLaMA summarizer
├── requirements.txt         # Python dependencies
├── templates/
│   └── index.html          # Giao diện web
├── final-adapter/          # LLaMA LoRA adapter
├── best-lora/              # BARTpho LoRA adapter
└── best-lora-adapter/      # PhoBERT LoRA adapter
```
