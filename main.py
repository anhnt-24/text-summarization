from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from typing import List
import uvicorn

from abstractive import AbstractiveSummarizer
from extractive import ExtractiveSummarizer

app = FastAPI()
templates = Jinja2Templates(directory="templates")

abs_summarizer = AbstractiveSummarizer(model_path="./best-lora")
ext_summarizer = ExtractiveSummarizer(adapter_path="./best-lora-adapter")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html", 
        context={}
    )

@app.post("/summarize", response_class=HTMLResponse)
async def do_summarize(
    request: Request, 
    text: str = Form(...), 
    methods: List[str] = Form(...) 
):
    if not text.strip():
        return templates.TemplateResponse(
            request=request,
            name="index.html", 
            context={"error": "Vui lòng nhập văn bản cần tóm tắt!"}
        )

    results = {}
    try:
        if "extractive" in methods:
            results["extractive"] = ext_summarizer.summarize(text, top_k=2)
            
        if "abstractive" in methods:
            results["abstractive"] = abs_summarizer.generate_summary(text)
            
    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="index.html", 
            context={"error": f"Lỗi xử lý: {str(e)}", "original_text": text}
        )

    return templates.TemplateResponse(
        request=request,
        name="index.html", 
        context={
            "original_text": text,
            "results": results, 
            "selected_methods": methods
        }
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)