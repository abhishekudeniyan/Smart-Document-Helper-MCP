# summary.py
from modal import Stub, web_endpoint
from transformers import pipeline

stub = Stub("docbot-summary")

@stub.function()
@web_endpoint(method="POST")
def summarize(req):
    data = req.json
    text = data.get("text", "")
    if not text:
        return {"summary": "No input provided"}
    
    summarizer = pipeline("summarization", model="google/pegasus-xsum")
    output = summarizer(text[:1024])[0]["summary_text"]
    
    return {"summary": output}
