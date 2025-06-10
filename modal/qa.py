# qa.py
from modal import Stub, web_endpoint
from transformers import pipeline

stub = Stub("docbot-qa")

@stub.function()
@web_endpoint(method="POST")
def answer(req):
    data = req.json
    context = data.get("context", "")
    question = data.get("question", "")
    
    if not context or not question:
        return {"answer": "Missing context or question", "score": 0}
    
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    result = qa_pipeline(question=question, context=context[:1024])
    
    return {"answer": result["answer"], "score": result["score"]}
