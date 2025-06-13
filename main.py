import gradio as gr
import fitz  # PyMuPDF
import docx
from pathlib import Path
from transformers import pipeline
import datetime
import graphviz
import tempfile
import os
import hashlib
from typing import List, Dict
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ======================= CONFIG =======================
SUPPORTED_EXT = [".pdf", ".docx", ".txt"]
MAX_TOKENS = 1024
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit per file
MAX_TOTAL_SIZE = 50 * 1024 * 1024  # 50MB total limit

# API Configuration
NEBIUS_API_KEY = os.getenv('NEBIUS_API_KEY')
QWEN_MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
print("🔐 NEBIUS_API_KEY loaded:", "Yes" if NEBIUS_API_KEY else "No")
print("📦 MODEL_ID:", QWEN_MODEL_ID)


# ================== UTILITY FUNCTIONS ==================
def validate_file_size(file_path):
    """Validate individual file size"""
    size = Path(file_path).stat().st_size
    if size > MAX_FILE_SIZE:
        raise ValueError(f"File size exceeds limit of {MAX_FILE_SIZE/1024/1024:.1f}MB")
    return size

def validate_files(file_list):
    """Validate total file size across all files"""
    if not file_list:
        raise ValueError("No files provided")
    
    total_size = 0
    for file in file_list:
        size = validate_file_size(file)
        total_size += size
        if total_size > MAX_TOTAL_SIZE:
            raise ValueError(f"Total file size exceeds limit of {MAX_TOTAL_SIZE/1024/1024:.1f}MB")

# ================== DOCUMENT HANDLER ==================
def extract_text(filepath):
    """Extract text from supported file formats"""
    ext = Path(filepath).suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported file format: {ext}")

    try:
        if ext == ".pdf":
            with fitz.open(filepath) as doc:
                if doc.needs_pass:
                    raise ValueError("PDF is password protected")
                text = "\n".join(page.get_text() for page in doc)
        elif ext == ".docx":
            doc = docx.Document(filepath)
            text = "\n".join(p.text for p in doc.paragraphs)
        elif ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = ""
    except Exception as e:
        raise ValueError(f"Error reading file {filepath}: {str(e)}")

    return text.strip()

class ModelManager:
    """Singleton class to manage AI models"""
    _instance = None
    
    def __init__(self):
        self.summarizer = None
        self.qa_model = None
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance
    
    def load_summarizer(self):
        def qwen_summarize(text):
            try:
                print("🧠 Using Nebius Qwen summarizer...")
                client = OpenAI(
                    base_url="https://api.studio.nebius.com/v1/",
                    api_key=NEBIUS_API_KEY
                )

                response = client.chat.completions.create(
                    model=QWEN_MODEL_ID,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes documents."},
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": f"Summarize this:\n{text}"}]
                        }
                    ]
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"⚠️ Nebius summarizer failed: {e}")
                return None

        def fallback_summarizer(text):
            print("🔁 Falling back to Pegasus summarizer...")
            if self.summarizer is None:
                self.summarizer = pipeline("summarization", model="google/pegasus-xsum")
            return self.summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

        def summarize(text):
            summary = qwen_summarize(text)
            return summary if summary else fallback_summarizer(text)

        return summarize

    def load_qa_model(self):
        """Load question-answering model"""
        if self.qa_model is None:
            try:
                self.qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
            except Exception as e:
                raise RuntimeError(f"Failed to load QA model: {str(e)}")
        return self.qa_model

# Initialize model manager
model_manager = ModelManager.get_instance()

class DocumentProcessor:
    """Handle document processing with caching"""
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_path(self, filepath: str) -> Path:
        """Generate cache file path"""
        file_hash = hashlib.md5(str(filepath).encode()).hexdigest()
        return self.cache_dir / f"{file_hash}.json"
    
    def _chunk_text(self, text: str, chunk_size: int = MAX_TOKENS) -> List[str]:
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_len = len(word) + 1  # +1 for space
            if current_length + word_len > chunk_size:
                if current_chunk:  # Only add if not empty
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_len
            else:
                current_chunk.append(word)
                current_length += word_len
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
    
    def process_text_input(self, text: str, source_name: str = "Direct Input") -> Dict:
        """Process direct text input without file"""
        if not text.strip():
            raise ValueError("Empty text provided")
            
        chunks = self._chunk_text(text)
        
        result = {
            'full_text': text,
            'chunks': chunks,
            'metadata': {
                'filename': source_name,
                'processed_date': datetime.datetime.now().isoformat(),
                'num_chunks': len(chunks),
                'source_type': 'direct_input'
            }
        }
        return result

    def process_document(self, filepath: str) -> Dict:
        """Process document with caching"""
        cache_path = self._get_cache_path(filepath)
        
        # Check cache
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                # If cache is corrupted, process fresh
                pass
        
        # Validate file size
        validate_file_size(filepath)
        
        # Process document
        text = extract_text(filepath)
        chunks = self._chunk_text(text)
        
        result = {
            'full_text': text,
            'chunks': chunks,
            'metadata': {
                'filename': Path(filepath).name,
                'processed_date': datetime.datetime.now().isoformat(),
                'num_chunks': len(chunks),
                'source_type': 'file',
                'file_size': Path(filepath).stat().st_size
            }
        }
        
        # Save to cache
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception:
            # Cache save failed, but continue
            pass
            
        return result

# Initialize document processor
doc_processor = DocumentProcessor()

# ================== SUMMARIZER ==================
def generate_summary(text):
    """Generate AI summary of text"""
    if not text:
        return "❗ No document text available."
    
    # Limit text to MAX_TOKENS
    tokens = text[:MAX_TOKENS]
    try:
        summarizer = model_manager.load_summarizer()
        summary = summarizer(tokens)
        return summary
    except Exception as e:
        return f"❗ Error generating summary: {str(e)}"

# ================== QA ==================
def answer_question(text, question):
    """Answer question based on document text"""
    if not text:
        return {"answer": "❗ No document loaded", "score": 0}
    if not question:
        return {"answer": "❗ No question provided", "score": 0}
    try:
        qa_model = model_manager.load_qa_model()
        result = qa_model(question=question, context=text[:MAX_TOKENS])
        return result
    except Exception as e:
        return {"answer": f"❗ QA error: {str(e)}", "score": 0}

# ================== FLOWCHART ==================
def build_flowchart(text):
    """Generate flowchart from text"""
    if not text:
        return "digraph { ERROR [label=\"No document loaded\"] }"
        
    try:
        dot = graphviz.Digraph(comment='Document Analysis Flowchart')
        dot.attr(rankdir='TB')
        
        # Document node
        dot.attr('node', shape='note', style='filled', fillcolor='lightblue')
        doc_name = text[:30].replace('\n', ' ').replace('"', "'")
        dot.node('DOC', f'Document\n{doc_name}...')
        
        # Summary node
        dot.attr('node', shape='rectangle', style='rounded,filled', fillcolor='lightgreen')
        summary = generate_summary(text)[:100].replace('\n', ' ').replace('"', "'") + '...'
        dot.node('SUM', f'Summary\n{summary}')
        dot.edge('DOC', 'SUM')
        
        # Key points cluster
        with dot.subgraph(name='cluster_0') as c:
            c.attr(label='Key Points', style='rounded', color='gray')
            
            # Extract sentences and create nodes
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20][:5]
            for i, sent in enumerate(sentences):
                node_id = f'P{i}'
                clean_sent = sent[:40].replace('\n', ' ').replace('"', "'") + '...'
                c.node(node_id, clean_sent, shape='box', style='filled', fillcolor='lightyellow')
                dot.edge('SUM', node_id)
        
        return dot.source
    except Exception as e:
        return f"digraph {{ ERROR [label=\"Flowchart failed: {str(e)}\"] }}"

# ================== GRADIO INTERFACE FUNCTIONS ==================
def load_docs(file_list):
    """Load and process multiple documents"""
    try:
        if not file_list:
            raise ValueError("No files provided")
            
        validate_files(file_list)
        results = []
        
        for file in file_list:
            doc_data = doc_processor.process_document(file)
            results.append(doc_data)
        
        combined_text = "\n".join(doc["full_text"] for doc in results)
        doc_info = {
            "num_documents": len(results),
            "total_chunks": sum(doc["metadata"]["num_chunks"] for doc in results),
            "files": [doc["metadata"] for doc in results]
        }
        
        return combined_text, doc_info, "✅ Documents loaded successfully", doc_info
    except Exception as e:
        return None, None, f"❗ Error: {str(e)}", None

def process_direct_text(text, name):
    """Process direct text input"""
    try:
        if not text.strip():
            raise ValueError("Please enter some text")
        
        doc_data = doc_processor.process_text_input(text, name or "Direct Input")
        return (
            doc_data["full_text"],
            {"direct_input": doc_data["metadata"]},
            "✅ Text processed successfully",
            doc_data["metadata"]
        )
    except Exception as e:
        return None, None, f"❗ Error: {str(e)}", None

def use_current_document(doc_text):
    """Use current document text for flowchart"""
    if not doc_text:
        return ""
    return doc_text

def clear_all():
    """Clear all loaded documents"""
    return None, None, "Cleared all documents", None, ""

# ================== GRADIO UI ==================
with gr.Blocks(title="SmartDoc MCP", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    ## 🤖 SmartDoc MCP - Document Intelligence
    Upload documents, get summaries, QA, flowcharts, and more.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tab("File Upload"):
                files = gr.File(
                    label="Upload Files", 
                    file_types=SUPPORTED_EXT, 
                    file_count="multiple", 
                    type="filepath"
                )
                load_btn = gr.Button("📂 Load Documents", variant="primary")
            
            with gr.Tab("Direct Text"):
                direct_text = gr.Textbox(
                    label="Enter Text Directly",
                    placeholder="Paste or type your text here...",
                    lines=5
                )
                text_name = gr.Textbox(
                    label="Text Name (Optional)",
                    placeholder="Give your text a name...",
                    value="Direct Input"
                )
                load_text_btn = gr.Button("📝 Process Text", variant="primary")
            
            doc_text = gr.State()
            doc_info = gr.State()
        
        with gr.Column(scale=1):
            status = gr.Textbox(label="Status", interactive=False)
            file_info = gr.JSON(label="Document Info", visible=True)
            clear_btn = gr.Button("🗑️ Clear", variant="secondary")

    with gr.Tabs() as tabs:
        with gr.TabItem("📄 Summary"):
            with gr.Row():
                summary_output = gr.Textbox(label="AI Summary", lines=8)
                summary_btn = gr.Button("Generate Summary", variant="primary")

        with gr.TabItem("❓ Ask a Question"):
            with gr.Row():
                question_input = gr.Textbox(
                    label="Enter your question",
                    placeholder="What would you like to know?"
                )
            with gr.Row():
                answer_output = gr.JSON(label="Answer with Confidence")
                ask_btn = gr.Button("Get Answer", variant="primary")

        with gr.TabItem("🌐 Flowchart"):
            with gr.Row():
                with gr.Column(scale=2):
                    chart_text = gr.Textbox(
                        label="Text for Flowchart",
                        placeholder="Enter or paste text to create flowchart...",
                        lines=5
                    )
                    use_doc_btn = gr.Button("📄 Use Current Document", variant="secondary")
                with gr.Column(scale=1):
                    chart_btn = gr.Button("🔄 Generate Flowchart", variant="primary")
            
            with gr.Row():
                dot_code = gr.Textbox(label="DOT Source Code", lines=10)
            gr.Markdown("You can copy and paste the DOT code to [Graphviz Online](https://dreampuf.github.io/GraphvizOnline/)")

    # ================== EVENT HANDLERS ==================
    load_btn.click(
        fn=load_docs,
        inputs=[files],
        outputs=[doc_text, doc_info, status, file_info]
    )
    
    load_text_btn.click(
        fn=process_direct_text,
        inputs=[direct_text, text_name],
        outputs=[doc_text, doc_info, status, file_info]
    )
    
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[doc_text, doc_info, status, file_info, chart_text]
    )
    
    use_doc_btn.click(
        fn=use_current_document,
        inputs=[doc_text],
        outputs=[chart_text]
    )
    
    summary_btn.click(
        fn=generate_summary,
        inputs=[doc_text],
        outputs=[summary_output]
    )
    
    ask_btn.click(
        fn=answer_question,
        inputs=[doc_text, question_input],
        outputs=[answer_output]
    )
    
    chart_btn.click(
        fn=build_flowchart,
        inputs=[chart_text],
        outputs=[dot_code]
    )

# ================== LAUNCH ==================
if __name__ == "__main__":
    demo.launch(mcp_server=True,
                share=True)