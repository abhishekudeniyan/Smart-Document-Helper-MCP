# 🧠 SmartDoc MCP - Document Intelligence App

A powerful multi-function document analysis tool built with **Gradio**, supporting:

- 📄 Document Summarization (Nebius Qwen / Pegasus fallback)
- ❓ Question Answering
- 🌐 Flowchart (Graphviz DOT code) generation

---

## 🚀 Features

| Functionality         | Description |
|-----------------------|-------------|
| 📂 File Upload        | Supports `.pdf`, `.docx`, `.txt` |
| 📝 Direct Text Input  | Summarize or ask questions on raw text |
| 📄 AI Summarization   | Uses **Qwen 2.5 32B (via Nebius)** or **Google Pegasus** |
| ❓ QA on Text         | Uses `roberta-base-squad2` from HuggingFace |
| 🌐 Flowchart Builder  | Visual summary of key points via Graphviz |
| 🔁 Fallback Logic     | Automatically switches to Pegasus if Nebius fails |

---

## ⚙️ Technologies Used

- [Gradio](https://gradio.app/) — UI Framework
- [Nebius AI Studio](https://studio.nebius.com/) — Qwen summarization
- [Transformers (Hugging Face)](https://huggingface.co/) — For Pegasus & QA
- [Graphviz](https://graphviz.org/) — Flowchart rendering
- [PyMuPDF (`fitz`)](https://pymupdf.readthedocs.io/) — PDF text extraction
- [python-docx](https://python-docx.readthedocs.io/) — Word file reader
- [OpenAI Python SDK](https://github.com/openai/openai-python) — Nebius-compatible

---

## 🔐 .env Configuration

Create a `.env` file in your root directory with:

```env
NEBIUS_API_KEY=your_actual_api_key_here
MODEL_ID=Qwen/Qwen2.5-32B-Instruct
```

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Smart_Document_Helper_MCP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install system dependencies (Ubuntu/Debian):
```bash
sudo apt-get update && sudo apt-get install -y graphviz libgl1-mesa-glx libglib2.0-0
```

4. Launch the application:
```bash
python main.py
```


## Server Deployment

1. Make the launch script executable:
```bash
chmod +x launch.sh
```

2. Run the launch script:
```bash
./launch.sh
```

## System Requirements
- Python 3.8+
- 8GB RAM minimum
- GPU recommended for better performance
