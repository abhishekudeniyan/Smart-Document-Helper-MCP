
# 🧠 SmartDoc MCP Server

An intelligent document processing application powered by AI, built with Gradio and implementing Model Context Protocol (MCP) for seamless integration.

## 🚀 Features

### 🛠️ Three Core MCP Tools:

1. **📄 Document Summarizer** 
   - AI-powered text summarization using BART-Large-CNN
   - Compression ratio tracking
   - Support for PDF, DOCX, and TXT files

2. **💬 Chat with Documents**
   - Interactive Q&A with your documents  
   - Context-aware responses using DistilBERT
   - Conversation history tracking
   - Confidence scoring

3. **🗺️ Mindmap Generator**
   - Visual flowchart/mindmap creation
   - Multiple diagram styles (flowchart/mindmap)
   - DOT format output for Graphviz rendering

## 🎯 Use Cases

- **📚 Research & Study**: Summarize academic papers and research documents
- **💼 Business**: Process reports, contracts, and meeting notes  
- **📖 Content Analysis**: Extract insights from large text documents
- **🎓 Education**: Create study materials and visual summaries

## 🔧 Technology Stack

- **Frontend**: Gradio 4.0+ with custom CSS styling
- **AI Models**: 
  - Summarization: `facebook/bart-large-cnn` (fallback: `t5-small`)
  - Q&A: `distilbert-base-cased-distilled-squad` (fallback: `deepset/roberta-base-squad2`)
- **Document Processing**: PyMuPDF, python-docx
- **Visualization**:

## Quick Start

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
