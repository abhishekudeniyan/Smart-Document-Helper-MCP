# SmartDoc MCP Pro

A powerful document processing and analysis tool with AI capabilities.

## Features
- Document Processing (PDF, DOCX, TXT)
- AI-powered Summarization
- Question Answering
- Entity Recognition
- Visual Knowledge Graph

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

The application will be available at http://localhost:7860

Default credentials:
- Username: admin
- Password: mcp@2024

## Server Deployment

1. Make the launch script executable:
```bash
chmod +x launch.sh
```

2. Run the launch script:
```bash
./launch.sh
```

## Environment Variables (Optional)
- `GRADIO_SERVER_NAME`: Server host (default: "0.0.0.0")
- `GRADIO_SERVER_PORT`: Server port (default: 7860)
- `SHARE`: Enable public URL (default: false)

## System Requirements
- Python 3.8+
- 8GB RAM minimum
- GPU recommended for better performance