# local-ragbot
Implementation of a local chatbot with CLI interface performing RAG over documents in folder

### Create conda environment and install dependencies
```bash
conda create -n "ragbot" python=3.12
conda activate ragbot
pip install -r requirements.txt
```

### Launch local Ollama server

```bash
ollama serve
```

### On a new terminal window, run the main script

```bash
python main.py
```

```bash
usage: main.py [-h] [--chat_model CHAT_MODEL] [--docs_dir DOCS_DIR] [--max_tokens MAX_TOKENS] [--temperature TEMPERATURE] [--stream]
               [--plot_graph]

Chatbot parameters

options:
  -h, --help            show this help message and exit
  --chat_model CHAT_MODEL
  --docs_dir DOCS_DIR
  --max_tokens MAX_TOKENS
  --temperature TEMPERATURE
  --stream
  --plot_graph
```