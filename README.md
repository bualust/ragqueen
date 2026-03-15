# RagQueen

This is a RAG (Retrieval-Augmented Generation) to retrieve information from documentation written in markdown.
It uses LangChain to process the inputs, HuggingFaces' `all-MiniLM-L6-v2` as sentence transformer and OllamaModel `llama3.1` for the inferance.
 

Firstly set the User Agent string to identify yourself when scraping the inputs

```bash
export USER_AGENT="ragqueen/1.0"
```

Set
```bash
export TOKENIZERS_PARALLELISM=false
```

You can then run the RAG via
```bash
uv run main.py --config config.yaml 
```
