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

To download the documentation and process it into embeddings
```bash
uv run process_input.py --config config.yaml
```

To ask a query
```bash
uv run query.py
```
