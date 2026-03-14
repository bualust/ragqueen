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
