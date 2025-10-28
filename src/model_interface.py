# src/model_interface.py
import subprocess
import json

class OllamaModel:
    def __init__(self, model_name="llama3.1"):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        result = subprocess.run(
            ["ollama", "run", self.model_name],
            input=prompt.encode("utf-8"),
            capture_output=True
        )
        return result.stdout.decode("utf-8")
