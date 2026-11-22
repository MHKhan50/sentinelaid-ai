import os, requests
from dotenv import load_dotenv
load_dotenv()

def ask(model, system, prompt):
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    timeout = int(os.getenv("REQUEST_TIMEOUT", "240"))
    try:
        r = requests.post(base + "/api/chat", json={
            "model": model, "stream": False,
            "messages": [{"role":"system","content":system},
                         {"role":"user","content":prompt}],
            "options": {"temperature": 0.2}
        }, timeout=timeout)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama connection failed ({base}). Start Ollama and retry. {e}") from e
