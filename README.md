# SentinelAid AI
**Local multi-agent humanitarian crisis intelligence and civilian-protection research assistant.**

SentinelAid organizes researcher-supplied public reports into claims, humanitarian needs, evidence gaps, contradictions, and a cautious situation brief. It is a research prototype, not an operational command system.

## Safety and scope
- No targeting, combat, troop movement, weapon, or tactical recommendations.
- No individual-level tracking or identification.
- Does not determine that a legal violation occurred.
- Uses only material pasted by the researcher; it does not independently verify events.
- Use public, non-sensitive sources only. Human review is mandatory.

## Requirements
- macOS, Python 3.10+
- Ollama installed and running
- Models: `qwen2.5:7b`, `llama3.1:8b`, `mistral:7b`, `qwen2.5:3b`, `llama3.2:3b`

## Run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama list
streamlit run app.py
```
Open the local URL printed by Streamlit (usually http://localhost:8501).

If Ollama is not running, start it in another terminal with `ollama serve`.

## Demo
Click **Load fictional demo**, then **Run assessment**. The included scenario is synthetic and does not describe a real conflict.

## Research directions
Compare single-model summarization with the multi-agent pipeline. Evaluate claim extraction, evidence attribution, humanitarian-needs categorization, contradiction detection, unsupported-claim rate, expert-rated usefulness, latency, and local memory use. Publish the annotation protocol and report failure cases.
