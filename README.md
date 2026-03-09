# GlioblastomaGPT

<div align="center">

![GlioblastomaGPT](https://img.shields.io/badge/GlioblastomaGPT-v0.1.0-6366f1?style=for-the-badge&logo=brain)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-10b981?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge)

**A RAG + Multi-Agent AI system for patients, caregivers, and clinicians navigating Glioblastoma**

</div>

---

## Why I Built This

My father was diagnosed with Glioblastoma in 2025.

I still remember sitting in a hospital waiting room at 2am, surrounded by medical terms no one had time to explain — MGMT methylation, IDH status, Stupp protocol, temozolomide. The internet gave me either oversimplified articles that said nothing, or raw research papers I couldn't understand. My father's oncologist was doing everything right, but there was no time in those appointments to ask every question I had.

I needed something that could sit between me and that mountain of information. Something that could translate what I was reading, find clinical trials that might be relevant, explain what a drug actually does in plain English — and also just acknowledge, sometimes, that this is terrifying.

**GlioblastomaGPT is that thing.**

It is not a replacement for medical care. It will never pretend to be. But it is the research companion I wish I had — one that speaks both the language of patients and the language of science, and knows which one to use.

---

## What It Does

GlioblastomaGPT is a multi-agent AI system that:

- **Translates medical jargon** into plain English, or into full clinical terminology — you choose the literacy level
- **Searches 50,000+ PubMed abstracts and clinical guidelines** using hybrid retrieval (keyword + semantic)
- **Finds open clinical trials** by querying the live clinicaltrials.gov API in real time
- **Explains drugs** — how they work, side effects, what questions to ask your oncologist
- **Responds with warmth** when the question is really about fear, not facts
- **Never gives dosage advice** — a hard safety layer blocks it every time and redirects to your medical team

### Literacy Modes

The same question, answered differently depending on who is asking:

| Mode | Who it's for | How it responds |
|------|-------------|-----------------|
| `patient` | The person diagnosed | Plain English, no jargon, compassionate tone |
| `caregiver` | Family members | Clear with some detail, medical terms explained inline |
| `clinician` | Doctors, nurses, researchers | Full technical terminology, formatted citations |

---

## How It Works

### The Core Problem: Standard RAG Fails on Medical Text

Most AI systems fail badly at GBM questions. Here is why:

A patient types: *"will chemo help?"*
A standard RAG system searches for: *"will chemo help"*
It finds nothing useful, because the actual research papers say: *"TMZ efficacy in IDH-wildtype glioblastoma following Stupp protocol chemoradiotherapy"*

The gap between how patients ask questions and how medical literature is written is enormous. GlioblastomaGPT bridges that gap with a 3-stage query expansion system before any retrieval happens.

### Stage 1 — Ontology Expansion
A custom `gbm_ontology.json` with 210 GBM-specific entries maps:
- **Acronyms** → full terms: `TMZ` → `temozolomide, Temodar`
- **Synonyms** → clinical language: `"Stupp protocol"` → `"TMZ + radiation, temozolomide chemoradiotherapy"`
- **Gene names** → full context: `MGMT` → `O6-methylguanine-DNA methyltransferase`
- **Patient language** → medical terminology

### Stage 2 — Named Entity Recognition (NER)
`scispaCy` with the `en_core_sci_lg` biomedical model extracts entities from the query — diseases, chemicals, genes, cell types — so they can be injected into the search.

### Stage 3 — GPT-4o Query Rewrite
The expanded query is passed to GPT-4o with a medical search optimization prompt. It produces a final search query that sounds like it came from a clinician, not a patient.

**Result: +28–35% retrieval hit-rate improvement on acronym-heavy queries** (measured in ablation tests against the golden evaluation set).

### Retrieval: Hybrid BM25 + Dense Embeddings

Once the query is expanded, retrieval uses two methods in parallel:

- **BM25** (keyword-based): catches exact matches — drug names, gene names, trial IDs
- **BGE-M3** (dense semantic): catches meaning — finds papers about a concept even if the words differ
- **Reciprocal Rank Fusion (RRF)**: merges both result sets with weighted scoring

This hybrid approach outperforms either method alone, especially for GBM's mix of precise technical terms and broad conceptual questions.

Two backend options are available via a single `.env` setting:
- **Version A (`RETRIEVAL_MODE=pinecone`)**: Pinecone vector database — scales to millions of documents
- **Version B (`RETRIEVAL_MODE=numpy`)**: Pure NumPy cosine similarity — fully local, zero external dependencies

---

## Agent Architecture

Every query flows through a pipeline of specialized agents orchestrated by LangGraph.

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                      Triage Agent                            │
│  Classifies query into: treatment | trial | drug |          │
│                          emotional | research                │
│  + Runs 3-stage medical query expansion in parallel         │
└──────┬──────────┬────────────┬──────────────┬───────────────┘
       │          │            │              │
       ▼          ▼            ▼              ▼
  Research    Clinical      Drug          Emotional
   Agent      Trial         Info          Support
              Agent         Agent          Agent
  PubMed +   Live           FDA labels   Warmth +
  Guidelines clinicalt.   + Ontology     Resources
  50k+ docs   .gov API
       │          │            │              │
       └──────────┴────────────┴──────────────┘
                              │
                              ▼
                    Synthesizer Agent
               Merges outputs from all agents,
               removes contradictions, adds
               citations, translates to the
               correct literacy level
                              │
                              ▼
                    Safety Layer (HITL)
         ┌────────────────────────────────────────┐
         │  Dosage advice detected?  → BLOCK       │
         │  Prognosis statistics?    → Add context │
         │  Crisis signal in query?  → Route to    │
         │                             crisis help │
         │  Low confidence score?    → Flag it     │
         └────────────────────────────────────────┘
                              │
                              ▼
                        Final Answer
                  With citations, confidence
                  score, and any disclaimers
```

### The Agents in Detail

**Triage Agent**
Classifies every incoming query into one of five types and runs medical query expansion in parallel so there is no latency penalty. Can activate multiple agents simultaneously — a question about a clinical trial also gets research context automatically.

**Research Agent**
The main knowledge engine. Runs hybrid retrieval against 50,000+ PubMed abstracts and NCCN/ASCO clinical guidelines. Generates responses calibrated to the selected literacy mode. This is where questions about treatments, biology, prognosis, and mechanisms get answered.

**Clinical Trial Agent**
Calls the live `clinicaltrials.gov` API with filters for recruiting GBM trials, phases II/III. Extracts location from natural language queries ("near Mumbai", "in Boston") and formats results in plain English — not the dense government jargon of the original listings.

**Drug Info Agent**
Queries the openFDA API for drug labels and cross-references the GBM drug ontology. Explains what a drug does, what side effects are common, and what questions to ask your oncologist — without ever specifying doses.

**Emotional Support Agent**
This is what makes GlioblastomaGPT different from a cold research tool.

When someone types *"I'm terrified. My mother was just diagnosed and I don't know what to do"* — they do not need clinical data. They need to be seen.

The emotional support agent detects when a query is driven by fear, grief, or exhaustion rather than a clinical question, and responds with warmth before anything else. It validates the experience without false optimism, connects people to real support resources (NBTS, ABTA, CancerCare, Crisis Text Line), and knows when to stay out of the way of the clinical agents.

**Synthesizer Agent**
When multiple agents respond, the synthesizer merges their outputs into a single coherent answer. It removes redundancies, resolves contradictions (newer/more specific sources win), structures the response clearly, and attaches inline citations. Emotional content always comes before clinical data when both are present.

**Safety Layer**
Non-negotiable. Every single response passes through it before reaching the user.

| Signal detected | Action |
|-----------------|--------|
| Dosage advice in the answer | Hard block — replaced with oncologist referral |
| Survival/prognosis statistics | Individual variation disclaimer appended |
| Crisis signals in the query | Crisis resources prepended immediately |
| Confidence score < 0.70 | "Limited evidence" warning added |
| Confidence score < 0.50 | Answer blocked entirely |
| Treatment/drug mentions | General medical disclaimer appended |

The safety layer uses regex pattern matching (not LLM-based) for the dosage and crisis detection, so it cannot be hallucinated away or prompt-injected out of existence.

---

## Data Sources

| Source | What it provides | How it is accessed |
|--------|-----------------|-------------------|
| PubMed (NCBI) | 50,000+ peer-reviewed GBM research abstracts | NCBI E-utilities API, async batch fetch |
| NCCN / ASCO Guidelines | Clinical treatment standards and guidelines | Ingested and chunked at setup |
| clinicaltrials.gov | Live recruiting GBM clinical trials | Live API call on every trial query |
| openFDA | Drug labels for GBM medications | Live API call on drug queries |
| GBM Ontology | 210 curated GBM terms, acronyms, synonyms | Local JSON file, loaded at startup |

---

## Project Structure

```
Glioblastoma/
├── .env.example                   ← Copy to .env and fill in your keys
├── requirements.txt               ← All dependencies
├── Dockerfile                     ← Container for API + UI
├── docker-compose.yml             ← One-command startup
└── gbm_copilot/
    ├── config.py                  ← All settings from environment variables
    │
    ├── ontology/
    │   ├── gbm_ontology.json      ← 210 GBM terms (acronyms, synonyms, genes, drugs)
    │   └── ontology_loader.py     ← Acronym/synonym expansion logic
    │
    ├── ingest/
    │   ├── pubmed_fetcher.py      ← Async NCBI E-utilities batch fetcher
    │   ├── clinical_trials_fetcher.py  ← clinicaltrials.gov API
    │   ├── fda_fetcher.py         ← FDA drug labels (openFDA)
    │   ├── chunker.py             ← Semantic chunking (256 tokens, 20% overlap)
    │   ├── contextual_retrieval.py ← Anthropic-style context prepend
    │   └── ingest.py              ← Full pipeline orchestrator
    │
    ├── embeddings/
    │   ├── embedder.py            ← BGE-M3 via fastembed
    │   ├── pinecone_store.py      ← Version A: Pinecone vector store
    │   └── numpy_store.py         ← Version B: NumPy in-memory store
    │
    ├── retrieval/
    │   ├── bm25_retriever.py      ← BM25 keyword retrieval
    │   ├── hybrid_retriever.py    ← BM25 + dense → RRF fusion
    │   └── query_expander.py      ← 3-stage medical query expansion
    │
    ├── agents/
    │   ├── state.py               ← LangGraph AgentState TypedDict
    │   ├── graph.py               ← LangGraph StateGraph — wires everything together
    │   ├── triage_agent.py        ← Query classification + medical expansion
    │   ├── research_agent.py      ← PubMed + guidelines RAG
    │   ├── clinical_trial_agent.py ← Live trial search and formatting
    │   ├── drug_agent.py          ← FDA labels + drug ontology
    │   ├── emotional_support_agent.py ← Compassionate response + resource routing
    │   ├── synthesizer_agent.py   ← Merge + cite + translate to literacy level
    │   └── safety_layer.py        ← Hard safety filter (non-negotiable)
    │
    ├── api/
    │   ├── main.py                ← FastAPI with streaming SSE support
    │   └── schemas.py             ← Pydantic request/response models
    │
    ├── ui/
    │   └── app.py                 ← Streamlit chat interface
    │
    ├── mcp_tools/
    │   └── server.py              ← FastMCP server (for Claude Desktop integration)
    │
    └── eval/
        ├── golden_set.json        ← 30 curated evaluation questions
        ├── ragas_eval.py          ← Ragas evaluation (faithfulness, relevancy, recall)
        └── ablation.py            ← Synonym expansion ablation study
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key
- Any email address (for PubMed API terms of service)

### Setup

```bash
# 1. Clone and create virtual environment
git clone https://github.com/yourusername/gbm-copilot
cd gbm-copilot
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the scispaCy biomedical NLP model (~800MB)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

# 4. Configure environment
cp .env.example .env
# Open .env and set:
#   OPENAI_API_KEY=sk-...
#   NCBI_EMAIL=your@email.com
#   RETRIEVAL_MODE=numpy     ← start with numpy (no external DB needed)

# 5. Ingest data (quick mode: ~500 PubMed abstracts, takes ~5 minutes)
python -m gbm_copilot.ingest.ingest --quick

# 6. Start the API and UI
uvicorn gbm_copilot.api.main:app --reload &
streamlit run gbm_copilot/ui/app.py
```

Open **http://localhost:8501** and start asking questions.

### Docker (One Command)

```bash
cp .env.example .env   # Fill in OPENAI_API_KEY and NCBI_EMAIL
docker-compose up
```

| Service | URL |
|---------|-----|
| Chat UI (Streamlit) | http://localhost:8501 |
| REST API (FastAPI) | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| MCP Server | http://localhost:8001 |

---

## Environment Variables

Minimum required to run:

```env
OPENAI_API_KEY=sk-...          # Required — GPT-4o for all agents
NCBI_EMAIL=your@email.com      # Required — any email for PubMed API terms
RETRIEVAL_MODE=numpy           # numpy (local) or pinecone (cloud)
```

Optional for full features:

```env
PINECONE_API_KEY=pcsk_...      # Only if RETRIEVAL_MODE=pinecone
LANGCHAIN_API_KEY=ls__...      # LangSmith tracing (recommended for debugging)
LANGCHAIN_TRACING_V2=true
OPENAI_MODEL=gpt-4o            # Default, change to gpt-4o-mini to reduce cost
MAX_PUBMED_RESULTS=50000       # Full ingest (takes ~45 min first run)
```

See `.env.example` for the complete list with descriptions.

---

## API Reference

The FastAPI backend supports both standard JSON and streaming (Server-Sent Events).

```bash
# Ask a question (standard)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the Stupp protocol and does MGMT methylation affect how well it works?",
    "literacy_mode": "patient"
  }'

# Ask a question (streaming — tokens appear as they are generated)
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Are there any open clinical trials for recurrent GBM near Mumbai?", "literacy_mode": "caregiver"}'

# Find clinical trials directly
curl "http://localhost:8000/trials?location=Boston&phase=2&max_results=5"

# Health check
curl http://localhost:8000/health

# Trigger a new ingest (quick mode)
curl -X POST http://localhost:8000/ingest -d '{"quick_mode": true}'
```

### Response Format

```json
{
  "answer": "The Stupp protocol is the standard first-line treatment for GBM...",
  "query_type": "treatment",
  "literacy_mode": "patient",
  "confidence_score": 0.87,
  "citations": [
    {
      "title": "Radiotherapy plus concomitant and adjuvant temozolomide for glioblastoma",
      "url": "https://pubmed.ncbi.nlm.nih.gov/15758009/",
      "source": "pubmed",
      "year": "2005",
      "pmid": "15758009"
    }
  ],
  "safety_flags": [],
  "is_blocked": false
}
```

---

## MCP Tools

GlioblastomaGPT exposes two tools via FastMCP that can be used with Claude Desktop or any MCP-compatible client:

```bash
# Start the MCP server
python -m gbm_copilot.mcp_tools.server
```

**`search_clinical_trials`** — Find recruiting GBM trials by location and phase
**`lookup_drug_info`** — Get plain-English drug information from FDA labels

---

## Evaluation

The system is evaluated against a 30-question golden set covering all five query types and all three literacy modes.

```bash
# Full evaluation with Ragas (faithfulness, answer_relevancy, context_recall)
python -m gbm_copilot.eval.ragas_eval --golden gbm_copilot/eval/golden_set.json

# Quick evaluation without Ragas (keyword coverage only — faster)
python -m gbm_copilot.eval.ragas_eval --no-ragas --max 10

# Run the synonym expansion ablation study
python -m gbm_copilot.eval.ablation
```

**Target metrics after full ingest:**

| Metric | Target |
|--------|--------|
| Ragas Faithfulness | > 0.70 |
| Answer Relevancy | > 0.75 |
| Context Recall | > 0.65 |
| Keyword Coverage | > 70% |
| Acronym Query Improvement | +25–35% vs no expansion |

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Agent orchestration | LangGraph | Stateful multi-agent graphs with parallel fan-out |
| LLM | GPT-4o (OpenAI) | Best-in-class medical reasoning and generation |
| Embeddings | BGE-M3 (fastembed) | Strong on biomedical text, runs locally |
| Vector store | Pinecone or NumPy | Flexible — cloud scale or zero-dependency local |
| Keyword retrieval | BM25 (rank-bm25) | Precise matching for drug names, gene names, IDs |
| Medical NLP | scispaCy en_core_sci_lg | Biomedical NER trained on PubMed |
| API | FastAPI + SSE | Async, streaming, production-ready |
| UI | Streamlit | Fast to build, good for chat interfaces |
| MCP | FastMCP | Claude Desktop integration |
| Tracing | LangSmith | End-to-end agent observability |
| Evaluation | Ragas | RAG-specific evaluation metrics |

---

## Safety and Disclaimers

GlioblastomaGPT is built for education and research. It is not a medical device and it does not give medical advice.

Every response, without exception, passes through a safety layer that:

1. **Blocks all dosage and prescribing information** — replaced with an oncologist referral
2. **Contextualizes survival statistics** — adds individual variation context to every prognosis number
3. **Routes crisis signals immediately** — crisis resources are prepended before anything else
4. **Flags low-confidence answers** — rather than presenting uncertain information as fact
5. **Blocks very low confidence answers entirely** — if confidence drops below 0.50, the system says so

The safety layer uses hard-coded regex patterns for dosage and crisis detection — not LLM calls — so it cannot be hallucinated past or prompt-injected around.

**Always consult your neuro-oncology team for all medical decisions.**

---

## Support Resources

If you or someone you care about is navigating GBM, these organizations exist specifically to help:

- **National Brain Tumor Society** — braintumor.org — 1-800-934-CURE
- **American Brain Tumor Association** — abta.org — 1-800-886-2282
- **CancerCare** — cancercare.org — 1-800-813-4673 (free counseling and support groups)
- **Caregiver Action Network** — caregiveraction.org
- **Crisis Text Line** — Text HOME to 741741 (24/7)
- **988 Suicide & Crisis Lifeline** — Call or text 988 (24/7)

---

## License

MIT License

---

*Built for everyone sitting in a hospital room at 2am, trying to understand what's happening and what comes next. You are not alone.*
