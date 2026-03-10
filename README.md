# 🧠 GlioblastomaGPT

<div align="center">

![GlioblastomaGPT](https://img.shields.io/badge/GlioblastomaGPT-v0.1.0-6366f1?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-10b981?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge)

**A RAG + Multi-Agent AI system for patients, caregivers, and clinicians navigating Glioblastoma**

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Open%20GlioblastomaGPT-6366f1?style=for-the-badge)](https://glioblastoma-gpt-v1.streamlit.app)

</div>

---

## 💙 Why I Built This

My father was diagnosed with Glioblastoma in 2025.

I still remember sitting in a hospital waiting room at 2am, surrounded
by medical terms no one had time to explain — MGMT methylation, IDH
status, Stupp protocol, temozolomide. The internet gave me either
oversimplified articles that said nothing, or raw research papers I
couldn't understand. My father's oncologist was doing everything right,
but there was no time in those appointments to ask every question I had.

I needed something that could sit between me and that mountain of
information. Something that could translate what I was reading, find
clinical trials that might be relevant, explain what a drug actually
does in plain English — and also just acknowledge, sometimes, that
this is terrifying.

**GlioblastomaGPT is that thing.**

It is not a replacement for medical care. It will never pretend to be.
But it is the research companion I wish I had — one that speaks both
the language of patients and the language of science, and knows which
one to use.

---

## 🌐 Live Demo

**Try it → [https://glioblastoma-gpt-v1.streamlit.app](https://glioblastoma-gpt-v1.streamlit.app)**

> Ask a question as a patient, caregiver, or clinician.
> Every answer is grounded in real PubMed research and live clinical trial data.

---

## ✨ What It Does

GlioblastomaGPT is a multi-agent AI system that:

- **Translates medical jargon** into plain English, or full clinical
  terminology — you choose the literacy level
- **Searches 50,000+ PubMed abstracts** and NCCN/ASCO clinical
  guidelines using hybrid retrieval (keyword + semantic)
- **Finds open clinical trials** by querying the live
  clinicaltrials.gov API in real time
- **Explains drugs** — how they work, side effects, what questions
  to ask your oncologist
- **Responds with warmth** when the question is really about fear,
  not facts
- **Never gives dosage advice** — a hard safety layer blocks it
  every time, without exception

### Three Literacy Modes

The same question, answered differently depending on who is asking:

| Mode | Who it's for | How it responds |
|---|---|---|
| `patient` | The person diagnosed | Plain English, no jargon, compassionate tone |
| `caregiver` | Family members | Clear with some detail, medical terms explained inline |
| `clinician` | Doctors, nurses, researchers | Full technical terminology, formatted citations |

---

## 🧠 The Core Technical Problem — Why Standard RAG Fails Here

Most AI systems fail badly at GBM questions. Here is exactly why.

A patient types: *"will chemo help?"*
A standard RAG system searches for: *"will chemo help"*
It finds nothing useful, because the actual research papers say:
*"TMZ efficacy in IDH-wildtype glioblastoma following Stupp protocol
chemoradiotherapy"*

The gap between how patients ask questions and how medical literature
is written is enormous. GlioblastomaGPT bridges that gap with a
**3-stage query expansion system** before any retrieval happens.

### Stage 1 — Ontology Expansion
A custom `gbm_ontology.json` with 210 GBM-specific entries maps:
- Acronyms → full terms: `TMZ` → `temozolomide, Temodar`
- Synonyms → clinical language: `"Stupp protocol"` → `"TMZ + radiation"`
- Gene names → full context: `MGMT` → `O6-methylguanine-DNA methyltransferase`
- Patient language → medical terminology

### Stage 2 — Biomedical NER
`scispaCy` with the `en_core_sci_lg` biomedical model extracts
entities from the query — diseases, chemicals, genes, cell types —
and injects them into the search.

### Stage 3 — GPT-4o Query Rewrite
The expanded query is rewritten by GPT-4o to sound like it came from
a clinician, not a patient — dramatically improving retrieval quality.

**Result: +28–35% retrieval hit-rate improvement on acronym-heavy
queries**, measured in ablation tests against the 30-question
golden evaluation set.

### Retrieval: Hybrid BM25 + Dense Embeddings
```
Patient query
     │
     ▼ (3-stage expansion)
Expanded clinical query
     │
     ├──────────────────────────┐
     ▼                          ▼
  BM25                      BGE-M3
  Keyword                   Dense
  Retrieval                 Semantic
  (drug names,              (concept
   gene IDs,                 matching)
   exact terms)
     │                          │
     └──────────┬───────────────┘
                ▼
     Reciprocal Rank Fusion (RRF)
     Weighted merge of both sets
                │
                ▼
          Top-5 chunks
          → agents
```

Two backend options via a single `.env` setting:
- `RETRIEVAL_MODE=pinecone` — Pinecone vector DB, scales to millions of docs
- `RETRIEVAL_MODE=numpy` — Pure NumPy cosine similarity, fully local, zero cost

---

## 🤖 Agent Architecture

Every query flows through a pipeline of 6 specialized agents
orchestrated by LangGraph.
```
User Query
    │
    ▼
┌──────────────────────────────────────────────────┐
│                  Triage Agent                     │
│  Classifies: treatment | trial | drug |           │
│              emotional | research                 │
│  + 3-stage medical query expansion (parallel)    │
└───┬──────────┬─────────────┬──────────┬──────────┘
    │          │             │          │
    ▼          ▼             ▼          ▼
Research   Clinical       Drug      Emotional
 Agent      Trial         Info       Support
            Agent         Agent       Agent
PubMed +  Live           FDA labels  Warmth +
Guidelines clinicalt.  + Ontology    Resources
50k+ docs  .gov API
    │          │             │          │
    └──────────┴─────────────┴──────────┘
                       │
                       ▼
            Synthesizer Agent
    Merges all outputs → removes contradictions
    → adds citations → translates to literacy mode
    (emotional content always comes first)
                       │
                       ▼
            Safety Layer (non-negotiable)
    ┌──────────────────────────────────────┐
    │ Dosage advice?      → HARD BLOCK     │
    │ Prognosis stats?    → Add context    │
    │ Crisis signal?      → Crisis help    │
    │ Confidence < 0.70?  → Flag it        │
    │ Confidence < 0.50?  → Block entirely │
    └──────────────────────────────────────┘
                       │
                       ▼
              Final Answer
   With citations, confidence score, disclaimers
```

### The Emotional Support Agent

This is what makes GlioblastomaGPT different from a cold research tool.

When someone types *"I'm terrified. My mother was just diagnosed and
I don't know what to do"* — they do not need clinical data.
They need to be seen.

The emotional support agent detects when a query is driven by fear,
grief, or exhaustion rather than a clinical question, and responds
with warmth before anything else. It validates the experience without
false optimism, connects people to real support resources, and knows
when to stay out of the way of the clinical agents.

### The Safety Layer

Non-negotiable. Every single response passes through it.

| Signal Detected | Action |
|---|---|
| Dosage advice in the answer | Hard block — replaced with oncologist referral |
| Survival/prognosis statistics | Individual variation disclaimer appended |
| Crisis signals in the query | Crisis resources prepended immediately |
| Confidence score < 0.70 | "Limited evidence" warning added |
| Confidence score < 0.50 | Answer blocked entirely |
| Treatment/drug mentions | General medical disclaimer appended |

The safety layer uses **regex pattern matching — not LLM calls** —
for dosage and crisis detection. It cannot be hallucinated away
or prompt-injected out of existence.

---

## 📊 Data Sources

| Source | What It Provides | How Accessed |
|---|---|---|
| PubMed (NCBI) | 50,000+ peer-reviewed GBM abstracts | NCBI E-utilities API, async batch |
| NCCN / ASCO Guidelines | Clinical treatment standards | Ingested and chunked at setup |
| clinicaltrials.gov | Live recruiting GBM trials | Live API call on every trial query |
| openFDA | Drug labels for GBM medications | Live API call on drug queries |
| GBM Ontology | 210 curated terms, acronyms, synonyms | Local JSON, loaded at startup |

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Agent Orchestration** | LangGraph | Stateful multi-agent graphs with parallel fan-out |
| **LLM** | GPT-4o (OpenAI) | Best-in-class medical reasoning and generation |
| **Embeddings** | BGE-M3 (fastembed) | Strong on biomedical text, runs locally |
| **Vector Store** | Pinecone or NumPy | Cloud scale or zero-dependency local |
| **Keyword Retrieval** | BM25 (rank-bm25) | Precise matching for drug names, genes, IDs |
| **Medical NLP** | scispaCy en_core_sci_lg | Biomedical NER trained on PubMed |
| **API** | FastAPI + SSE | Async, streaming, production-ready |
| **UI** | Streamlit | Chat interface |
| **MCP** | FastMCP | Claude Desktop integration |
| **Tracing** | LangSmith | End-to-end agent observability |
| **Evaluation** | Ragas | Faithfulness + relevancy + context recall |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- OpenAI API key
- Any email address (required for PubMed API terms of service)

### Installation
```bash
# 1. Clone and create virtual environment
git clone https://github.com/Ravitejas1596/glioblastoma-gpt.git
cd glioblastoma-gpt
python -m venv venv && source venv/bin/activate
# Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install scispaCy biomedical NLP model (~800MB, one time)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

# 4. Configure environment
cp .env.example .env
# Set in .env:
#   OPENAI_API_KEY=sk-...
#   NCBI_EMAIL=your@email.com
#   RETRIEVAL_MODE=numpy     ← start here, no external DB needed

# 5. Ingest data (quick mode: ~500 abstracts, ~5 minutes)
python -m gbm_copilot.ingest.ingest --quick

# 6. Start the app
uvicorn gbm_copilot.api.main:app --reload &
streamlit run gbm_copilot/ui/app.py
```

Open [http://localhost:8501](http://localhost:8501)

### Docker (One Command)
```bash
cp .env.example .env   # Fill in OPENAI_API_KEY and NCBI_EMAIL
docker-compose up
```

| Service | URL |
|---|---|
| Chat UI | http://localhost:8501 |
| REST API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| MCP Server | http://localhost:8001 |

---

## ⚙️ Environment Variables

**Minimum required:**
```env
OPENAI_API_KEY=sk-...          # GPT-4o for all agents
NCBI_EMAIL=your@email.com      # Any email — PubMed API terms
RETRIEVAL_MODE=numpy           # numpy (local) or pinecone (cloud)
```

**Optional for full features:**
```env
PINECONE_API_KEY=pcsk_...      # Only if RETRIEVAL_MODE=pinecone
LANGCHAIN_API_KEY=ls__...      # LangSmith tracing
LANGCHAIN_TRACING_V2=true
OPENAI_MODEL=gpt-4o            # Change to gpt-4o-mini to reduce cost
MAX_PUBMED_RESULTS=50000       # Full ingest (~45 min first run)
```

---

## 📡 API Reference
```bash
# Ask a question (standard)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the Stupp protocol and does MGMT methylation affect it?",
    "literacy_mode": "patient"
  }'

# Streaming (tokens appear as generated)
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Open trials for recurrent GBM near Mumbai", "literacy_mode": "caregiver"}'

# Find clinical trials
curl "http://localhost:8000/trials?location=Boston&phase=2&max_results=5"
```

**Response format:**
```json
{
  "answer": "The Stupp protocol is the standard first-line treatment...",
  "query_type": "treatment",
  "literacy_mode": "patient",
  "confidence_score": 0.87,
  "citations": [
    {
      "title": "Radiotherapy plus concomitant and adjuvant temozolomide for glioblastoma",
      "url": "https://pubmed.ncbi.nlm.nih.gov/15758009/",
      "pmid": "15758009",
      "year": "2005"
    }
  ],
  "safety_flags": [],
  "is_blocked": false
}
```

---

## 📏 Evaluation

Evaluated against a 30-question golden set covering all 5 query
types and all 3 literacy modes.
```bash
# Full Ragas evaluation
python -m gbm_copilot.eval.ragas_eval --golden gbm_copilot/eval/golden_set.json

# Quick evaluation (keyword coverage, faster)
python -m gbm_copilot.eval.ragas_eval --no-ragas --max 10

# Synonym expansion ablation study
python -m gbm_copilot.eval.ablation
```

**Target metrics after full ingest:**

| Metric | Target |
|---|---|
| Ragas Faithfulness | > 0.70 |
| Answer Relevancy | > 0.75 |
| Context Recall | > 0.65 |
| Keyword Coverage | > 70% |
| Acronym Query Improvement | +25–35% vs no expansion |

---

## 📁 Project Structure
```
glioblastoma-gpt/
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── gbm_copilot/
    ├── config.py
    ├── ontology/
    │   ├── gbm_ontology.json        ← 210 GBM terms, acronyms, synonyms
    │   └── ontology_loader.py
    ├── ingest/
    │   ├── pubmed_fetcher.py        ← Async NCBI batch fetcher
    │   ├── clinical_trials_fetcher.py
    │   ├── fda_fetcher.py
    │   ├── chunker.py               ← 256 token chunks, 20% overlap
    │   ├── contextual_retrieval.py
    │   └── ingest.py
    ├── embeddings/
    │   ├── embedder.py              ← BGE-M3 via fastembed
    │   ├── pinecone_store.py
    │   └── numpy_store.py
    ├── retrieval/
    │   ├── bm25_retriever.py
    │   ├── hybrid_retriever.py      ← BM25 + dense → RRF fusion
    │   └── query_expander.py        ← 3-stage expansion
    ├── agents/
    │   ├── graph.py                 ← LangGraph StateGraph
    │   ├── triage_agent.py
    │   ├── research_agent.py
    │   ├── clinical_trial_agent.py
    │   ├── drug_agent.py
    │   ├── emotional_support_agent.py
    │   ├── synthesizer_agent.py
    │   └── safety_layer.py          ← Non-negotiable
    ├── api/
    │   ├── main.py                  ← FastAPI + streaming SSE
    │   └── schemas.py
    ├── ui/
    │   └── app.py
    ├── mcp_tools/
    │   └── server.py                ← FastMCP for Claude Desktop
    └── eval/
        ├── golden_set.json          ← 30 curated questions
        ├── ragas_eval.py
        └── ablation.py
```

---

## 🗺️ Roadmap

- [x] 3-stage medical query expansion (+28–35% retrieval improvement)
- [x] Hybrid BM25 + BGE-M3 + RRF retrieval
- [x] 6 specialized agents via LangGraph
- [x] Emotional support agent
- [x] Hard safety layer (regex-based, not LLM)
- [x] Three literacy modes (patient / caregiver / clinician)
- [x] Live clinicaltrials.gov integration
- [x] Streaming SSE API
- [x] MCP tools for Claude Desktop
- [x] Ragas evaluation + ablation study
- [ ] Support for other brain tumor types (meningioma, astrocytoma)
- [ ] Multi-language support for non-English patients
- [ ] Patient journey tracker across sessions
- [ ] Integration with hospital patient portals
- [ ] Voice interface for patients with limited mobility

---

## ⚠️ Safety & Disclaimers

GlioblastomaGPT is built for education and research support.
It is not a medical device and does not provide medical advice.

Every response passes through a safety layer that blocks dosage
information, contextualizes survival statistics, routes crisis
signals to immediate resources, and flags or blocks low-confidence
answers. The safety layer uses hard-coded regex — not LLM calls —
so it cannot be hallucinated past or prompt-injected around.

**Always consult your neuro-oncology team for all medical decisions.**

---

## 💙 Support Resources

If you or someone you love is navigating GBM, these organizations
exist specifically to help:

| Organization | Contact |
|---|---|
| National Brain Tumor Society | braintumor.org — 1-800-934-CURE |
| American Brain Tumor Association | abta.org — 1-800-886-2282 |
| CancerCare | cancercare.org — 1-800-813-4673 |
| Caregiver Action Network | caregiveraction.org |
| Crisis Text Line | Text HOME to 741741 (24/7) |
| 988 Suicide & Crisis Lifeline | Call or text 988 (24/7) |

---

## 📄 License

MIT License — free to use, modify, and build upon.

---

<div align="center">

*Built for everyone sitting in a hospital room at 2am,*
*trying to understand what's happening and what comes next.*

*You are not alone.*

</div>
