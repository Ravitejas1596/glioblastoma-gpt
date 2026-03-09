"""
Clinical Trial Agent
Performs live search of clinicaltrials.gov API and returns structured results.
Also extracts location info and formats trials for easy comprehension.
"""
from __future__ import annotations

import json
from openai import AsyncOpenAI

from gbm_copilot.config import OPENAI_API_KEY, OPENAI_MODEL
from gbm_copilot.agents.state import AgentState
from gbm_copilot.ingest.clinical_trials_fetcher import search_trials

_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

TRIAL_SYSTEM_PROMPT = """You are a clinical trial navigator for GBM (glioblastoma) patients.
Given a list of clinical trials and a user's question, provide a helpful summary.

Format each trial as:
**[Trial Name]** (NCT ID)
- Phase: [phase]
- What it tests: [brief description of the intervention in plain English]
- Who can join: [brief eligibility summary]
- Where: [locations - top 3]
- How to learn more: [URL]

End with a brief note on how to discuss trial eligibility with their oncologist.
Tone: hopeful but realistic, {literacy_mode} level."""


async def clinical_trial_agent(state: AgentState) -> dict:
    """
    Clinical trial agent: live API search + plain-English formatting.
    """
    query = state["query"]
    literacy = state.get("literacy_mode", "patient")

    # Extract location from query if mentioned
    location = _extract_location(query)

    # Live clinicaltrials.gov search
    try:
        trials = await search_trials(
            condition="glioblastoma",
            status="RECRUITING",
            location=location,
            max_results=20,
        )
    except Exception as e:
        trials = []

    state_update = {"trial_results": trials}

    if not trials:
        return {
            **state_update,
            "trial_answer": (
                "I searched clinicaltrials.gov but couldn't find recruiting GBM trials "
                "matching your criteria right now. The database is updated frequently — "
                "please check clinicaltrials.gov directly or ask your oncologist."
            ),
        }

    # Format trials as text for GPT-4o
    trial_text = json.dumps(
        [{k: v for k, v in t.items() if k not in {"eligibility", "summary"}} for t in trials[:10]],
        indent=2
    )

    response = await _openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": TRIAL_SYSTEM_PROMPT.format(literacy_mode=literacy)
            },
            {
                "role": "user",
                "content": f"User question: {query}\n\nTrials found:\n{trial_text}"
            },
        ],
        max_tokens=1200,
        temperature=0.3,
    )

    answer = response.choices[0].message.content.strip()
    return {**state_update, "trial_answer": answer}


def _extract_location(query: str) -> str | None:
    """Simple heuristic to extract city/state from query."""
    import re
    # Common patterns: "in Boston", "near New York", "at MD Anderson"
    patterns = [
        r'\bin\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?(?:,\s*[A-Z]{2})?)\b',
        r'\bnear\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b',
        r'\bat\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*(?:\s+(?:Hospital|Center|Cancer|Medical))?)\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return match.group(1)
    return None
