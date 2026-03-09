"""
Emotional Support Agent
Recognizes when queries are about fear, grief, or uncertainty rather than clinical info.
Responds with warmth, validates the emotional experience, and routes to support resources.

This is what separates GlioblastomaGPT from a cold research tool.
"""
from __future__ import annotations

from gbm_copilot.llm_client import get_client, get_model
from gbm_copilot.config import OPENAI_API_KEY, OPENAI_MODEL
from gbm_copilot.agents.state import AgentState

SUPPORT_RESOURCES = [
    {
        "name": "National Brain Tumor Society",
        "description": "Support groups, caregiver resources, and patient navigation services",
        "url": "https://braintumor.org",
        "phone": "1-800-934-CURE",
    },
    {
        "name": "ABTA (American Brain Tumor Association)",
        "description": "Patient guides, support communities, and clinical trial matching",
        "url": "https://www.abta.org",
        "phone": "1-800-886-2282",
    },
    {
        "name": "Ben & Catherine Ivy Foundation",
        "description": "GBM-specific patient support and research funding",
        "url": "https://www.ivyfoundation.org",
        "phone": None,
    },
    {
        "name": "CancerCare",
        "description": "Free counseling, support groups, and financial assistance for cancer patients",
        "url": "https://www.cancercare.org",
        "phone": "1-800-813-4673",
    },
    {
        "name": "Caregiver Action Network",
        "description": "Resources specifically for family caregivers",
        "url": "https://caregiveraction.org",
        "phone": None,
    },
    {
        "name": "Crisis Text Line",
        "description": "24/7 crisis support via text message",
        "url": "https://www.crisistextline.org",
        "phone": "Text HOME to 741741",
    },
]

EMOTIONAL_SYSTEM_PROMPT = """You are a compassionate companion for families facing glioblastoma (GBM).
The person you are talking to may be a patient, family member, or caregiver going through one of the hardest 
experiences a family can face.

Your role right now is NOT to provide medical information — it is to:
1. Acknowledge their feelings with genuine warmth and validation
2. Normalize their emotional experience (fear, grief, exhaustion, uncertainty are all valid)
3. Gently offer that there IS a community of people who understand
4. Provide practical next steps they can take (e.g., support groups, counseling)
5. End on a note of authentic hope — not false optimism, but the truth that they are not alone

Tone: warm, human, present. Not clinical. Not hollow. Like a trusted friend who has been through this.
Do NOT: use medical jargon, give clinical advice, or minimize their experience.
Do NOT: say "I understand how you feel" — instead acknowledge what they said specifically.
Limit: 3-4 paragraphs maximum. Every word should count."""


async def emotional_support_agent(state: AgentState) -> dict:
    """
    Emotional support agent: responds with warmth and resource routing.
    """
    query = state["query"]

    _client = get_client()
    _model = get_model()
    response = await _client.chat.completions.create(
        model=_model,
        messages=[
            {"role": "system", "content": EMOTIONAL_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        max_tokens=600,
        temperature=0.7,
    )

    answer = response.choices[0].message.content.strip()

    return {
        "emotional_answer": answer,
        "emotional_resources": SUPPORT_RESOURCES,
    }


def is_emotional_query(query: str) -> bool:
    """
    Quick heuristic check for emotional content.
    Used to optionally route to emotional support in parallel.
    """
    emotional_keywords = [
        "scared", "afraid", "fear", "terrified", "devastated", "hopeless", "hopeless",
        "crying", "can't stop", "don't know what to do", "lost", "broken",
        "giving up", "no hope", "miracle", "pray", "god", "why",
        "how do i cope", "how do i deal", "how do we tell", "telling the children",
        "how long", "will he die", "is she going to die", "can they survive",
        "suffering", "pain", "exhausted", "caregiver burnout",
        "i'm so", "we're so", "i feel", "we feel",
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in emotional_keywords)
