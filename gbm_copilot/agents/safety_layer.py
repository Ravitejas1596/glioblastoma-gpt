"""
HITL Safety Layer
Non-negotiable safety filter that runs on every answer before delivery.
Detects dosage advice, distress signals, and prognosis statements.
Adds disclaimers or blocks answers accordingly.
"""
from __future__ import annotations

import re

from gbm_copilot.config import MIN_CONFIDENCE_THRESHOLD, FAITHFULNESS_WARNING_THRESHOLD
from gbm_copilot.agents.state import AgentState

# Hard-coded regex patterns for dosage-related content
DOSAGE_PATTERNS = [
    r'\b\d+\s*mg\s*/\s*(?:m2|kg|day|dose)\b',
    r'\b(?:take|give|administer|prescribe)\s+\d+',
    r'\b(?:dose|dosage)\s+(?:is|should be|of)\s+\d+',
    r'\b\d+\s*mg\b.*\b(?:daily|twice|thrice|per day|bid|tid|qd)\b',
    r'\b(?:maximum|minimum)\s+(?:dose|dosage)\b',
]

PROGNOSIS_PATTERNS = [
    r'\bmedian(?: overall)? survival\b',
    r'\b\d+[-\s]month survival\b',
    r'\bsurvival rate\b',
    r'\blife expectancy\b',
    r'\bhow long .{0,20}(?:live|survive|has|have)\b',
    r'\bprognosis\b',
]

DISTRESS_PATTERNS = [
    r'\b(?:suicid|self.harm|end.it|don.t want to (?:live|go on))\b',
    r'\b(?:hopeless|worthless|can.t go on|giving up)\b',
]

ONCOLOGIST_DISCLAIMER = (
    "\n\n> ⚕️ **Medical Disclaimer**: This information is for educational purposes only. "
    "Always consult your oncologist or neuro-oncology team before making any treatment decisions. "
    "Dosing, protocols, and treatment choices must be individualized by your medical team."
)

PROGNOSIS_DISCLAIMER = (
    "\n\n> 📊 **About Statistics**: Survival statistics reflect population averages across many patients. "
    "Individual outcomes vary significantly based on molecular profile (MGMT, IDH status), age, "
    "performance status, and treatment response. Many patients live well beyond median survival. "
    "Your oncologist can give you a more personalized assessment."
)

LIMITED_EVIDENCE_WARNING = (
    "\n\n> ⚠️ **Limited Evidence**: The available medical literature has limited information on this "
    "specific question. This answer reflects what was found, but may not be complete."
)

BLOCKED_RESPONSE = (
    "I'm not able to provide specific dosage or prescribing information — this must come from your "
    "medical team. Your oncologist or neuro-oncologist has access to your full medical picture and "
    "can provide the right dosing for your specific situation.\n\n"
    "What I CAN help with: understanding how a medication works, what side effects to watch for, "
    "or helping you prepare questions for your next appointment."
)

CRISIS_RESPONSE = (
    "I hear you, and what you're feeling right now is real and valid.\n\n"
    "If you're in crisis right now, please reach out immediately:\n"
    "- **Crisis Text Line**: Text HOME to 741741\n"
    "- **988 Suicide & Crisis Lifeline**: Call or text 988\n"
    "- **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/\n\n"
    "You don't have to face this alone. There are people who understand what you're going through."
)


async def safety_layer(state: AgentState) -> dict:
    """
    Safety filter: analyze final answer for safety issues.
    Either blocks, modifies with disclaimers, or passes through.
    """
    answer = state.get("final_answer", "")
    query = state.get("query", "")
    confidence = state.get("confidence_score", 0.8)
    safety_flags: list[str] = []
    disclaimer_parts: list[str] = []
    is_blocked = False

    # 1. Distress/crisis signals in QUERY (immediate priority)
    if any(re.search(p, query, re.IGNORECASE) for p in DISTRESS_PATTERNS):
        safety_flags.append("distress_signal")
        # These get emergency response prepended
        answer = CRISIS_RESPONSE + "\n\n---\n\n" + answer

    # 2. Dosage advice in ANSWER (hard block)
    if any(re.search(p, answer, re.IGNORECASE) for p in DOSAGE_PATTERNS):
        safety_flags.append("dosage_advice")
        is_blocked = True
        answer = BLOCKED_RESPONSE

    # 3. Prognosis statistics (add context disclaimer)
    if any(re.search(p, answer, re.IGNORECASE) for p in PROGNOSIS_PATTERNS):
        safety_flags.append("prognosis_stats")
        disclaimer_parts.append(PROGNOSIS_DISCLAIMER)

    # 4. Medical treatment mentions (always add general disclaimer)
    medical_terms = ["treatment", "therapy", "clinical trial", "chemotherapy", "radiation",
                     "surgery", "protocol", "medication", "drug"]
    if not is_blocked and any(t in answer.lower() for t in medical_terms):
        disclaimer_parts.append(ONCOLOGIST_DISCLAIMER)

    # 5. Low confidence warning
    if not is_blocked and confidence < FAITHFULNESS_WARNING_THRESHOLD:
        safety_flags.append("low_confidence")
        disclaimer_parts.append(LIMITED_EVIDENCE_WARNING)

    # 6. Block entirely if confidence too low
    if confidence < MIN_CONFIDENCE_THRESHOLD and not is_blocked:
        safety_flags.append("confidence_blocked")
        is_blocked = True
        answer = ("I don't have enough reliable information to answer this question confidently. "
                  "Please consult your neuro-oncology team or a resource like the National Brain "
                  "Tumor Society (braintumor.org) for more specific guidance.")

    # Append all disclaimers
    if disclaimer_parts and not is_blocked:
        answer = answer + "\n".join(set(disclaimer_parts))

    return {
        "final_answer": answer,
        "safety_flags": safety_flags,
        "safety_disclaimer": "\n".join(set(disclaimer_parts)),
        "is_blocked": is_blocked,
    }
