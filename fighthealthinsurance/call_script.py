"""What to say on the phone: a script for calling the insurer (issue #568).

People freeze on the phone with their insurer, whose reps are trained for
the call. The script gives them the words for one of three goals (check
that the insurer has their appeal, find out why it was denied, or push
back), the questions to ask in order, what to write down, what to say when
the rep deflects, and a call-notes sheet that prints with it.

The script is a fixed template, so the parts every call needs are always
there whatever the models do: the rep's name and a call reference number
first, whether AI or automation was involved, and written confirmation at
the end. For a case, our models may rewrite two short parts of it ("why
I'm calling" and a few questions about the reason and what was denied);
when they can't, or what they write fails the checks below, the template's
own text stays. Without a case (the standalone form) no model is asked.

Nothing is stored: the page is built from the case, or the form, each time.
We never call anyone; the person makes the call.
"""

import asyncio
import re
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from loguru import logger

from fighthealthinsurance.chat.safety_filters import detect_false_promises
from fighthealthinsurance.ml import llm_usage
from fighthealthinsurance.ml.bad_output_utils import (
    is_bad_output,
    strip_boilerplate_service,
)
from fighthealthinsurance.ml.denial_triage import (
    CATEGORIES,
    DENIAL_TYPE_FOR_CATEGORY,
)
from fighthealthinsurance.ml.ml_inference import infer_with_fallback
from fighthealthinsurance.ml.ml_models import RemoteModelLike
from fighthealthinsurance.ml.ml_router import ml_router
from fighthealthinsurance.models import Denial
from fighthealthinsurance.utils import aget_related

GOAL_CHECK_STATUS = "check_status"
GOAL_FIND_OUT_WHY = "find_out_why"
GOAL_ESCALATE = "escalate"
GOALS: tuple[tuple[str, str], ...] = (
    (GOAL_CHECK_STATUS, "Check they got my appeal"),
    (GOAL_FIND_OUT_WHY, "Find out why it was denied"),
    (GOAL_ESCALATE, "Push back / escalate"),
)
GOAL_LABELS = dict(GOALS)

# What each goal is, for the model, which only writes part of the script.
_GOAL_FOR_MODEL = {
    GOAL_CHECK_STATUS: "check that the insurer received the patient's appeal "
    "and find out where it stands",
    GOAL_FIND_OUT_WHY: "find out exactly why the claim was denied and what the "
    "patient needs for an appeal",
    GOAL_ESCALATE: "escalate the denial: reach a supervisor, ask for a "
    "peer-to-peer review or file a formal appeal",
}

# The denial reasons are the triage categories (ml/denial_triage.py), in
# words a person picks from on the form.
REASON_OTHER = "other"
REASON_LABELS: dict[str, str] = {
    "medical_necessity": "Not medically necessary",
    "prior_authorization": "No prior authorization",
    "out_of_network": "Out of network",
    "not_a_covered_benefit": "Not covered by my plan",
    "experimental_investigational": "Called experimental or unproven",
    "coding_or_billing": "A billing or coding problem",
    "eligibility": "They say I wasn't covered (eligibility)",
    REASON_OTHER: "Something else, or I'm not sure",
}

# "My claim was denied ..." for each reason; nothing for "other", whose
# reason the script doesn't know.
_DENIED_AS = {
    "medical_necessity": "as not medically necessary",
    "prior_authorization": "for not having prior authorization",
    "out_of_network": "as out of network",
    "not_a_covered_benefit": "as not covered by my plan",
    "experimental_investigational": "as experimental",
    "coding_or_billing": "because of a billing or coding problem",
    "eligibility": "because you show me as not covered",
}

# The questions about the reason itself, asked when no model has tailored
# them (and every time on the standalone page).
REASON_QUESTIONS: dict[str, tuple[str, ...]] = {
    "medical_necessity": (
        "What would the policy need to see in my records to approve this?",
        "Was this reviewed by a doctor, and in which specialty?",
    ),
    "prior_authorization": (
        "Did you receive a prior authorization request from my provider, and when?",
        "Can my provider still ask for authorization now, after the fact?",
    ),
    "out_of_network": (
        "Is there an in-network provider who can do this, and how soon?",
        "Can this be covered at the in-network rate if no in-network provider "
        "is available?",
    ),
    "not_a_covered_benefit": (
        "Where in my plan documents does it say this isn't covered? Can you "
        "read me the wording?",
        "Is there an exception process for something the plan excludes?",
    ),
    "experimental_investigational": (
        "What evidence or criteria did you use to decide this is experimental?",
        "Which medical policy covers this, and when was it last updated?",
    ),
    "coding_or_billing": (
        "Which code or billing detail caused the denial?",
        "If my provider fixes it and sends the claim again, will it be "
        "processed again?",
    ),
    "eligibility": (
        "What dates do you show my coverage as active?",
        "What do you need from me to fix my eligibility?",
    ),
    REASON_OTHER: (
        "Can you explain the reason for the denial in plain words?",
        "What would need to change for this to be approved?",
    ),
}

# The issue's own example question, asked on every call.
AI_QUESTION = (
    "Was any automated tool, algorithm or AI involved in this decision? Who "
    "reviewed it, and what are their credentials?"
)
URGENT_QUESTION = (
    "This is urgent for my health. Does it qualify for an expedited (fast) appeal?"
)
# Reasons where a doctor-to-doctor review can change the answer.
_PEER_TO_PEER_REASONS = frozenset(
    {"medical_necessity", "prior_authorization", "experimental_investigational"}
)


@dataclass(frozen=True)
class Question:
    text: str
    # Worth writing down: the notes sheet has a row for the answer.
    note: bool = False
    # In the part our models may rewrite for a case.
    tailorable: bool = False


# How every call opens: who you are, then the rep's name and a reference
# number before anything else, so the call is on record from the start.
OPENING: tuple[Question, ...] = (
    Question("Hi, my name is ______. My member ID is ______."),
    Question(
        "Before we start, can I get your name and a reference number for this call?",
        note=True,
    ),
)

IF_THEY_SAY: tuple[tuple[str, str], ...] = (
    (
        "You'll need to ask your doctor.",
        "I'm asking about the plan's decision. Who at the plan can answer that?",
    ),
    (
        "I can't give you that.",
        "Please note in my file that I asked. How do I request it in writing?",
    ),
    ("It's in your letter.", "Could you read me the exact wording so I have it right?"),
    ("(You're stuck, or going in circles.)", "Can I speak with a supervisor, please?"),
)

WRAP_UP: tuple[str, ...] = (
    "Please add a note to my file about this call.",
    "Can you send me written confirmation of what we discussed?",
    "Just to confirm: your name is ______ and the reference number is ______.",
    "Thank you for your help.",
)

# The rows of the call-notes sheet: the same on every call, then the goal's own.
_NOTES_ROWS = (
    "Date and time",
    "Number called",
    "Rep's name (and ID)",
    "Call reference number",
)
_GOAL_NOTES_ROWS = {
    GOAL_CHECK_STATUS: (
        "Date they received my appeal",
        "When to expect a decision",
    ),
    GOAL_FIND_OUT_WHY: (
        "Policy or guideline used",
        "Was AI or automation involved?",
        "Appeal deadline, and how to send it",
    ),
    GOAL_ESCALATE: (
        "Supervisor's name",
        "Next step they agreed to",
    ),
}
_NOTES_TAIL = (
    "What they told me",
    "What happens next, and by when",
    "Transferred to",
)

# Longest procedure the script repeats back; the form's field is capped here too.
PROCEDURE_MAX = 120
INSURER_MAX = 200
_FALLBACK_INSURER = "your insurance company"


@dataclass(frozen=True)
class CallScriptInputs:
    """What a script is built from: a case, or the standalone form."""

    goal: str
    insurer_name: str = ""
    reason_key: str = REASON_OTHER
    procedure: str = ""
    # The insurer's appeals line from our own records, never a model's.
    phone: str = ""
    claim_id: str = ""
    urgent: bool = False
    # The person's consent to outside models (Denial.use_external).
    use_external: bool = False


@dataclass(frozen=True)
class Tailoring:
    """The two parts a model wrote, after the checks passed."""

    why_calling: str
    questions: tuple[str, ...]


@dataclass(frozen=True)
class Script:
    inputs: CallScriptInputs
    goal_label: str
    reason_label: str
    insurer: str
    before_you_call: tuple[str, ...]
    opening: tuple[Question, ...]
    why_calling: str
    questions: tuple[Question, ...]
    if_they_say: tuple[tuple[str, str], ...]
    wrap_up: tuple[str, ...]
    notes_rows: tuple[str, ...]

    @property
    def summary(self) -> str:
        """The line under the page's title: insurer, reason, what was denied."""
        parts = [self.insurer[:1].upper() + self.insurer[1:], self.reason_label]
        if self.inputs.procedure:
            parts.append(self.inputs.procedure)
        return " · ".join(parts)


def _one_line(value: Optional[str], limit: int) -> str:
    return " ".join((value or "").split())[:limit]


def _insurer(inputs: CallScriptInputs) -> str:
    return inputs.insurer_name or _FALLBACK_INSURER


def why_calling(inputs: CallScriptInputs) -> str:
    """The template's own "why I'm calling", for when no model wrote one."""
    claim = f"My claim for {inputs.procedure}" if inputs.procedure else "My claim"
    denied_as = _DENIED_AS.get(inputs.reason_key)
    denied = f"was denied {denied_as}" if denied_as else "was denied"
    if inputs.goal == GOAL_CHECK_STATUS:
        what = f"for {inputs.procedure}" if inputs.procedure else "of a denied claim"
        return (
            f"I sent an appeal {what}, and I'm calling to make sure you "
            f"received it and to check on its status."
        )
    if inputs.goal == GOAL_ESCALATE:
        return f"{claim} {denied}, and I'd like to escalate it."
    return (
        f"{claim} {denied}. I'd like to understand exactly why, so I can "
        f"decide what to do next."
    )


def _goal_questions(inputs: CallScriptInputs) -> list[Question]:
    """The goal's questions in order; the tailorable ones where the reason's go."""
    reason = [
        Question(text, tailorable=True)
        for text in REASON_QUESTIONS.get(
            inputs.reason_key, REASON_QUESTIONS[REASON_OTHER]
        )
    ]
    urgent = [Question(URGENT_QUESTION)] if inputs.urgent else []
    if inputs.goal == GOAL_CHECK_STATUS:
        return [
            Question(
                "I sent an appeal on ______ by ______ (fax, mail or online). "
                "Can you confirm you received it, and on what date?",
                note=True,
            ),
            Question(
                "Has it been assigned to a reviewer? When should I expect a decision?",
                note=True,
            ),
            Question("Is anything missing from my appeal file?"),
            *reason,
            Question(
                "If you can't find it, where should I send it again: what fax "
                "number or address?",
                note=True,
            ),
            Question(AI_QUESTION, note=True),
            *urgent,
            Question(
                "Can you send me written confirmation that you received my appeal?"
            ),
        ]
    if inputs.goal == GOAL_ESCALATE:
        peer = (
            [
                Question(
                    "I'd like my doctor to have a peer-to-peer review with your "
                    "medical director. How does my doctor set that up?",
                    note=True,
                )
            ]
            if inputs.reason_key in _PEER_TO_PEER_REASONS
            else []
        )
        return [
            Question(
                "I'd like to escalate this denial. Can I speak with a supervisor "
                "or someone in the appeals department?",
                note=True,
            ),
            Question(AI_QUESTION, note=True),
            *peer,
            Question(
                "I want to file a formal appeal. Can you note that in my file, "
                "and tell me exactly how to send it?",
                note=True,
            ),
            *urgent,
            *reason,
            Question(
                "If my appeal is denied, how do I ask for an independent external review?",
                note=True,
            ),
        ]
    return [
        Question(
            "Can you confirm the claim number and date of service you're looking at?"
        ),
        Question("What exact reason is listed for this denial?", note=True),
        Question(
            "Which medical policy or clinical guideline was used? What is its "
            "name or number?",
            note=True,
        ),
        Question(AI_QUESTION, note=True),
        *reason,
        Question(
            "What is my deadline to appeal, and how do I send it: fax, mail or online?",
            note=True,
        ),
        *urgent,
        Question("Can you send me a copy of the policy you used and my claim file?"),
    ]


def _before_you_call(inputs: CallScriptInputs) -> tuple[str, ...]:
    if inputs.phone:
        number = (
            f"Call {inputs.phone}, {_insurer(inputs)}'s appeals line from our "
            f"records. Check it against your card or letter."
        )
    else:
        number = "Call the member services number on the back of your insurance card."
    lines = [
        "Have your denial letter, your insurance card and this page in front of you.",
        number,
    ]
    if inputs.claim_id:
        lines.append(
            f"Your claim number, as we read it from your letter: {inputs.claim_id}."
        )
    lines.append("Write down what they tell you on the call notes sheet.")
    return tuple(lines)


def build_script(
    inputs: CallScriptInputs, tailoring: Optional[Tailoring] = None
) -> Script:
    """The script for ``inputs``, with ``tailoring`` in place of the template's
    own "why I'm calling" and reason questions when a model wrote them."""
    questions = _goal_questions(inputs)
    if tailoring is not None:
        # Every goal has the reason's questions; the tailored ones take their
        # place (or close the list, should a goal ever have none).
        at = next((i for i, q in enumerate(questions) if q.tailorable), len(questions))
        kept = [q for q in questions if not q.tailorable]
        tailored = [Question(text, tailorable=True) for text in tailoring.questions]
        questions = kept[:at] + tailored + kept[at:]
    return Script(
        inputs=inputs,
        goal_label=GOAL_LABELS[inputs.goal],
        reason_label=REASON_LABELS.get(inputs.reason_key, REASON_LABELS[REASON_OTHER]),
        insurer=_insurer(inputs),
        before_you_call=_before_you_call(inputs),
        opening=OPENING,
        why_calling=tailoring.why_calling if tailoring else why_calling(inputs),
        questions=tuple(questions),
        if_they_say=IF_THEY_SAY,
        wrap_up=WRAP_UP,
        notes_rows=_NOTES_ROWS + _GOAL_NOTES_ROWS[inputs.goal] + _NOTES_TAIL,
    )


def inputs_from_form(cleaned: Mapping[str, Any]) -> CallScriptInputs:
    """The standalone form's answers. No consent was asked, so never outside models."""
    reason = cleaned.get("reason") or REASON_OTHER
    return CallScriptInputs(
        goal=cleaned["goal"],
        insurer_name=_one_line(cleaned.get("insurer"), INSURER_MAX),
        reason_key=reason if reason in REASON_LABELS else REASON_OTHER,
        procedure=_one_line(cleaned.get("procedure"), PROCEDURE_MAX),
    )


def reason_for_denial(denial: Denial, denial_type_names: Iterable[str]) -> str:
    """The case's reason key: the triage category, else the first reason the
    person ticked that maps to one, else "other"."""
    if denial.triage_category in CATEGORIES:
        return str(denial.triage_category)
    by_type = {name: key for key, name in DENIAL_TYPE_FOR_CATEGORY.items()}
    for name in denial_type_names:
        if name in by_type:
            return by_type[name]
    return REASON_OTHER


# What strip_boilerplate_service can leave of "the above referenced service".
_LEFTOVER = re.compile(r"^(?:the|a|an|this|that|your|my)$", re.IGNORECASE)


def _procedure(denial: Denial) -> str:
    """What was denied, as the script repeats it back; empty when extraction
    read only a letter's boilerplate ("the above referenced service")."""
    procedure = _one_line(denial.value_or_candidate("procedure"), PROCEDURE_MAX)
    cleaned = strip_boilerplate_service(procedure) if procedure else None
    return "" if not cleaned or _LEFTOVER.match(cleaned) else cleaned


def inputs_from_denial(
    denial: Denial, goal: str, denial_type_names: Iterable[str]
) -> CallScriptInputs:
    """A case's script inputs. From async code, ``denial`` must come from
    acase_denial, which loads its insurer and plan, so this makes no query."""
    company = denial.insurance_company_obj
    plan = denial.insurance_plan_obj
    if plan is not None:
        destinations = plan.appeal_destinations()
    elif company is not None:
        destinations = company.appeal_destinations()
    else:
        destinations = {}
    insurer = company.name if company is not None else (denial.insurance_company or "")
    # Extraction writes UNKNOWN when it can't read the insurer.
    if insurer.strip().upper() == "UNKNOWN":
        insurer = ""
    return CallScriptInputs(
        goal=goal,
        insurer_name=_one_line(insurer, INSURER_MAX),
        reason_key=reason_for_denial(denial, denial_type_names),
        procedure=_procedure(denial),
        phone=_one_line(destinations.get("phone"), 40),
        claim_id=_one_line(denial.claim_id, 60),
        urgent=bool(denial.urgent),
        use_external=bool(denial.use_external),
    )


def case_fields(data: Mapping[str, Any]) -> dict[str, str]:
    """The case a post names, as the hidden fields that name it again: the
    send page's (denial_id, email, semi_sekret), or the follow-up page's
    (uuid, follow_up_semi_sekret). Empty when it names none, or names one in
    a shape no case could have (a mangled id is no case, not a 500)."""
    denial_id = str(data.get("denial_id") or "")
    email = str(data.get("email") or "")
    semi_sekret = str(data.get("semi_sekret") or "")
    if denial_id.isdigit() and email and semi_sekret:
        return {"denial_id": denial_id, "email": email, "semi_sekret": semi_sekret}
    follow_up_semi_sekret = str(data.get("follow_up_semi_sekret") or "")
    try:
        case_uuid = str(uuid.UUID(str(data.get("uuid") or "")))
    except ValueError:
        return {}
    if follow_up_semi_sekret:
        return {"uuid": case_uuid, "follow_up_semi_sekret": follow_up_semi_sekret}
    return {}


def _case_filter(fields: Mapping[str, str]) -> Optional[dict[str, Any]]:
    """The match for a case_fields() result: the one get_denial_for_action
    makes for the send page's fields, or FollowUpHelper.fetch_denial for the
    follow-up page's."""
    if "semi_sekret" in fields:
        return {
            "denial_id": int(fields["denial_id"]),
            "hashed_email": Denial.get_hashed_email(fields["email"]),
            "semi_sekret": fields["semi_sekret"],
        }
    if "follow_up_semi_sekret" in fields:
        return {
            "uuid": fields["uuid"],
            "follow_up_semi_sekret": fields["follow_up_semi_sekret"],
        }
    return None


def case_denial(fields: Mapping[str, str]) -> Optional[Denial]:
    """The case ``fields`` names, or None."""
    match = _case_filter(fields)
    return Denial.objects.filter(**match).first() if match else None


async def acase_denial(fields: Mapping[str, str]) -> Optional[Denial]:
    """case_denial on the native async ORM, with the insurer and plan that
    inputs_from_denial reads already loaded. Not select_related: a case with
    no insurer would join NULLs into InsuranceCompany's regex fields, which
    refuse them."""
    match = _case_filter(fields)
    denial = await Denial.objects.filter(**match).afirst() if match else None
    if denial is None:
        return None
    await aget_related(denial, "insurance_company_obj")
    plan = await aget_related(denial, "insurance_plan_obj")
    if plan is not None:
        await aget_related(plan, "insurance_company")
    return denial


def denial_type_names(denial: Denial) -> list[str]:
    return [denial_type.name for denial_type in denial.denial_type.all()]


async def adenial_type_names(denial: Denial) -> list[str]:
    return [denial_type.name async for denial_type in denial.denial_type.all()]


# --- Tailoring: our models rewrite two parts of a case's script. ---

# How many models one tailoring may ask: ours first, then (only with the
# person's consent to outside models) one outside model.
MAX_INTERNAL_ATTEMPTS = 3
MAX_EXTERNAL_ATTEMPTS = 1
# Each model's own timeout, and the whole tailoring's: the page already
# shows the template, so a slow model costs nothing but the wait.
MODEL_TIMEOUT_SECONDS = 20.0
TAILOR_BUDGET_SECONDS = 45.0

CALL_SCRIPT_SYSTEM_PROMPT = (
    "You write two short parts of a phone script that a patient reads aloud "
    "when they call their health insurance company about a denied claim. The "
    "rest of the script is already written.\n"
    "Rules:\n"
    "- Plain, polite, firm words the patient says, in the first person.\n"
    "- Use only the facts you are given. Never invent policy names or "
    "numbers, phone numbers, dates, deadlines, dollar amounts, laws or "
    "regulations.\n"
    "- Do not give legal or medical advice, threaten legal action, or "
    "promise any outcome.\n"
    "- Ask only questions an insurance representative could answer about "
    "this denial, and do not repeat a question the script already asks.\n"
    "Output exactly this, and nothing else:\n"
    "WHY I'M CALLING:\n"
    "<two or three sentences>\n"
    "QUESTIONS:\n"
    "- <question>\n"
    "- <question>\n"
    "with two to four questions, each ending with a question mark."
)


def make_call_script_prompt(inputs: CallScriptInputs) -> str:
    """The prompt for a case's tailoring: structured facts only. The denial
    letter, names, IDs and the diagnosis are never in it."""
    fixed = "\n".join(
        f"- {q.text}" for q in _goal_questions(inputs) if not q.tailorable
    )
    return (
        f"Goal of the call: {_GOAL_FOR_MODEL[inputs.goal]}.\n"
        f"Insurance company: {_insurer(inputs)}.\n"
        f"Reason for the denial: {CATEGORIES.get(inputs.reason_key, CATEGORIES[REASON_OTHER])}.\n"
        f"What was denied: {inputs.procedure or 'not given'}.\n"
        f"Questions the script already asks:\n{fixed}"
    )


_TAILORING = re.compile(
    r"WHY\s+I'?M\s+CALLING:\s*(?P<why>.+?)\s*QUESTIONS:\s*(?P<questions>.+)\Z",
    re.IGNORECASE | re.DOTALL,
)
_LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(?P<item>\S.*)$")
# A script makes no legal claims and cites nothing.
_LEGAL = re.compile(
    r"\b(?:sue|suing|lawsuit|lawyer|attorney|court|legal|illegal|unlawful"
    r"|violat\w*|statutes?|regulations?|laws?)\b"
    r"|\$|§|\bU\.?S\.?C\b|\bC\.?F\.?R\b",
    re.IGNORECASE,
)
_QUOTES = "\"'“”‘’"


def _grounded(text: str, prompt: str) -> bool:
    """Every number in ``text`` is one the prompt gave: a phone number, date,
    deadline, policy number or amount the model made up is a number the
    prompt doesn't have."""
    given = set(re.findall(r"\d+", prompt))
    return all(number in given for number in re.findall(r"\d+", text))


def parse_tailoring(text: Optional[str], prompt: str) -> Optional[Tailoring]:
    """A model's answer as a Tailoring, or None when it is the wrong shape or
    says something a script must not."""
    match = _TAILORING.search(text or "")
    if not match:
        return None
    why = " ".join(match.group("why").split()).strip(_QUOTES).strip()
    questions = []
    for line in match.group("questions").splitlines():
        item = _LIST_ITEM.match(line)
        if item:
            questions.append(
                " ".join(item.group("item").split()).strip(_QUOTES).strip()
            )
        elif line.strip():
            # Anything but list items after QUESTIONS: is chatter, not a script.
            return None
    if not 20 <= len(why) <= 500 or not 2 <= len(questions) <= 4:
        return None
    if any(len(q) > 240 or not q.endswith("?") for q in questions):
        return None
    whole = " ".join([why, *questions])
    if (
        not _grounded(whole, prompt)
        or _LEGAL.search(whole)
        or detect_false_promises(whole)
        or is_bad_output(whole, check_guardrail_phrases=True)
    ):
        return None
    return Tailoring(why_calling=why, questions=tuple(questions))


def tailor_backends(use_external: bool) -> list[RemoteModelLike]:
    """The models to ask, in order: ours, then one outside model only with
    the person's consent."""
    models: list[RemoteModelLike] = list(
        ml_router.general_purpose_internal_models()[:MAX_INTERNAL_ATTEMPTS]
    )
    if use_external:
        models += ml_router.best_external_models(limit=MAX_EXTERNAL_ATTEMPTS)
    return models


@llm_usage.for_denial(task="call_script")
async def tailor_script(
    inputs: CallScriptInputs, denial: Optional[Denial] = None
) -> Optional[Tailoring]:
    """The two tailored parts of a case's script, or None (the template
    stays). ``denial`` is the case, so the model calls count from its origin."""
    models = tailor_backends(inputs.use_external)
    if not models:
        return None
    prompt = make_call_script_prompt(inputs)
    try:
        text = await asyncio.wait_for(
            infer_with_fallback(
                system_prompts=[CALL_SCRIPT_SYSTEM_PROMPT],
                prompt=prompt,
                models=models,
                timeout=MODEL_TIMEOUT_SECONDS,
                label="call_script",
                validator=lambda answer: parse_tailoring(answer, prompt) is not None,
            ),
            timeout=TAILOR_BUDGET_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.info("call script: tailoring ran out of time; the template stays")
        return None
    return parse_tailoring(text, prompt)
