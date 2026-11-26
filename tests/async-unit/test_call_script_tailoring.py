"""Our models tailor two parts of a case's phone script (call_script.py).

The prompt carries only structured facts, never the letter or the
diagnosis. Outside models are asked only with the person's consent, and
after ours. What a model writes is used only when it is the right shape and
says nothing a script must not: a number the prompt didn't give (a made-up
phone number, deadline or policy number), a law or citation, a promised
outcome. Otherwise the next model is asked, and with none left the template
stays.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from django.test import SimpleTestCase

from fighthealthinsurance import call_script
from fighthealthinsurance.call_script import (
    GOAL_FIND_OUT_WHY,
    CallScriptInputs,
    make_call_script_prompt,
    parse_tailoring,
)

INPUTS = CallScriptInputs(
    goal=GOAL_FIND_OUT_WHY,
    insurer_name="Aetna",
    reason_key="medical_necessity",
    procedure="MRI of the lumbar spine",
)
PROMPT = make_call_script_prompt(INPUTS)

GOOD = (
    "WHY I'M CALLING:\n"
    "My MRI of the lumbar spine was denied as not medically necessary. I want "
    "to understand the decision.\n"
    "QUESTIONS:\n"
    "- Which clinical criteria did the reviewer apply to my MRI?\n"
    "- Did the reviewer see my doctor's notes about my symptoms?\n"
)


def _with(why=None, questions=None):
    why = why or "My MRI was denied and I want to understand the decision."
    questions = questions or [
        "Which clinical criteria did you apply?",
        "Who reviewed my MRI request?",
    ]
    return "WHY I'M CALLING:\n" + why + "\nQUESTIONS:\n" + "".join(
        f"- {q}\n" for q in questions
    )


def _model(name, reply, *, external=False):
    model = MagicMock()
    model.name = name
    model.external = external
    model._infer_no_context = AsyncMock(return_value=reply)
    return model


def _router(internal, external=()):
    return SimpleNamespace(
        general_purpose_internal_models=lambda: list(internal),
        best_external_models=lambda limit: list(external)[:limit],
    )


class PromptTest(SimpleTestCase):
    def test_the_prompt_holds_no_denial_letter_or_diagnosis(self):
        denial = SimpleNamespace(
            insurance_company_obj=None,
            insurance_plan_obj=None,
            insurance_company="Aetna",
            triage_category="medical_necessity",
            claim_id="CLM-0000",
            urgent=False,
            use_external=True,
            denial_text="Dear Jane Doe, your MRI is denied.",
            diagnosis="Lumbar radiculopathy",
            value_or_candidate=lambda field: {
                "procedure": "MRI",
                "diagnosis": "Lumbar radiculopathy",
            }[field],
        )
        prompt = make_call_script_prompt(
            call_script.inputs_from_denial(denial, GOAL_FIND_OUT_WHY, [])
        )
        self.assertFalse(
            any(
                secret in prompt
                for secret in ("Jane Doe", "Lumbar radiculopathy", "CLM-0000")
            )
        )

    def test_the_prompt_lists_the_questions_the_script_already_asks(self):
        self.assertIn(call_script.AI_QUESTION, PROMPT)


class ParseTest(SimpleTestCase):
    def test_a_well_formed_answer_is_used(self):
        tailoring = parse_tailoring(GOOD, PROMPT)
        self.assertEqual(
            tailoring.questions,
            (
                "Which clinical criteria did the reviewer apply to my MRI?",
                "Did the reviewer see my doctor's notes about my symptoms?",
            ),
        )

    def test_a_number_the_prompt_did_not_give_is_refused(self):
        answer = _with(why="My MRI was denied. Please call me back at 555-0100 today.")
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_a_law_or_citation_is_refused(self):
        answer = _with(why="My MRI was denied, and the law says you must cover it.")
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_a_threat_of_legal_action_is_refused(self):
        answer = _with(questions=["Should I get a lawyer?", "Who reviewed it?"])
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_a_promised_outcome_is_refused(self):
        answer = _with(why="My MRI was denied, but I am guaranteed approval on appeal.")
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_an_answer_without_the_two_parts_is_refused(self):
        self.assertIsNone(parse_tailoring("Sure! Here is a script for you.", PROMPT))

    def test_chatter_after_the_questions_is_refused(self):
        self.assertIsNone(parse_tailoring(GOOD + "I hope this helps!\n", PROMPT))

    def test_a_question_without_a_question_mark_is_refused(self):
        answer = _with(questions=["Tell me the criteria.", "Who reviewed it?"])
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_one_question_is_too_few(self):
        answer = _with(questions=["Which clinical criteria did you apply?"])
        self.assertIsNone(parse_tailoring(answer, PROMPT))


class BackendsTest(SimpleTestCase):
    def test_no_outside_model_is_asked_without_consent(self):
        ours = _model("ours", None)
        outside = _model("outside", None, external=True)
        with patch.object(call_script, "ml_router", _router([ours], [outside])):
            self.assertEqual(call_script.tailor_backends(use_external=False), [ours])

    def test_with_consent_the_outside_model_comes_after_ours(self):
        ours = _model("ours", None)
        outside = _model("outside", None, external=True)
        with patch.object(call_script, "ml_router", _router([ours], [outside])):
            self.assertEqual(
                call_script.tailor_backends(use_external=True), [ours, outside]
            )


class TailorTest(SimpleTestCase):
    def _tailor(self, router, inputs=INPUTS):
        with patch.object(call_script, "ml_router", router):
            return asyncio.run(call_script.tailor_script(inputs))

    def test_a_well_formed_answer_tailors_the_script(self):
        tailoring = self._tailor(_router([_model("ours", GOOD)]))
        self.assertTrue(tailoring.why_calling.startswith("My MRI of the lumbar spine"))

    def test_an_answer_that_fails_the_checks_moves_on_to_the_next_model(self):
        bad = _model("bad", _with(why="Call me at 555-0100, it was denied today."))
        good = _model("good", GOOD)
        tailoring = self._tailor(_router([bad, good]))
        self.assertIsNotNone(tailoring)

    def test_when_every_model_fails_the_template_stays(self):
        router = _router([_model("one", None), _model("two", "not a script")])
        self.assertIsNone(self._tailor(router))

    def test_with_no_model_to_ask_the_template_stays(self):
        self.assertIsNone(self._tailor(_router([])))
