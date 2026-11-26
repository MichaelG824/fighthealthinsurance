"""Our models tailor two parts of a case's phone script (call_script.py).

The prompt carries only structured facts, never the letter or the
diagnosis. Outside models are asked only with the person's consent, and
after ours. What a model writes is used only when it is the right shape and
says nothing a script must not: a number the prompt didn't give (a made-up
phone number or policy number), any mention of a deadline, even in a
question (the prompt never gives one and the script already asks for it, so
one in digits, in words or as a date is made up, even when its digits
appear in the procedure), a law or citation, a promised outcome. Otherwise
the next model is asked, and with none left the template stays.
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
from fighthealthinsurance.ml import spend

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


def _prompt_for(procedure):
    return make_call_script_prompt(
        CallScriptInputs(
            goal=GOAL_FIND_OUT_WHY,
            insurer_name="Aetna",
            reason_key="medical_necessity",
            procedure=procedure,
        )
    )


class DeadlineTest(SimpleTestCase):
    # Each procedure puts the deadline's digits or words in the prompt, so
    # matching against the prompt alone would let the invented deadline in.
    def test_a_deadline_whose_digit_is_in_the_procedure_is_refused(self):
        answer = _with(why="My appeal deadline is 4 days away.")
        self.assertIsNone(parse_tailoring(answer, _prompt_for("MRI L4-L5")))

    def test_a_deadline_the_procedure_repeats_word_for_word_is_refused(self):
        answer = _with(why="My appeal deadline is 12 weeks from the denial.")
        prompt = _prompt_for("Botox injections every 12 weeks")
        self.assertIsNone(parse_tailoring(answer, prompt))

    def test_a_due_date_the_procedure_repeats_is_refused(self):
        answer = _with(why="My appeal is due by March 3, so I need answers.")
        self.assertIsNone(parse_tailoring(answer, _prompt_for("MRI on March 3")))

    def test_a_number_of_days_in_words_is_refused(self):
        answer = _with(why="I have fourteen days to appeal this denial.")
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_a_compound_number_of_days_in_words_is_refused(self):
        answer = _with(why="I have one hundred eighty days to appeal this denial.")
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_a_due_weekday_is_refused(self):
        answer = _with(why="My appeal is due by Friday, so I need answers soon.")
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_due_today_is_refused(self):
        answer = _with(why="My appeal is due today, so I need answers right away.")
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_a_deadline_asserted_without_a_value_is_refused(self):
        answer = _with(why="My deadline is coming up and I need to understand this.")
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_a_month_given_as_the_last_chance_to_appeal_is_refused(self):
        answer = _with(why="I have until March to appeal this denial.")
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_a_month_to_file_before_is_refused(self):
        answer = _with(why="I need to file my appeal before March.")
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_the_end_of_a_month_is_refused(self):
        answer = _with(why="I must appeal by the end of March.")
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_a_deadline_to_appeal_asserted_without_a_value_is_refused(self):
        answer = _with(why="My deadline to appeal is coming up soon.")
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_a_deadline_for_the_appeal_asserted_without_a_value_is_refused(self):
        answer = _with(why="The deadline for my appeal is close.")
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_a_closing_appeal_window_is_refused(self):
        answer = _with(why="My appeal window is closing soon.")
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_the_middle_of_a_month_is_refused(self):
        answer = _with(why="I only have until the middle of March.")
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_mid_month_is_refused(self):
        answer = _with(why="I have until mid-March, so I need answers soon.")
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_early_in_a_month_is_refused(self):
        answer = _with(why="I have until early April to appeal.")
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_the_end_of_may_is_refused(self):
        answer = _with(why="I must appeal by the end of May.")
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_a_month_asked_about_is_refused(self):
        answer = _with(
            questions=["Is my appeal due in March?", "Who reviewed my MRI request?"]
        )
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_an_almost_closed_appeal_window_is_refused(self):
        answer = _with(why="My appeal window is almost closed.")
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_time_to_appeal_running_out_is_refused(self):
        answer = _with(why="My time to appeal runs out soon.")
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_asking_indirectly_what_the_deadline_for_the_appeal_is_is_refused(self):
        answer = _with(questions=['Can you tell me what the deadline for my appeal is?', "Who reviewed my MRI request?"])
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_asking_indirectly_what_the_deadline_to_file_is_is_refused(self):
        answer = _with(questions=['Could you confirm what the deadline to file is?', "Who reviewed my MRI request?"])
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_asking_what_the_deadline_is_is_refused(self):
        answer = _with(questions=['What is the deadline for my appeal?', "Who reviewed my MRI request?"])
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_asking_whether_there_is_a_deadline_is_refused(self):
        answer = _with(questions=['Is there a deadline for my doctor to send more records?', "Who reviewed my MRI request?"])
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_a_deadline_claim_inside_a_question_is_refused(self):
        answer = _with(questions=['My deadline is coming up, so can you expedite this?', "Who reviewed my MRI request?"])
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_a_bare_due_weekday_is_refused(self):
        answer = _with(why='My appeal is due Friday, so I need answers.')
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_before_a_weekday_is_refused(self):
        answer = _with(why='I must file before Friday.')
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_tomorrow_is_refused(self):
        answer = _with(why='I have to send it tomorrow.')
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_before_tomorrow_is_refused(self):
        answer = _with(why='I need this fixed before tomorrow.')
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_a_fortnight_is_refused(self):
        answer = _with(why='I only have a fortnight to appeal.')
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_within_the_week_is_refused(self):
        answer = _with(why='My appeal must be in within the week.')
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_may_as_a_verb_is_accepted(self):
        answer = _with(
            why="My MRI was denied, and I may need to resend my doctor's notes."
        )
        self.assertIsNotNone(parse_tailoring(answer, PROMPT))

    def test_late_followed_by_may_as_a_verb_is_accepted(self):
        answer = _with(
            questions=[
                "Records sent late may not be considered, right?",
                "Who reviewed my MRI request?",
            ]
        )
        self.assertIsNotNone(parse_tailoring(answer, PROMPT))

    def test_late_may_is_refused(self):
        answer = _with(why="I have to appeal in late May, so I need answers.")
        self.assertIsNone(parse_tailoring(answer, PROMPT))

    def test_denied_due_to_is_accepted(self):
        answer = _with(why="My claim was denied due to missing records.")
        self.assertIsNotNone(parse_tailoring(answer, PROMPT))

    def test_calling_today_is_accepted(self):
        answer = _with(
            why="I'm calling today because my MRI was denied as not medically necessary."
        )
        self.assertIsNotNone(parse_tailoring(answer, PROMPT))


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

    def test_an_invented_deadline_never_reaches_the_script(self):
        inputs = CallScriptInputs(
            goal=GOAL_FIND_OUT_WHY,
            insurer_name="Aetna",
            reason_key="medical_necessity",
            procedure="MRI L4-L5",
        )
        router = _router([_model("ours", _with(why="My appeal deadline is 4 days away."))])
        self.assertIsNone(self._tailor(router, inputs))

    def test_when_every_model_fails_the_template_stays(self):
        router = _router([_model("one", None), _model("two", "not a script")])
        self.assertIsNone(self._tailor(router))

    def test_with_no_model_to_ask_the_template_stays(self):
        self.assertIsNone(self._tailor(_router([])))


class TailorSpendChannelTest(SimpleTestCase):
    def _channel_seen(self, denial):
        seen = []
        model = _model("ours", GOOD)

        async def infer(*args, **kwargs):
            seen.append(spend.assistant_work())
            return GOOD

        model._infer_no_context = AsyncMock(side_effect=infer)
        with patch.object(call_script, "ml_router", _router([model])):
            asyncio.run(call_script.tailor_script(INPUTS, denial=denial))
        return seen

    def test_an_assistant_case_tailors_under_the_assistant_channel(self):
        denial = SimpleNamespace(channel="assistant", denial_text="denied")
        self.assertEqual(self._channel_seen(denial), [True])

    def test_a_site_case_tailors_under_the_site_channel(self):
        denial = SimpleNamespace(channel="site", denial_text="denied")
        self.assertEqual(self._channel_seen(denial), [False])
