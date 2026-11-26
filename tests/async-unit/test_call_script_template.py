"""The phone script's fixed template (call_script.py, issue #568).

Every script, whatever its goal and whether or not a model tailored it, asks
for the rep's name and a call reference number first, asks whether AI or
automation was involved, and ends by asking for written confirmation. The
script is built from a case's fields or the standalone form, never from the
letter itself.
"""

from types import SimpleNamespace

from django.test import SimpleTestCase

from fighthealthinsurance import call_script
from fighthealthinsurance.call_script import (
    GOAL_CHECK_STATUS,
    GOAL_ESCALATE,
    GOAL_FIND_OUT_WHY,
    CallScriptInputs,
    Tailoring,
    build_script,
)
from fighthealthinsurance.ml.denial_triage import CATEGORIES

ALL_GOALS = [goal for goal, _ in call_script.GOALS]


def _script(goal=GOAL_FIND_OUT_WHY, **fields):
    return build_script(CallScriptInputs(goal=goal, **fields))


def _texts(script):
    return [question.text for question in script.questions]


def _denial(**fields):
    """A case as inputs_from_denial reads it: an insurer, a plan, the
    reason fields, and what was denied."""
    procedure = fields.pop("procedure", "MRI of the lumbar spine")
    values = {
        "insurance_company_obj": None,
        "insurance_plan_obj": None,
        "insurance_company": "Aetna",
        "triage_category": None,
        "claim_id": "CLM-0000",
        "urgent": False,
        "use_external": True,
        "value_or_candidate": lambda field: procedure if field == "procedure" else None,
    }
    values.update(fields)
    return SimpleNamespace(**values)


class TemplateTest(SimpleTestCase):
    def test_every_goal_opens_by_asking_for_the_reps_name_and_a_reference_number(self):
        for goal in ALL_GOALS:
            with self.subTest(goal=goal):
                opening = " ".join(line.text for line in _script(goal).opening)
                self.assertIn("can I get your name and a reference number", opening)

    def test_every_goal_asks_whether_ai_was_involved(self):
        for goal in ALL_GOALS:
            with self.subTest(goal=goal):
                self.assertIn(call_script.AI_QUESTION, _texts(_script(goal)))

    def test_every_goal_ends_by_asking_for_written_confirmation(self):
        for goal in ALL_GOALS:
            with self.subTest(goal=goal):
                self.assertTrue(
                    any("written confirmation" in line for line in _script(goal).wrap_up)
                )

    def test_every_goal_has_reason_questions_a_model_may_tailor(self):
        for goal in ALL_GOALS:
            with self.subTest(goal=goal):
                self.assertTrue(any(q.tailorable for q in _script(goal).questions))

    def test_every_triage_category_has_a_label_and_a_question_bank(self):
        self.assertEqual(set(call_script.REASON_LABELS), set(CATEGORIES))
        self.assertEqual(set(call_script.REASON_QUESTIONS), set(CATEGORIES))

    def test_why_calling_names_what_was_denied_and_why(self):
        script = _script(reason_key="medical_necessity", procedure="an MRI")
        self.assertEqual(
            script.why_calling,
            "My claim for an MRI was denied as not medically necessary. I'd like "
            "to understand exactly why, so I can decide what to do next.",
        )

    def test_escalating_a_clinical_denial_asks_for_a_peer_to_peer_review(self):
        texts = _texts(_script(GOAL_ESCALATE, reason_key="medical_necessity"))
        self.assertTrue(any("peer-to-peer" in text for text in texts))

    def test_escalating_a_billing_denial_does_not_ask_for_a_peer_to_peer_review(self):
        texts = _texts(_script(GOAL_ESCALATE, reason_key="coding_or_billing"))
        self.assertFalse(any("peer-to-peer" in text for text in texts))

    def test_an_urgent_case_asks_for_an_expedited_appeal(self):
        self.assertIn(call_script.URGENT_QUESTION, _texts(_script(urgent=True)))

    def test_a_case_that_is_not_urgent_does_not_ask_for_an_expedited_appeal(self):
        self.assertNotIn(call_script.URGENT_QUESTION, _texts(_script(urgent=False)))

    def test_without_our_number_it_points_to_the_card(self):
        self.assertIn(
            "Call the member services number on the back of your insurance card.",
            _script().before_you_call,
        )

    def test_our_number_is_labelled_as_from_our_records(self):
        lines = _script(insurer_name="Aetna", phone="800-555-0100").before_you_call
        self.assertIn(
            "Call 800-555-0100, Aetna's appeals line from our records. Check it "
            "against your card or letter.",
            lines,
        )

    def test_each_goal_has_its_own_rows_on_the_notes_sheet(self):
        self.assertIn(
            "Date they received my appeal", _script(GOAL_CHECK_STATUS).notes_rows
        )

    def test_tailoring_takes_the_place_of_the_reason_questions(self):
        tailoring = Tailoring(
            why_calling="I'm calling about my denied MRI.",
            questions=("Which criteria did you apply?", "Who reviewed it?"),
        )
        plain = _texts(_script(reason_key="medical_necessity"))
        tailored = _texts(
            build_script(
                CallScriptInputs(goal=GOAL_FIND_OUT_WHY, reason_key="medical_necessity"),
                tailoring,
            )
        )
        reason = list(call_script.REASON_QUESTIONS["medical_necessity"])
        at = plain.index(reason[0])
        expected = plain[:at] + list(tailoring.questions) + plain[at + len(reason) :]
        self.assertEqual(tailored, expected)

    def test_tailoring_replaces_why_calling(self):
        tailoring = Tailoring(
            why_calling="I'm calling about my denied MRI.", questions=("Why?", "Who?")
        )
        script = build_script(CallScriptInputs(goal=GOAL_FIND_OUT_WHY), tailoring)
        self.assertEqual(script.why_calling, tailoring.why_calling)


class FormInputsTest(SimpleTestCase):
    def test_the_standalone_form_never_allows_outside_models(self):
        inputs = call_script.inputs_from_form(
            {"goal": GOAL_FIND_OUT_WHY, "insurer": "Aetna", "reason": "other"}
        )
        self.assertFalse(inputs.use_external)

    def test_an_unknown_reason_on_the_form_is_other(self):
        inputs = call_script.inputs_from_form(
            {"goal": GOAL_FIND_OUT_WHY, "reason": "made_up"}
        )
        self.assertEqual(inputs.reason_key, call_script.REASON_OTHER)

    def test_no_insurer_on_the_form_reads_as_your_insurance_company(self):
        script = build_script(call_script.inputs_from_form({"goal": GOAL_FIND_OUT_WHY}))
        self.assertEqual(script.insurer, "your insurance company")


class DenialInputsTest(SimpleTestCase):
    def test_the_reason_comes_from_the_triage_category(self):
        inputs = call_script.inputs_from_denial(
            _denial(triage_category="out_of_network"), GOAL_FIND_OUT_WHY, []
        )
        self.assertEqual(inputs.reason_key, "out_of_network")

    def test_without_a_triage_category_the_reason_comes_from_the_denial_type(self):
        inputs = call_script.inputs_from_denial(
            _denial(), GOAL_FIND_OUT_WHY, ["Not Covered By Plan"]
        )
        self.assertEqual(inputs.reason_key, "not_a_covered_benefit")

    def test_with_neither_the_reason_is_other(self):
        inputs = call_script.inputs_from_denial(_denial(), GOAL_FIND_OUT_WHY, ["Other"])
        self.assertEqual(inputs.reason_key, call_script.REASON_OTHER)

    def test_a_boilerplate_procedure_is_left_out(self):
        inputs = call_script.inputs_from_denial(
            _denial(procedure="the above referenced service"), GOAL_FIND_OUT_WHY, []
        )
        self.assertEqual(inputs.procedure, "")

    def test_an_unknown_insurer_reads_as_no_insurer(self):
        inputs = call_script.inputs_from_denial(
            _denial(insurance_company="UNKNOWN"), GOAL_FIND_OUT_WHY, []
        )
        self.assertEqual(inputs.insurer_name, "")

    def test_the_plans_appeals_line_wins_over_the_companys(self):
        company = SimpleNamespace(
            name="Anthem", appeal_destinations=lambda: {"phone": "800-555-0100"}
        )
        plan = SimpleNamespace(appeal_destinations=lambda: {"phone": "800-555-0199"})
        inputs = call_script.inputs_from_denial(
            _denial(insurance_company_obj=company, insurance_plan_obj=plan),
            GOAL_FIND_OUT_WHY,
            [],
        )
        self.assertEqual(inputs.phone, "800-555-0199")

    def test_the_cases_consent_to_outside_models_carries_over(self):
        inputs = call_script.inputs_from_denial(
            _denial(use_external=False), GOAL_FIND_OUT_WHY, []
        )
        self.assertFalse(inputs.use_external)


class CaseFieldsTest(SimpleTestCase):
    def test_the_send_pages_fields_name_a_case(self):
        fields = call_script.case_fields(
            {"denial_id": "7", "email": "a@example.com", "semi_sekret": "s"}
        )
        self.assertEqual(
            fields, {"denial_id": "7", "email": "a@example.com", "semi_sekret": "s"}
        )

    def test_a_mangled_denial_id_names_no_case(self):
        fields = call_script.case_fields(
            {"denial_id": "7; drop", "email": "a@example.com", "semi_sekret": "s"}
        )
        self.assertEqual(fields, {})

    def test_a_mangled_uuid_names_no_case(self):
        fields = call_script.case_fields(
            {"uuid": "not-a-uuid", "follow_up_semi_sekret": "s"}
        )
        self.assertEqual(fields, {})

    def test_the_follow_up_pages_fields_name_a_case(self):
        case_uuid = "3f1c5a8e-2b7d-4c9a-9e6f-1a2b3c4d5e6f"
        fields = call_script.case_fields(
            {"uuid": case_uuid, "follow_up_semi_sekret": "s"}
        )
        self.assertEqual(fields, {"uuid": case_uuid, "follow_up_semi_sekret": "s"})
