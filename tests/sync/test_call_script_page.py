"""The phone script page (views.CallScriptView, call_script.html).

On its own the page is a short form, and the script it makes asks no model.
A case is named by the fields the send page or the follow-up page carries
(test_call_script_callout.py); a post naming a case we can't find gets the
form, never someone else's script. Everything is behind CALL_SCRIPT_ENABLED,
which Dev and the tests turn on.
"""

from unittest.mock import patch

from django.test import Client, TestCase, override_settings
from django.urls import reverse

from fighthealthinsurance.call_script import (
    GOAL_CHECK_STATUS,
    GOAL_ESCALATE,
    GOAL_FIND_OUT_WHY,
    GOALS,
)
from fighthealthinsurance.models import Denial, InsuranceCompany

EMAIL = "caller@example.com"


class CallScriptPageTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.company = InsuranceCompany.objects.create(
            name="Example Health Plan",
            regex="example health plan",
            negative_regex="zzznever",
            appeal_phone_number="800-555-0100",
        )
        self.denial = Denial.objects.create(
            denial_text="Your MRI was denied as not medically necessary.",
            hashed_email=Denial.get_hashed_email(EMAIL),
            insurance_company="Example Health Plan",
            insurance_company_obj=self.company,
            procedure="MRI of the lumbar spine",
            triage_category="medical_necessity",
            use_external=False,
        )

    def _case(self, goal=GOAL_FIND_OUT_WHY, **overrides):
        data = {
            "goal": goal,
            "denial_id": self.denial.denial_id,
            "email": EMAIL,
            "semi_sekret": self.denial.semi_sekret,
        }
        data.update(overrides)
        return self.client.post(reverse("call_script"), data)

    def _standalone(self, goal=GOAL_FIND_OUT_WHY, **fields):
        return self.client.post(reverse("call_script"), {"goal": goal, **fields})

    @override_settings(CALL_SCRIPT_ENABLED=False)
    def test_the_page_is_not_found_while_the_flag_is_off(self):
        self.assertEqual(self.client.get(reverse("call_script")).status_code, 404)

    def test_the_page_opens_on_the_form(self):
        response = self.client.get(reverse("call_script"))
        self.assertContains(response, 'id="call_script_form"')

    def test_the_form_makes_a_script_for_every_goal(self):
        for goal, label in GOALS:
            with self.subTest(goal=goal):
                response = self._standalone(goal, insurer="Aetna")
                self.assertContains(response, f"Goal: {label}")

    def test_the_standalone_script_names_the_insurer(self):
        response = self._standalone(insurer="Aetna", reason="out_of_network")
        self.assertContains(response, "Aetna · Out of network")

    def test_the_standalone_script_is_not_sent_for_tailoring(self):
        response = self._standalone(insurer="Aetna")
        self.assertNotContains(response, 'id="call-script-tailor"')

    def test_the_standalone_script_asks_no_model(self):
        with patch("fighthealthinsurance.call_script.tailor_script") as tailor:
            self._standalone(insurer="Aetna")
        tailor.assert_not_called()

    def test_a_case_script_is_built_from_the_case(self):
        response = self._case()
        self.assertContains(
            response, "Example Health Plan · Not medically necessary · MRI of the lumbar spine"
        )

    def test_a_case_script_gives_the_insurers_appeals_line_from_our_records(self):
        response = self._case()
        self.assertContains(response, "Call 800-555-0100, Example Health Plan")

    def test_a_case_script_is_sent_for_tailoring(self):
        response = self._case()
        self.assertContains(response, 'id="call-script-tailor"')

    def test_a_case_that_does_not_match_gets_the_form(self):
        response = self._case(semi_sekret="wrong")
        self.assertContains(response, "We couldn't open your case from here")

    def test_a_case_that_does_not_match_shows_none_of_the_case(self):
        response = self._case(semi_sekret="wrong")
        self.assertNotContains(response, "Example Health Plan")

    def test_the_follow_up_pages_fields_open_the_case(self):
        response = self.client.post(
            reverse("call_script"),
            {
                "goal": GOAL_CHECK_STATUS,
                "uuid": str(self.denial.uuid),
                "follow_up_semi_sekret": self.denial.follow_up_semi_sekret,
            },
        )
        self.assertContains(response, "MRI of the lumbar spine")

    def test_switching_goals_keeps_the_case(self):
        response = self._case(GOAL_ESCALATE)
        self.assertContains(
            response,
            f'<input type="hidden" name="semi_sekret" value="{self.denial.semi_sekret}">',
        )

    def test_an_invalid_goal_gets_the_form_back(self):
        response = self._standalone("call_them_for_me")
        self.assertContains(response, 'id="call_script_form"')

    def test_the_script_says_we_never_call_for_you(self):
        response = self._standalone(insurer="Aetna")
        self.assertContains(response, "You make the call. We never call for you.")

    def test_the_menu_links_to_the_page(self):
        response = self.client.get(reverse("call_script"))
        self.assertContains(response, ">What to Say on the Phone</a>")

    @override_settings(CALL_SCRIPT_ENABLED=False)
    def test_the_menu_does_not_link_to_the_page_while_the_flag_is_off(self):
        response = self.client.get(reverse("explain_denial"))
        self.assertNotContains(response, reverse("call_script"))
