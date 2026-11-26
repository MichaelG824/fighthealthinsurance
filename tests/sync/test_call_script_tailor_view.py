"""The endpoint call_script.ts asks for a case's tailored script parts
(views.CallScriptTailorView). It answers only for a case the post names, and
``{"tailored": null}`` whenever no model's words passed the checks, so the
page keeps the template's own."""

from unittest.mock import AsyncMock, patch

from django.test import Client, TestCase, override_settings
from django.urls import reverse

from fighthealthinsurance.call_script import GOAL_FIND_OUT_WHY, Tailoring
from fighthealthinsurance.models import Denial

EMAIL = "caller@example.com"
TAILORING = Tailoring(
    why_calling="My MRI was denied, and I want to understand why.",
    questions=("Which criteria did you apply?", "Who reviewed it?"),
)


class CallScriptTailorViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.denial = Denial.objects.create(
            denial_text="Your MRI was denied.",
            hashed_email=Denial.get_hashed_email(EMAIL),
            insurance_company="Aetna",
            use_external=False,
        )

    def _post(self, **overrides):
        data = {
            "goal": GOAL_FIND_OUT_WHY,
            "denial_id": self.denial.denial_id,
            "email": EMAIL,
            "semi_sekret": self.denial.semi_sekret,
        }
        data.update(overrides)
        return self.client.post(reverse("call_script_tailor"), data)

    def test_a_case_that_does_not_match_is_not_found(self):
        with patch("fighthealthinsurance.call_script.tailor_script", AsyncMock()):
            self.assertEqual(self._post(semi_sekret="wrong").status_code, 404)

    def test_an_invalid_goal_is_not_found(self):
        with patch("fighthealthinsurance.call_script.tailor_script", AsyncMock()):
            self.assertEqual(self._post(goal="call_them_for_me").status_code, 404)

    @override_settings(CALL_SCRIPT_ENABLED=False)
    def test_it_is_not_found_while_the_flag_is_off(self):
        self.assertEqual(self._post().status_code, 404)

    def test_tailored_parts_come_back_as_json(self):
        with patch(
            "fighthealthinsurance.call_script.tailor_script",
            AsyncMock(return_value=TAILORING),
        ):
            response = self._post()
        self.assertEqual(
            response.json(),
            {
                "tailored": {
                    "why_calling": TAILORING.why_calling,
                    "questions": list(TAILORING.questions),
                }
            },
        )

    def test_when_no_model_could_tailor_it_the_answer_is_null(self):
        with patch(
            "fighthealthinsurance.call_script.tailor_script",
            AsyncMock(return_value=None),
        ):
            self.assertEqual(self._post().json(), {"tailored": None})

    def test_the_cases_consent_to_outside_models_reaches_the_tailoring(self):
        tailor = AsyncMock(return_value=None)
        with patch("fighthealthinsurance.call_script.tailor_script", tailor):
            self._post()
        self.assertFalse(tailor.call_args.args[0].use_external)
