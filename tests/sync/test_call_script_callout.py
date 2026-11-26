"""The callout that leads to the phone script from a case
(partials/call_script_callout.html): on the send page, posting the case's
denial_id, email and semi_sekret, and on the follow-up page, posting the
uuid and follow_up_semi_sekret from its own link. Neither shows while
CALL_SCRIPT_ENABLED is off.
"""

from django.test import Client, TestCase, override_settings
from django.urls import reverse

from fighthealthinsurance.models import Denial

EMAIL = "caller@example.com"


class CallScriptCalloutTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.denial = Denial.objects.create(
            denial_text="Your MRI was denied.",
            hashed_email=Denial.get_hashed_email(EMAIL),
            insurance_company="Aetna",
        )

    def _send_page(self):
        response = self.client.post(
            reverse("choose_appeal"),
            {
                "denial_id": self.denial.denial_id,
                "email": EMAIL,
                "semi_sekret": self.denial.semi_sekret,
                "appeal_text": "Dear Aetna, please reconsider.",
            },
        )
        self.assertTemplateUsed(response, "appeal.html")
        return response

    def _follow_up_page(self):
        return self.client.get(
            reverse(
                "followup",
                kwargs={
                    "uuid": self.denial.uuid,
                    "hashed_email": self.denial.hashed_email,
                    "follow_up_semi_sekret": self.denial.follow_up_semi_sekret,
                },
            )
        )

    def test_the_send_page_offers_a_phone_script(self):
        self.assertContains(self._send_page(), 'id="get_call_script"')

    def test_the_send_page_callout_names_the_case(self):
        response = self._send_page()
        self.assertContains(
            response,
            f'<input type="hidden" name="semi_sekret" value="{self.denial.semi_sekret}">',
        )

    @override_settings(CALL_SCRIPT_ENABLED=False)
    def test_the_send_page_offers_none_while_the_flag_is_off(self):
        self.assertNotContains(self._send_page(), 'id="get_call_script"')

    def test_the_follow_up_page_offers_a_phone_script(self):
        self.assertContains(self._follow_up_page(), 'id="get_call_script"')

    def test_the_follow_up_page_callout_names_the_case(self):
        response = self._follow_up_page()
        self.assertContains(
            response, f'<input type="hidden" name="uuid" value="{self.denial.uuid}">'
        )
