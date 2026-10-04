import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import ValidationError

import graph
from nodes import CustomerInfoExtraction, capture_customer_info


class StubStructuredModel:
    def __init__(self, response: CustomerInfoExtraction) -> None:
        self.response = response
        self.schema: type[CustomerInfoExtraction] | None = None
        self.messages: list[BaseMessage] | None = None

    def with_structured_output(
        self,
        schema: type[CustomerInfoExtraction],
    ) -> "StubStructuredModel":
        self.schema = schema
        return self

    def invoke(self, messages: list[BaseMessage]) -> CustomerInfoExtraction:
        self.messages = messages
        return self.response


class CaptureCustomerInfoTests(unittest.TestCase):
    def test_extracts_explicit_details_with_structured_schema(self) -> None:
        model = StubStructuredModel(
            CustomerInfoExtraction(
                customer_name="Jane Doe",
                email="jane@example.com",
                mobile_number="+91 98765 43210",
                product_id="TEST-SKIN-001",
            )
        )
        messages = [
            HumanMessage(content="Earlier unrelated message"),
            AIMessage(content="Would you share your contact details?"),
            HumanMessage(content="My name is Jane Doe, and my email is jane@example.com"),
        ]

        with patch("nodes.get_model_from_gcp", return_value=model):
            result = capture_customer_info({"messages": messages})

        self.assertEqual(
            result,
            {
                "customer_name": "Jane Doe",
                "email": "jane@example.com",
                "mobile_number": "+91 98765 43210",
                "product_id": "TEST-SKIN-001",
                "lead_status": "captured",
            },
        )
        self.assertIs(model.schema, CustomerInfoExtraction)
        self.assertIsNotNone(model.messages)
        self.assertEqual(len(model.messages), 2)
        self.assertIn("Would you share your contact details?", model.messages[1].content)
        self.assertIn("jane@example.com", model.messages[1].content)
        self.assertNotIn("Earlier unrelated message", model.messages[1].content)

    def test_schema_validates_and_normalizes_customer_fields(self) -> None:
        extraction = CustomerInfoExtraction(
            customer_name=" Jane Doe ",
            email="jane@example.com",
            mobile_number="+91 98765 43210",
            product_id="test-skin-001",
        )

        self.assertEqual(extraction.customer_name, "Jane Doe")
        self.assertEqual(extraction.email, "jane@example.com")
        self.assertEqual(extraction.mobile_number, "+91 98765 43210")
        self.assertEqual(extraction.product_id, "test-skin-001")
        email_schema = CustomerInfoExtraction.model_json_schema()["properties"]["email"]
        self.assertEqual(email_schema["anyOf"][0]["format"], "email")

    def test_schema_rejects_invalid_field_values(self) -> None:
        invalid_fields = (
            {"email": "not-an-email"},
            {"mobile_number": "12345"},
            {"mobile_number": "+91 abcdefghij"},
            {"product_id": "not a product id"},
            {"customer_name": " "},
        )
        for fields in invalid_fields:
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                CustomerInfoExtraction(**fields)

    def test_explicit_contact_decline_updates_lead_status(self) -> None:
        model = StubStructuredModel(CustomerInfoExtraction(contact_declined=True))

        with patch("nodes.get_model_from_gcp", return_value=model):
            result = capture_customer_info(
                {
                    "messages": [
                        AIMessage(content="Would you like to share your phone number?"),
                        HumanMessage(content="No, thanks"),
                    ]
                }
            )

        self.assertEqual(result, {"lead_status": "declined"})

    def test_empty_extraction_returns_no_updates(self) -> None:
        model = StubStructuredModel(CustomerInfoExtraction())

        with patch("nodes.get_model_from_gcp", return_value=model):
            result = capture_customer_info(
                {
                    "messages": [
                        AIMessage(content="What kind of product are you looking for?"),
                        HumanMessage(content="Something lightweight"),
                    ]
                }
            )

        self.assertEqual(result, {})

    def test_no_customer_message_does_not_call_model(self) -> None:
        with patch("nodes.get_model_from_gcp") as get_model:
            result = capture_customer_info({"messages": [AIMessage(content="Hello")]})

        self.assertEqual(result, {})
        get_model.assert_not_called()

    def test_graph_applies_extracted_state(self) -> None:
        model = StubStructuredModel(
            CustomerInfoExtraction(product_id="TEST-SKIN-001", email="jane@example.com")
        )

        with (
            patch(
                "graph.assistant",
                return_value={"messages": [AIMessage(content="Here is a sample option.")]},
            ),
            patch("nodes.get_model_from_gcp", return_value=model),
        ):
            result = graph.create_graph().compile().invoke(
                {"messages": [HumanMessage(content="Please tell me about TEST-SKIN-001; email jane@example.com")]}
            )

        self.assertEqual(result["product_id"], "TEST-SKIN-001")
        self.assertEqual(result["email"], "jane@example.com")
        self.assertEqual(result["lead_status"], "captured")


if __name__ == "__main__":
    unittest.main()
