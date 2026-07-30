from routelabs_router.privacy import redact_sensitive_text


def test_redact_sensitive_text_covers_common_pii_and_secret_patterns() -> None:
    result = redact_sensitive_text(
        "Email alice@example.com, call 312-555-0199, SSN 123-45-6789, "
        "card 4111 1111 1111 1111, OPENAI_API_KEY=sk-testsecret1234567890."
    )

    assert result.applied is True
    assert result.replacement_count == 5
    assert "alice@example.com" not in result.text
    assert "312-555-0199" not in result.text
    assert "123-45-6789" not in result.text
    assert "4111 1111 1111 1111" not in result.text
    assert "sk-testsecret1234567890" not in result.text
    assert set(result.categories) == {
        "private_email",
        "private_phone",
        "private_identifier",
        "payment_card",
        "secret",
    }
