import re

from routelabs_router.models import PrivacyDetectionResult


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
SECRET_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z\-_]{20,})\b"
)
ENV_SECRET_RE = re.compile(
    r"\b[A-Z][A-Z0-9_]{2,}(?:API_KEY|TOKEN|SECRET|PASSWORD)\s*=\s*[^\s,;]+"
)

CODE_SIGNALS = [
    "def ",
    "class ",
    "import ",
    "from ",
    "function ",
    "const ",
    "let ",
    "var ",
    "SELECT ",
    "INSERT ",
    "UPDATE ",
    "DELETE ",
    "CREATE TABLE",
    "```",
    "{",
    "};",
]


class HeuristicPrivacyDetector:
    def evaluate(self, text: str, explicitly_private: bool = False) -> PrivacyDetectionResult:
        categories: list[str] = []
        reasons: list[str] = []
        lowered = text.lower()

        if EMAIL_RE.search(text):
            categories.append("private_email")
            reasons.append("detected email-like text")
        if PHONE_RE.search(text):
            categories.append("private_phone")
            reasons.append("detected phone-like text")
        if SSN_RE.search(text):
            categories.append("private_identifier")
            reasons.append("detected ssn-like text")
        if CARD_RE.search(text):
            categories.append("account_number")
            reasons.append("detected account-or-card-like text")
        if SECRET_RE.search(text):
            categories.append("secret")
            reasons.append("detected secret-like token")

        code_matches = sum(1 for signal in CODE_SIGNALS if signal.lower() in lowered)
        if code_matches >= 2:
            categories.append("code")
            reasons.append("detected code-like content")

        detected = explicitly_private or bool(categories)
        if explicitly_private and "explicit_private_flag" not in categories:
            categories.append("explicit_private_flag")
            reasons.append("request was explicitly marked private")

        return PrivacyDetectionResult(
            detected=detected,
            categories=categories,
            reasons=reasons,
            forced_local=detected,
        )


class RedactionResult:
    def __init__(self, text: str, categories: list[str], replacement_count: int) -> None:
        self.text = text
        self.categories = categories
        self.replacement_count = replacement_count

    @property
    def applied(self) -> bool:
        return self.replacement_count > 0


def redact_sensitive_text(text: str) -> RedactionResult:
    redacted = text
    categories: list[str] = []
    replacement_count = 0

    for category, pattern, replacement in [
        ("private_email", EMAIL_RE, "[REDACTED_EMAIL]"),
        ("private_identifier", SSN_RE, "[REDACTED_IDENTIFIER]"),
        ("payment_card", CARD_RE, "[REDACTED_CARD]"),
        ("private_phone", PHONE_RE, "[REDACTED_PHONE]"),
        ("secret", ENV_SECRET_RE, "[REDACTED_SECRET]"),
        ("secret", SECRET_RE, "[REDACTED_SECRET]"),
    ]:
        redacted, count = pattern.subn(replacement, redacted)
        if count:
            replacement_count += count
            if category not in categories:
                categories.append(category)

    return RedactionResult(
        text=redacted,
        categories=categories,
        replacement_count=replacement_count,
    )
