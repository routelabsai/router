import re

from routelabs_router.models import PrivacyDetectionResult


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
SECRET_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z\-_]{20,})\b"
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
