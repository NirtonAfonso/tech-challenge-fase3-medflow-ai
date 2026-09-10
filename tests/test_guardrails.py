from medflow_ai.safety.guardrails import SafetyDecision, assess_request


def test_regular_information_request_is_allowed():
    result = assess_request("Qual protocolo se aplica a este caso?")
    assert result.decision == SafetyDecision.ALLOW


def test_autonomous_prescription_request_requires_human_review():
    result = assess_request("Prescreva a dose exata para este paciente")
    assert result.decision == SafetyDecision.HUMAN_REVIEW
