from medflow_ai.logging_utils.audit import build_audit_event, pseudonymize


def test_pseudonymization_is_stable_and_does_not_expose_identifier():
    raw_id = "patient-001"
    hashed = pseudonymize(raw_id)
    assert hashed == pseudonymize(raw_id)
    assert raw_id not in hashed


def test_audit_event_stores_only_hashed_patient_id():
    event = build_audit_event(
        trace_id="trace-1",
        route="clinical_question",
        patient_id="patient-001",
        sources=["protocol-a"],
        safety_status="allow",
    )
    assert event["patient_id_hash"] != "patient-001"
    assert event["sources"] == ["protocol-a"]
