import claims


def _decompose_with_response(monkeypatch, response):
    monkeypatch.setattr(claims, "ask_model", lambda *a, **k: response)
    return claims.decompose_answer("irrelevant answer", model="m", temperature=0.0, backend="hf")


def test_numbered_dot_marker_is_stripped(monkeypatch):
    result = _decompose_with_response(monkeypatch, "1. Aspirin thins blood")
    assert result == ["Aspirin thins blood"]


def test_dash_bullet_marker_is_stripped(monkeypatch):
    result = _decompose_with_response(monkeypatch, "- Bar")
    assert result == ["Bar"]


def test_dose_leading_number_is_preserved(monkeypatch):
    result = _decompose_with_response(monkeypatch, "500 mg daily is the max dose")
    assert result == ["500 mg daily is the max dose"]


def test_year_leading_number_is_preserved(monkeypatch):
    result = _decompose_with_response(monkeypatch, "2019 guidelines recommend X")
    assert result == ["2019 guidelines recommend X"]


def test_numbered_paren_marker_is_stripped(monkeypatch):
    result = _decompose_with_response(monkeypatch, "3) Foo")
    assert result == ["Foo"]


def test_star_bullet_marker_is_stripped(monkeypatch):
    result = _decompose_with_response(monkeypatch, "* Star bullet")
    assert result == ["Star bullet"]


def test_blank_lines_are_skipped(monkeypatch):
    response = "1. Aspirin thins blood\n\n   \n- Bar"
    result = _decompose_with_response(monkeypatch, response)
    assert result == ["Aspirin thins blood", "Bar"]


def test_mixed_claims_multiline(monkeypatch):
    response = (
        "1. Aspirin thins blood\n"
        "- Bar\n"
        "500 mg daily is the max dose\n"
        "2019 guidelines recommend X\n"
        "3) Foo"
    )
    result = _decompose_with_response(monkeypatch, response)
    assert result == [
        "Aspirin thins blood",
        "Bar",
        "500 mg daily is the max dose",
        "2019 guidelines recommend X",
        "Foo",
    ]
