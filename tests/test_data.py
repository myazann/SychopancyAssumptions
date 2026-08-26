import warnings

from syco.data import recover_messages


def test_recovery_does_not_emit_invalid_escape_syntax_warning():
    malformed = "[{'role': 'user', 'content': 'dash \\- text'}]"
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        messages = recover_messages(malformed)
    assert messages == [{"role": "user", "content": "dash \\- text"}]
    assert not any(item.category is SyntaxWarning for item in seen)
