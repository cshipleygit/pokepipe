"""The root CLI must print non-ASCII text on a cp1252 console without crashing."""

import io

from pokepipe.cli import use_utf8


def test_use_utf8_makes_cp1252_stream_print_arrows_and_symbols() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    use_utf8(stream)
    stream.write("sent → Nidoran♀")
    stream.flush()
    assert raw.getvalue().decode("utf-8") == "sent → Nidoran♀"


def test_use_utf8_ignores_streams_without_reconfigure() -> None:
    use_utf8(io.StringIO())  # e.g. a test runner's captured stream: no error
