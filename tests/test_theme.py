"""Theme plumbing: overrides from a private theme, icons, fonts handed to libass."""

from pathlib import Path

from pokepipe.render import captions, ffmpeg, theme
from pokepipe.render.captions import Overlay, build_ass


def overlay(**kw):
    return Overlay("", 0, 360, 10, {1: 14}, {1: 4}, {1: 5.0}, [], **kw)


def test_the_suite_sees_the_public_default_look():
    assert theme.BRAND is None and theme.FONT == "Gill Sans Ultra Bold Condensed"


def test_brand_file_is_found_in_data_brand_unless_disabled(tmp_path, monkeypatch):
    from pokepipe import config

    (tmp_path / "brand").mkdir()
    (tmp_path / "brand" / "theme.py").write_text("FONT = 'X'")
    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(data_dir=tmp_path,
                                                                        _env_file=None))
    assert theme.brand_file() is None  # POKEPIPE_NO_BRAND, set by conftest
    monkeypatch.delenv("POKEPIPE_NO_BRAND")
    assert theme.brand_file() == tmp_path / "brand" / "theme.py"


def test_text_width_falls_back_without_a_font_file(tmp_path):
    assert theme.text_width("$5.99", 84, None) == 0.6 * 84 * 5
    assert theme.text_width("$5.99", 84, tmp_path) == 0.6 * 84 * 5


def test_shape_is_placed_at_its_top_left_and_drawn_from_zero():
    text = captions.shape([(110, 50), (130, 60), (120, 80)], "&H00FFFFFF", 2)
    assert r"\an7\pos(110.0,50.0)" in text
    assert "m 0.0 0.0 l 20.0 10.0 10.0 30.0" in text


def test_default_look_has_text_only_badges_and_prices():
    ass = build_ass([], [], overlay(prices=((2.0, 4.0, "$12.34"),)), 8.0)
    assert sum(",Price," in line for line in ass.splitlines()) == 1
    assert sum(",Badge," in line for line in ass.splitlines()) == 1


def test_theme_icons_are_drawn_before_badge_and_price_text(monkeypatch):
    def dot(x, cy, extra=""):
        return [captions.shape(captions.circle(x + 10, cy, 10), "&H0&")]

    monkeypatch.setattr(theme, "badge_icon", lambda x, cy: (dot(x, cy), 30))
    monkeypatch.setattr(theme, "price_icon", lambda x, cy, extra: dot(x, cy) * 2)
    monkeypatch.setattr(theme, "PRICE_ICON_W", 20)
    ass = build_ass([], [], overlay(prices=((2.0, 4.0, "$12.34"),)), 8.0)
    badge = [line for line in ass.splitlines() if ",Badge," in line]
    price = [line for line in ass.splitlines() if ",Price," in line]
    assert len(badge) == 2 and f"pos({captions.BADGE_X + 30:g}," in badge[-1]
    assert len(price) == 3 and "$12.34" in price[-1]


def test_subtitles_filter_points_libass_at_the_fonts_folder():
    cmd = ffmpeg.final_cmd_source_audio(Path("l.txt"), "c.ass", 5.0, Path("o.mp4"), False,
                                        fonts_dir="fonts")
    assert "subtitles=c.ass:fontsdir=fonts" in cmd[cmd.index("-filter_complex") + 1]
    plain = ffmpeg.final_cmd_source_audio(Path("l.txt"), "c.ass", 5.0, Path("o.mp4"), False)
    assert "fontsdir" not in plain[plain.index("-filter_complex") + 1]
