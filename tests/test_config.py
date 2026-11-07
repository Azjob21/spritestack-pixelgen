import dataclasses

import pytest

from pixelgen.pipeline import VARIANTS, Config, build_prompts, get_config

LADDER = ["v0", "e1", "e2", "e3", "e4", "e5", "e6"]


def test_each_rung_changes_exactly_one_field():
    ignore = {"name", "fixed_seed"}   # fixed_seed reset in e1 is a compat detail, not an experiment
    for prev, cur in zip(LADDER, LADDER[1:]):
        a, b = dataclasses.asdict(VARIANTS[prev]), dataclasses.asdict(VARIANTS[cur])
        diff = {k for k in a if a[k] != b[k] and k not in ignore}
        assert len(diff) == 1, (prev, cur, diff)


def test_v0_is_reproducible_baseline():
    v0 = VARIANTS["v0"]
    assert v0.fixed_seed == 42 and v0.post == "legacy" and v0.steps == 8 and v0.guidance == 1.8
    pos, _ = build_prompts(v0, "dragon")
    assert "PixArFK" not in pos and "8-bit pixel-art of dragon" in pos


def test_e1_prompt_has_trigger_and_no_backfiring_tokens():
    pos, neg = build_prompts(VARIANTS["e1"], "dragon")
    assert pos.startswith("PixArFK") and "no anti-aliasing" not in pos and "16-bit" not in pos
    assert "magenta" not in pos            # legacy post can't key it, so don't ask for it
    pos2, _ = build_prompts(VARIANTS["e2"], "dragon")
    assert "magenta background" in pos2


def test_overrides_and_validation():
    c = get_config("e2", steps=6, guidance=None)
    assert c.steps == 6 and c.guidance == VARIANTS["e2"].guidance and c.name == "e2+steps6"
    with pytest.raises(ValueError):
        get_config("e2", bogus=1)
    with pytest.raises(ValueError):
        get_config("nope")
    with pytest.raises(ValueError):
        Config(gen_size=500)


def test_g_ladder_changes_one_field_per_rung():
    ignore = {"name"}
    ladder = ["v0g", "g3", "g4", "g5", "g6"]
    for prev, cur in zip(ladder[1:], ladder[2:]):
        a, b = dataclasses.asdict(VARIANTS[prev]), dataclasses.asdict(VARIANTS[cur])
        assert len({k for k in a if a[k] != b[k] and k not in ignore}) == 1, (prev, cur)
    a, b = dataclasses.asdict(VARIANTS["v0g"]), dataclasses.asdict(VARIANTS["g3"])
    assert {k for k in a if a[k] != b[k]} == {"name", "guidance"}


def test_v0g_keeps_v0_prompt_and_seed_but_uses_grid_post():
    v0g = VARIANTS["v0g"]
    assert v0g.fixed_seed == 42 and v0g.post == "grid" and v0g.bg_tol < 60
    assert build_prompts(v0g, "dragon") == build_prompts(VARIANTS["v0"], "dragon")


def test_trigger_word_variant_only_adds_trigger():
    base, _ = build_prompts(VARIANTS["v0g"], "dragon")
    trig, _ = build_prompts(VARIANTS["v0gt"], "dragon")
    assert trig == "PixArFK, " + base


def test_rescue_variants_change_one_thing_from_their_parent():
    def diff(a, b):
        da, db = dataclasses.asdict(VARIANTS[a]), dataclasses.asdict(VARIANTS[b])
        return {k for k in da if da[k] != db[k] and k != "name"}
    assert diff("g3", "g3t") == {"prompt_style"}
    assert diff("g3", "g3l") == {"lora_scale"}
    assert diff("v0g", "s4") == {"steps"}


def test_override_names_are_distinct_so_output_dirs_dont_collide():
    a, b = get_config("g3", lora_scale=1.3), get_config("g3", lora_scale=1.5)
    assert a.name != b.name and a.name.startswith("g3+")
