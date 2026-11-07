import dataclasses

from pixelgen.pipeline import DEFAULT_VARIANT, VARIANTS, build_prompts


def test_default_is_recommended_config_not_magenta_prompt():
    assert DEFAULT_VARIANT == "s6"
    cfg = VARIANTS[DEFAULT_VARIANT]
    assert (cfg.steps, cfg.guidance, cfg.tiny_vae) == (6, 2.5, True)
    pos, _ = build_prompts(cfg, "dragon")
    assert "magenta" not in pos


def test_s6_differs_from_v0g_only_in_sampling_and_vae():
    a, b = dataclasses.asdict(VARIANTS["v0g"]), dataclasses.asdict(VARIANTS["s6"])
    assert {k for k in a if a[k] != b[k] and k != "name"} == {"steps", "guidance", "tiny_vae"}
