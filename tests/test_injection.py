from mirror.injection import generate

PROMPT = "The weather today is"


def test_alpha_zero_is_golden(model, vec):
    clean = generate(model, PROMPT, seed=0, max_new_tokens=12)
    zeroed = generate(model, PROMPT, vec, alpha=0.0, seed=0, max_new_tokens=12)
    assert clean == zeroed


def test_huge_alpha_derails(model, vec):
    clean = generate(model, PROMPT, seed=0, max_new_tokens=12)
    injected = generate(model, PROMPT, vec, alpha=200.0, seed=0, max_new_tokens=12)
    assert clean != injected


def test_spans_differ(model, vec):
    clean = generate(model, PROMPT, seed=0, max_new_tokens=12)
    single = generate(model, PROMPT, vec, alpha=200.0, span="single", seed=0, max_new_tokens=12)
    response = generate(model, PROMPT, vec, alpha=200.0, span="response", seed=0, max_new_tokens=12)
    assert single != clean
    assert single != response


def test_injection_touches_only_last_position(model, vec):
    import torch

    from mirror.injection import make_hook
    from mirror.vectors import hook_name

    tokens = model.to_tokens(PROMPT)
    with torch.no_grad():
        _, clean_cache = model.run_with_cache(tokens, names_filter=hook_name(3))
        hook = make_hook(vec, 8.0, "response")
        with model.hooks(fwd_hooks=[(hook_name(3), hook)]):
            _, injected_cache = model.run_with_cache(tokens, names_filter=hook_name(3))
    clean = clean_cache[hook_name(3)][0]
    injected = injected_cache[hook_name(3)][0]
    assert torch.equal(clean[:-1], injected[:-1])
    expected = clean[-1] + 8.0 * vec.sigma * vec.direction
    assert torch.allclose(injected[-1], expected, atol=1e-4)


def test_upstream_layers_untouched(model, vec):
    import torch

    from mirror.injection import make_hook
    from mirror.vectors import hook_name

    tokens = model.to_tokens(PROMPT)
    with torch.no_grad():
        _, clean_cache = model.run_with_cache(tokens, names_filter=hook_name(2))
        hook = make_hook(vec, 8.0, "response")
        with model.hooks(fwd_hooks=[(hook_name(3), hook)]):
            _, injected_cache = model.run_with_cache(tokens, names_filter=hook_name(2))
    assert torch.equal(clean_cache[hook_name(2)], injected_cache[hook_name(2)])
