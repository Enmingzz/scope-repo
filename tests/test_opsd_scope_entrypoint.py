from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import train_opsd_scope as core
from opsd.visionzip_aokvqa import train as runtime
from opsd.visionzip_aokvqa.aokvqa import normalize_reasoning_answer_jsonl_sample
from opsd.visionzip_aokvqa.prompting import format_chat_messages
from opsd.visionzip_aokvqa.losses import compute_forward_kl, compute_per_token_generalized_jsd


def arguments(tmp_path, method="scopd+", *extra):
    return core.parser().parse_args([
        "--method", method, "--dataset", str(tmp_path / "train.jsonl"),
        "--image-root", str(tmp_path / "images"), "--output-dir", str(tmp_path / method),
        *extra,
    ])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("count", [1, 31, 32, 65])
def test_forward_loss_and_gradient_match_baseline(dtype, count):
    torch.manual_seed(42)
    teacher = torch.randn(count, 37, dtype=dtype, requires_grad=True)
    student = torch.randn(count, 37, dtype=dtype, requires_grad=True)
    actual = core.forward_kl(teacher, student)
    expected = compute_forward_kl(teacher.detach(), student)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.autograd.grad(actual, student, retain_graph=True)[0]
    reference = torch.autograd.grad(expected, student)[0]
    torch.testing.assert_close(grad, reference, rtol=0, atol=0)
    assert teacher.grad is None and actual.dtype == torch.float32


def test_jsd_exact_symmetric_bounded_and_detached():
    torch.manual_seed(42)
    p, plus = (torch.randn(65, 37, requires_grad=True) for _ in range(2))
    actual = core.budget_jsd(p, plus)
    expected = compute_per_token_generalized_jsd(p, plus, beta=0.5, temperature=1, chunk_size=32)
    torch.testing.assert_close(actual, expected.detach(), rtol=0, atol=0)
    torch.testing.assert_close(actual, core.budget_jsd(plus, p))
    torch.testing.assert_close(core.budget_jsd(p, p), torch.zeros(65), atol=1e-6, rtol=0)
    assert (actual >= 0).all() and (actual <= 0.693148).all()
    assert not actual.requires_grad and p.grad is None and plus.grad is None


def test_selection_ceil_ties_mask_eos_and_no_kl_floor():
    scores = torch.tensor([1., 1., 999., 0., 1., 999.], requires_grad=True)
    # Masked prompt/padding positions 2,5 must never be selected. EOS at 4 is valid.
    valid = torch.tensor([True, True, False, True, True, False])
    assert core.select_response_tokens(scores, valid, 0.1).tolist() == [0]
    assert core.select_response_tokens(scores, valid, 0.6).tolist() == [0, 1, 4]
    assert core.select_response_tokens(scores, valid, 1.0).tolist() == [0, 1, 3, 4]


def test_scopd_plus_matches_established_selection_and_gradients():
    path = core.ROOT / "experiments/llm_only/opsd_r010_budgetjsd_top10_gap_sweep_forwardkl_20260917/selection.py"
    if not path.exists():
        pytest.skip("Historical experiment reference is not part of the core-only checkout")
    spec = importlib.util.spec_from_file_location("reference_scope_selection", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.manual_seed(23)
    teacher, student, probe = (torch.randn(65, 37, requires_grad=True) for _ in range(3))
    valid = torch.ones(65, dtype=torch.bool)
    valid[[1, 4]] = False
    b = core.budget_jsd(student, probe)
    selected = core.select_response_tokens(b, valid, .1)
    loss = core.forward_kl(teacher[selected], student[selected])
    ref_loss, ref_b, _, ref_selected = reference.selected_forward_loss(teacher, student, probe, valid)
    torch.testing.assert_close(b, ref_b, rtol=0, atol=0)
    assert selected.tolist() == ref_selected.tolist()
    torch.testing.assert_close(loss, ref_loss, rtol=0, atol=0)
    grad = torch.autograd.grad(loss, student, retain_graph=True)[0]
    torch.testing.assert_close(grad, torch.autograd.grad(ref_loss, student)[0], rtol=0, atol=0)
    assert not grad[~torch.isin(torch.arange(65), selected)].any()


def test_configuration_pairing_batch_and_checkpoint_contract(tmp_path, monkeypatch):
    a = core.build_config(arguments(tmp_path, "scope"), 4)
    b = core.build_config(arguments(tmp_path, "scopd+"), 4)
    for key in ("base_model", "dataset", "training", "generation", "pruning", "paired_sampling"):
        assert a[key] == b[key]
    assert b["training"]["lora_dropout"] == 0
    assert b["training"]["gradient_accumulation_steps"] == 8
    assert b["training"]["max_steps"] // 32 == 320
    assert a["experiment"]["name"] == "scope"
    assert b["experiment"]["name"] == "scopd+"
    assert a["experiment"]["algorithm_version"] == b["experiment"]["algorithm_version"] == 2
    assert not a["opsd"]["budget_token_selection"]["enabled"]
    assert b["opsd"]["budget_token_selection"]["enabled"]
    assert b["opsd"]["budget_token_selection"]["delta"] == .01
    assert b["opsd"]["budget_token_selection"]["top_fraction"] == .1
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert core.build_config(arguments(tmp_path, "scopd+"), 4) == b
    assert core.build_config(arguments(tmp_path), 1)["training"]["gradient_accumulation_steps"] == 32
    resumed = deepcopy(b)
    resumed["training"].update(start_step=1024, adapter_path="resume")
    resumed["checkpointing"].update(resume_from="resume", stop_at_step=2048)
    assert runtime.checkpoint_contract_sha256(resumed) == runtime.checkpoint_contract_sha256(b)


@pytest.mark.parametrize("method", ["scope", "scopd+"])
def test_training_response_length_default_override_and_resume_contract(tmp_path, method):
    default = core.build_config(arguments(tmp_path, method), 4)
    legacy = core.build_config(arguments(tmp_path, method, "--max-new-tokens", "512"), 4)
    assert default["generation"]["max_new_tokens"] == 1024
    assert legacy["generation"]["max_new_tokens"] == 512
    assert runtime.checkpoint_contract_sha256(default) != runtime.checkpoint_contract_sha256(legacy)


def test_image_budget_units_and_resume_contract(tmp_path):
    current = core.build_config(arguments(tmp_path), 4)
    old = core.build_config(arguments(tmp_path, "scopd+", "--image-token-budget", "1080"), 4)
    assert current["generation"]["max_new_tokens"] == 1024
    assert current["dataset"]["min_pixels"] == current["dataset"]["max_pixels"] == 1003520
    assert old["dataset"]["min_pixels"] == old["dataset"]["max_pixels"] == 846720
    assert runtime.checkpoint_contract_sha256(current) != runtime.checkpoint_contract_sha256(old)


@pytest.mark.parametrize("extra", [
    ("--delta", "0"), ("--delta", "nan"), ("--delta", "0.95"),
    ("--retention", "0"), ("--top-fraction", "0"), ("--max-new-tokens", "0"),
    ("--effective-batch-size", "3"), ("--max-samples", "33"), ("--stop-at-samples", "10272"),
    ("--image-token-budget", "0"),
])
def test_rejects_invalid_config(tmp_path, extra):
    with pytest.raises(ValueError):
        core.build_config(arguments(tmp_path, "scopd+", *extra), 4)


def test_old_opsd_cli_name_is_rejected(tmp_path):
    with pytest.raises(SystemExit) as error:
        arguments(tmp_path, "opsd")
    assert error.value.code == 2


@pytest.mark.parametrize("method,old_method", [("scope", "opsd"), ("scopd+", "scope")])
def test_pre_rename_checkpoints_are_not_silently_reinterpreted(tmp_path, method, old_method):
    cfg = core.build_config(arguments(tmp_path, method), 1)
    checkpoint = Path(cfg["output_dir"]) / "resume_checkpoints/step_001024"
    checkpoint.mkdir(parents=True)
    (checkpoint / "COMPLETE").touch()
    old = deepcopy(cfg)
    old["experiment"].update(name=old_method, algorithm_version=1)
    (checkpoint / "trainer_state.json").write_text(json.dumps({
        "config_contract_sha256": runtime.checkpoint_contract_sha256(old),
        "global_step": 1024,
    }))
    cfg["checkpointing"]["resume_from"] = str(checkpoint)
    with pytest.raises(ValueError, match="does not match checkpoint contract"):
        runtime.prepare_resume_config(cfg, Path(cfg["output_dir"]))


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(9, 13))
        self.config = SimpleNamespace(use_cache=False)
        self.generation_config = SimpleNamespace(use_cache=False)

    def forward(self, input_ids, attention_mask=None, ratio=1., **kwargs):
        length = input_ids.shape[1] if ratio == 1 else input_ids.shape[1] - (3 if ratio == .1 else 2)
        bias = torch.arange(13).float() * ratio
        return SimpleNamespace(logits=(self.weight[:length] + bias).unsqueeze(0))


class GroundTruthForbiddenSample(SimpleNamespace):
    def __getattribute__(self, name):
        if name in {"answer", "target", "reasoning", "raw", "correct_letter", "correct_index"}:
            raise AssertionError(f"Training read forbidden GT field: {name}")
        return super().__getattribute__(name)


def test_gt_and_reference_reasoning_do_not_change_model_prompt():
    raw = {
        "sample_id": "same_question", "image": "same_image.png",
        "question": "What is visible?", "answer": "GT_SENTINEL_A",
        "ground_truth": "GT_SENTINEL_A", "target": "GT_SENTINEL_A",
        "processed_target": "GT_SENTINEL_A", "reasoning": "GT_SENTINEL_A",
        "rationale": "GT_SENTINEL_A", "prompt": "GT_SENTINEL_A",
    }
    changed = {key: ("GT_SENTINEL_B" if value == "GT_SENTINEL_A" else value)
               for key, value in raw.items()}
    a, b = [normalize_reasoning_answer_jsonl_sample(row, prompt_mode="thinking")
            for row in (raw, changed)]
    assert a.target != b.target and a.reasoning != b.reasoning
    assert a.prompt == b.prompt and a.question == b.question and a.image == b.image
    chat_a, chat_b = format_chat_messages(a.prompt), format_chat_messages(b.prompt)
    assert chat_a == chat_b
    assert "GT_SENTINEL" not in json.dumps(chat_a)


@pytest.mark.parametrize("method", ["scope", "scopd+"])
@pytest.mark.parametrize("privileged", [
    {"teacher_uses_ground_truth": True}, {"teacher_adapter_name": "gt_adapter"},
    {"teacher_model": object()}, {"fixed_rollout_token_ids": torch.tensor([[1]])},
])
def test_privileged_or_external_rollout_branches_are_rejected(tmp_path, method, privileged):
    cfg = core.build_config(arguments(tmp_path, method), 1)
    with pytest.raises(ValueError, match="on-policy, no-GT"):
        core.training_step(None, None, None, cfg, .1, **privileged)


@pytest.mark.parametrize("prompt_len", [4, 7, 12])
def test_qwen_decoder_future_tokens_do_not_change_earlier_predictions(prompt_len):
    if not torch.cuda.is_available():
        pytest.skip("FlashAttention causal test requires an allocated GPU")
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

    torch.manual_seed(42)
    device, dtype = "cuda", torch.bfloat16
    cfg = Qwen2_5_VLTextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
        attention_dropout=0.0, use_cache=False, pad_token_id=0,
    )
    cfg._attn_implementation = "flash_attention_2"
    model = Qwen2_5_VLTextModel(cfg).to(device=device, dtype=dtype).eval()
    ids = torch.arange(1, prompt_len + 5, device=device).reshape(1, -1)
    changed = ids.clone()
    # Change the second response token and everything after it. Predictions for
    # both the first and second response tokens must remain exactly unchanged.
    changed[:, prompt_len + 1:] += 20
    projection = torch.randn(32, 64, device=device, dtype=dtype)
    with torch.no_grad():
        a = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
        b = model(input_ids=changed, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
        logits_a, logits_b = a @ projection, b @ projection
        response_a = runtime.extract_generated_logits(logits_a, prompt_len, 4)
        response_b = runtime.extract_generated_logits(logits_b, prompt_len, 4)
    torch.testing.assert_close(response_a[:2], response_b[:2], rtol=0, atol=0)
    assert not torch.equal(response_a[2:], response_b[2:])


@pytest.mark.parametrize("prompt_len", [4, 7, 12])
def test_causal_response_slice_excludes_target_itself(prompt_len):
    ids = torch.arange(1, prompt_len + 5).reshape(1, -1)
    causal_logits = ids.cumsum(-1).unsqueeze(-1)
    response = runtime.extract_generated_logits(causal_logits, prompt_len, 4)
    for token_index in range(4):
        assert response[token_index, 0] == ids[0, :prompt_len + token_index].sum()


@pytest.mark.parametrize("ema", [False, True])
@pytest.mark.parametrize("method", ["scope", "scopd+"])
def test_step_prefix_gradients_modes_ema_and_one_rollout(tmp_path, monkeypatch, ema, method):
    torch.manual_seed(7)
    model = ToyModel().train()
    original = model.weight.detach().clone()
    shadow = {"weight": original + .3} if ema else None
    calls, generates = [], []
    ids = torch.tensor([[5, 6, 7, 8]])
    prompt = {"input_ids": torch.arange(5).reshape(1, 5), "attention_mask": torch.ones(1, 5, dtype=torch.long)}
    sample = SimpleNamespace(sample_id="toy", question="What?", prompt="Question only", options=[], correct_letter="")
    processor = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=8))
    cfg = core.build_config(arguments(tmp_path, method), 1)
    cfg["output_dir"] = ""

    def generate(*args, **kwargs):
        assert kwargs["max_new_tokens"] == 1024
        generates.append((model.training, torch.is_grad_enabled(), model.config.use_cache))
        return ids, "generated text", {"num_kept_visual_tokens": 10, "num_full_visual_tokens": 100}

    def forward(m, seq, retention_ratio, prompt_len, **kwargs):
        calls.append((retention_ratio, seq["input_ids"].clone(), m.training, torch.is_grad_enabled()))
        length = prompt_len if retention_ratio == 1 else prompt_len - (3 if retention_ratio == .1 else 2)
        return m(**seq, ratio=retention_ratio), {"metadata": {
            "student_prompt_len": length, "num_full_visual_tokens": 100,
            "num_kept_visual_tokens": int(round(retention_ratio * 100)),
        }}

    monkeypatch.setattr(runtime, "encode_prompt", lambda *a, **k: prompt)
    monkeypatch.setattr(runtime, "generate_pruned", generate)
    monkeypatch.setattr(runtime, "forward_pruned", forward)
    forbidden_sample = GroundTruthForbiddenSample(**vars(sample))
    loss, metrics = core.training_step(model, processor, forbidden_sample, cfg, .1,
                                       ema_shadow=shadow, rollout_seed=17)
    assert generates == [(False, False, True)]
    assert [x[0] for x in calls] == ([1., .1, .11] if method == "scopd+" else [1., .1])
    for ratio, prefix, mode, grad in calls:
        assert torch.equal(prefix[:, -4:], ids)
        assert grad == (ratio == .1)
        assert mode == (ratio == .1)
    assert model.training and not model.config.use_cache and not model.generation_config.use_cache
    assert torch.equal(original, model.weight)
    assert metrics["teacher_visual_tokens"] == 100
    assert metrics["selected_tokens"] == (1 if method == "scopd+" else 4)
    assert metrics["loss_type"] == (
        "scopd_plus_top_budget_jsd_forward_kl" if method == "scopd+" else "scope_nogt_forward_kl"
    )
    assert metrics["teacher_ground_truth_access"] is False
    assert metrics["generated_tokens"] == 4
    loss.backward()
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all()
    if method == "scope":
        actual_grad = model.weight.grad.clone()
        model.zero_grad()
        ref_loss, _ = runtime.opsd_nogt_step(model, processor, sample, cfg, .1,
                                            ema_shadow=shadow, rollout_seed=17)
        ref_loss.backward()
        torch.testing.assert_close(loss, ref_loss, rtol=0, atol=0)
        torch.testing.assert_close(actual_grad, model.weight.grad, rtol=0, atol=0)


def test_hooks_restore_even_on_failure():
    old_step, old_lora = runtime.opsd_nogt_step, runtime.apply_lora
    with pytest.raises(RuntimeError):
        with core.install_core(runtime):
            assert runtime.opsd_nogt_step is core.training_step
            raise RuntimeError("deliberate")
    assert runtime.opsd_nogt_step is old_step and runtime.apply_lora is old_lora


def test_preflight_refuses_overwrite_repeat_and_missing_resume(tmp_path):
    args = arguments(tmp_path, "scope", "--max-samples", "32")
    args.dataset.write_text('{"sample_id":"a"}\n')
    args.image_root.mkdir()
    cfg = core.build_config(args, 1)
    with pytest.raises(ValueError, match="Not enough"):
        core.prepare_run(cfg, False)
    args.dataset.write_text("".join(json.dumps({"sample_id": str(i)}) + "\n" for i in range(32)))
    with pytest.raises(FileNotFoundError, match="COMPLETE"):
        core.prepare_run(cfg, True)
    assert core.prepare_run(cfg, False)
    (args.output_dir / "existing.txt").write_text("do not overwrite")
    with pytest.raises(FileExistsError):
        core.prepare_run(cfg, False)
