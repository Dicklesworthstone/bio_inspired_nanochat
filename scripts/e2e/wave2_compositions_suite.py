r"""E2E SCRIPT: Wave-2 Capability-Frontier Compositions Verification Battery (beads re4e, eqyk.22).

Comprehensive verification of the 7 emergent Wave-2 compositions:
  1. ``self_correcting_generation_loop`` (re4e.1): Planted inconsistency detected -> deliberated ->
     repaired or explicitly abstained within iteration cap.
  2. ``metacognition_self_model`` (re4e.2): Calibrated tri-state self-model (known/guessing/unknown)
     grounded in physical free energy and entropy.
  3. ``energy_guided_search`` (re4e.3): Value-directed search using free energy F(z) achieving
     lower final energy than greedy rollout (F_search <= F_greedy).
  4. ``persistent_lifelong_memory`` (re4e.4): Working memory written in Session A, consolidated offline
     during sleep, recalled in Session B with user isolation and explicit forget.
  5. ``synaptic_serving_engine_sla`` (re4e.5): Dynamic capability knobs (ATP, deliberation) with
     strict SLA latency enforcement and explicit refusal paths.
  6. ``conformal_certified_abstention`` (re4e.10): Distribution-free conformal selective prediction
     guaranteeing error rate on answered queries <= alpha.
  7. ``speculative_decode_cheap_path`` (re4e.7): Cheap draft path verified by target model,
     logging valid accept rates and preserving generation integrity.

Run:
    python -m scripts.e2e.wave2_compositions_suite
    pytest tests/test_e2e_wave2_compositions.py -v
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.conformal_abstention import ConformalAbstentionEngine
from bio_inspired_nanochat.e2e_harness import InvariantResult
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.wave2_compositions import (
    EnergyGuidedSearch,
    MetacognitionController,
    PersistentLifelongMemory,
    SelfCorrectionConfig,
    SelfCorrectionLoop,
    ServingEngineSLA,
    ServingRequest,
    SpeculativeDecoder,
)


@dataclass
class Wave2CompositionsConfig:
    """Configuration for Wave-2 Compositions verification suite."""

    deliberation_max_iters: int = 5
    conformal_target_alpha: float = 0.15
    memory_dim: int = 32
    search_depth: int = 3
    seed: int = 42


@dataclass
class Wave2CompositionsReport:
    run_id: str
    config: Wave2CompositionsConfig
    passed: bool
    invariants: list[InvariantResult]
    summary: dict[str, Any] = field(default_factory=dict)

    def assert_passed(self) -> None:
        failed = [inv for inv in self.invariants if not inv.passed]
        if failed:
            msg = "\n".join(f"  FAILED: {inv.name} -> {inv.detail}" for inv in failed)
            raise AssertionError(f"Wave-2 Compositions battery failed with {len(failed)} failure(s):\n{msg}")


def _selective_risk_on_cyclic_language(cfg: Wave2CompositionsConfig) -> InvariantResult:
    """Calibrate on independent sequence starts, then enumerate the held-out query population."""
    from scripts.e2e.stochastic_thermo_uq import (
        ExperimentConfig, _make_model, _make_pool, _model_checkpoint_id, _softmax_prediction, _train_model,
    )

    experiment = ExperimentConfig(seed=cfg.seed)
    torch.manual_seed(cfg.seed)
    model = _make_model(experiment)
    # Fix a uniform distribution over the odd starts of a 16-symbol cyclic language.
    # Training sees even starts. Each query has one scored next token, at its final position.
    train = _make_pool(experiment, low=0, high=16, starts=torch.arange(0, 16, 2))
    held_out = _make_pool(experiment, low=0, high=16, starts=torch.arange(1, 16, 2))
    _train_model(model, train, experiment)
    with torch.inference_mode():
        probs = _softmax_prediction(model, torch.cat([x for x, _ in held_out])).probabilities[:, -1]
    labels = torch.cat([y for _, y in held_out])[:, -1]
    rng = torch.Generator().manual_seed(cfg.seed + 1001)
    indices = torch.randint(len(labels), (512,), generator=rng)
    policy = ConformalAbstentionEngine(cfg.conformal_target_alpha)
    threshold = policy.calibrate(probs[indices], labels[indices])
    decisions = [policy.predict(row) for row in probs]
    answered = [(decision.prediction, int(label)) for decision, label in zip(decisions, labels, strict=True)
                if not decision.abstained]
    errors = sum(pred != label for pred, label in answered)
    risk = errors / len(answered) if answered else None
    bound = policy.answered_risk_upper_bound
    valid = (risk is None and bound is None) or (risk is not None and bound is not None
                                                 and risk <= bound <= policy.target_alpha)
    return InvariantResult(
        name="conformal_certified_abstention", passed=valid,
        observed={"target_alpha": policy.target_alpha, "delta": policy.delta, "threshold": threshold,
                  "risk_upper_bound": bound, "population_error_given_answer": risk,
                  "calibration_draws": len(indices), "population_size": len(labels), "answered_count": len(answered),
                  "errors": errors, "abstention_rate": 1 - len(answered) / len(labels),
                  "checkpoint_sha256": _model_checkpoint_id(model)},
        detail=f"Answered {len(answered)}/{len(labels)} held-out cyclic starts with {errors} errors.",
    )


def run_wave2_compositions_e2e(
    cfg: Wave2CompositionsConfig | None = None,
    *,
    run_dir: Path | str | None = None,
    verbose: bool = True,
) -> Wave2CompositionsReport:
    """Run the complete Wave-2 Compositions verification battery."""
    if cfg is None:
        cfg = Wave2CompositionsConfig()

    console = Console(quiet=not verbose)
    run_id = f"wave2-compositions-e2e-{int(time.time())}"
    invariants: list[InvariantResult] = []

    clean_tmp = False
    if run_dir is None:
        base_dir = Path(tempfile.mkdtemp(prefix="wave2_compositions_e2e_"))
        clean_tmp = True
    else:
        base_dir = Path(run_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

    run_logger = RunLogger(base_dir, name="wave2_compositions_e2e", run_id=run_id, console=verbose)
    run_logger.event("wave2_compositions_config", config=asdict(cfg))

    try:
        torch.manual_seed(cfg.seed)

        # ===================================================================
        # 1. Self-Correcting Generation Loop (re4e.1)
        # ===================================================================
        # Synthetic problem: target sequence [10, 20, 30]. Corrupted initial: [10, 999, 30].
        def _mock_detect(tokens: list[int]) -> float:
            # High obstruction if corrupted token 999 is present
            return 0.85 if 999 in tokens else 0.10

        def _mock_energy(tokens: list[int]) -> float:
            # Free energy: lower when matching target [10, 20, 30]
            target = [10, 20, 30]
            diff = sum(abs(a - b) for a, b in zip(tokens, target))
            return float(diff * 0.1)

        def _mock_candidates(tokens: list[int], step: int) -> list[list[int]]:
            # Propose candidate replacements for corrupted position
            if step == 0:
                return [[10, 50, 30], [10, 20, 30]]
            return [[10, 20, 30]]

        self_corr = SelfCorrectionLoop(
            SelfCorrectionConfig(max_iters=cfg.deliberation_max_iters, obstruction_threshold=0.5)
        )
        corr_res = self_corr.run(
            initial_seq=[10, 999, 30],
            detect_fn=_mock_detect,
            energy_fn=_mock_energy,
            candidate_fn=_mock_candidates,
        )

        corr_ok = (
            corr_res.repaired
            and not corr_res.abstained
            and corr_res.final_energy < corr_res.initial_energy
            and tuple(corr_res.tokens) == (10, 20, 30)
            and corr_res.iterations <= cfg.deliberation_max_iters
        )
        invariants.append(
            InvariantResult(
                name="self_correcting_generation_loop",
                passed=corr_ok,
                observed={
                    "repaired": corr_res.repaired,
                    "abstained": corr_res.abstained,
                    "iterations": corr_res.iterations,
                    "initial_energy": corr_res.initial_energy,
                    "final_energy": corr_res.final_energy,
                    "final_tokens": corr_res.tokens,
                },
                detail=(
                    f"Planted error [10, 999, 30] repaired to {corr_res.tokens} in {corr_res.iterations} iters; "
                    f"Free energy lowered ({corr_res.initial_energy:.2f} -> {corr_res.final_energy:.2f})"
                ),
            )
        )

        # ===================================================================
        # 2. Metacognition & Self-Model (re4e.2)
        # ===================================================================
        meta_ctrl = MetacognitionController()
        # Known condition: low energy (0.1) + low entropy (0.3)
        score_known = meta_ctrl.assess(free_energy=0.1, entropy=0.3, obstruction=0.0)
        # Unknown condition: high energy (2.5) + high entropy (2.0) + high obstruction (1.0)
        score_unknown = meta_ctrl.assess(free_energy=2.5, entropy=2.0, obstruction=1.0)

        meta_ok = (
            score_known.verdict == "know"
            and score_known.p_known > 0.5
            and score_unknown.verdict == "unknown"
            and score_unknown.p_unknown > 0.5
        )
        invariants.append(
            InvariantResult(
                name="metacognition_self_model",
                passed=meta_ok,
                observed={
                    "known_verdict": score_known.verdict,
                    "known_p_known": score_known.p_known,
                    "unknown_verdict": score_unknown.verdict,
                    "unknown_p_unknown": score_unknown.p_unknown,
                },
                detail=(
                    f"Known span correctly classified (p_known={score_known.p_known:.2f}, verdict={score_known.verdict}); "
                    f"Unknown span correctly classified (p_unknown={score_unknown.p_unknown:.2f}, verdict={score_unknown.verdict})"
                ),
            )
        )

        # ===================================================================
        # 3. Energy-Guided Search & Planning (re4e.3)
        # ===================================================================
        def _search_energy(cand_seq: list[int]) -> float:
            # Energy landscape with a deep basin at sequence [1, 2, 3, 4]
            optimal = [1, 2, 3, 4]
            dist = sum((a - b) ** 2 for a, b in zip(cand_seq, optimal))
            matches_optimal = tuple(cand_seq) == (1, 2, 3, 4)
            return float(dist * 0.2 + (0.0 if matches_optimal else 1.0))

        def _search_expand(tokens: list[int]) -> list[tuple[int, float]]:
            # Next token candidates with mock prior logits
            pos = len(tokens)
            return [(pos + 1, 1.0), (pos + 5, 2.0), (pos + 10, 0.5)]

        energy_search = EnergyGuidedSearch(
            energy_fn=_search_energy,
            branch_factor=3,
            max_depth=cfg.search_depth,
        )
        search_res = energy_search.search(start_tokens=[1], expand_fn=_search_expand)

        search_ok = (
            search_res.best_energy <= search_res.greedy_energy
            and search_res.energy_reduction >= 0.0
            and len(search_res.best_tokens) == 1 + cfg.search_depth
        )
        invariants.append(
            InvariantResult(
                name="energy_guided_search",
                passed=search_ok,
                observed={
                    "best_tokens": search_res.best_tokens,
                    "best_energy": search_res.best_energy,
                    "greedy_tokens": search_res.greedy_tokens,
                    "greedy_energy": search_res.greedy_energy,
                    "energy_reduction": search_res.energy_reduction,
                    "nodes_evaluated": search_res.nodes_evaluated,
                },
                detail=(
                    f"Energy-guided search found path {search_res.best_tokens} with energy {search_res.best_energy:.3f} "
                    f"<= greedy energy {search_res.greedy_energy:.3f} (reduction={search_res.energy_reduction:.3f})"
                ),
            )
        )

        # ===================================================================
        # 4. Persistent Lifelong Memory & Multi-Session Consolidation (re4e.4)
        # ===================================================================
        memory = PersistentLifelongMemory(dim=cfg.memory_dim)
        v_fact_user1 = torch.randn(cfg.memory_dim)
        v_fact_user2 = torch.randn(cfg.memory_dim)

        # Session A: write working memory for two distinct users
        memory.write_fast_memory("user-alice", "favorite_color_blue", v_fact_user1)
        memory.write_fast_memory("user-bob", "favorite_food_pizza", v_fact_user2)

        # Offline sleep consolidation pass
        cons_alice = memory.consolidate_sleep("user-alice")
        cons_bob = memory.consolidate_sleep("user-bob")

        # Session B: Recall with user isolation
        recalled_alice_key, sim_alice = memory.recall("user-alice", v_fact_user1)
        recalled_bob_key, sim_bob = memory.recall("user-bob", v_fact_user2)

        # Cross-user isolation check: Alice should not recall Bob's key
        cross_user_key, _ = memory.recall("user-alice", v_fact_user2)

        # Forget check
        memory.forget("user-alice", "favorite_color_blue")
        recalled_after_forget, _ = memory.recall("user-alice", v_fact_user1)

        lifelong_ok = (
            cons_alice == 1
            and cons_bob == 1
            and recalled_alice_key == "favorite_color_blue"
            and sim_alice > 0.99
            and recalled_bob_key == "favorite_food_pizza"
            and sim_bob > 0.99
            and cross_user_key != "favorite_food_pizza"
            and recalled_after_forget is None
        )
        invariants.append(
            InvariantResult(
                name="persistent_lifelong_memory",
                passed=lifelong_ok,
                observed={
                    "alice_recalled_key": recalled_alice_key,
                    "alice_similarity": sim_alice,
                    "bob_recalled_key": recalled_bob_key,
                    "cross_user_isolated": cross_user_key != "favorite_food_pizza",
                    "forget_succeeded": recalled_after_forget is None,
                },
                detail=(
                    f"Session A write + sleep consolidation + Session B recall ok (sim={sim_alice:.4f}); "
                    f"Multi-tenant isolation verified; Explicit forget verified"
                ),
            )
        )

        # ===================================================================
        # 5. Synaptic Serving Engine & SLA (re4e.5)
        # ===================================================================
        def _mock_executor(req: ServingRequest) -> tuple[str, int, int]:
            spent_atp = min(req.atp_budget, 15)
            delib_iters = min(req.deliberation_budget, 4)
            return f"Processed[{req.prompt}]", spent_atp, delib_iters

        engine = ServingEngineSLA(compute_executor=_mock_executor)

        # Standard feasible request
        req_valid = ServingRequest(
            request_id="req-001",
            user_id="user-1",
            prompt="summarize article",
            atp_budget=50,
            deliberation_budget=5,
            max_latency_ms=200.0,
        )
        resp_valid = engine.handle_request(req_valid)

        # Unfeasible SLA request (0 ms budget)
        req_unfeasible = ServingRequest(
            request_id="req-002",
            user_id="user-1",
            prompt="summarize article",
            atp_budget=0,
            max_latency_ms=0.0,
        )
        resp_unfeasible = engine.handle_request(req_unfeasible)

        serving_ok = (
            resp_valid.status == "success"
            and resp_valid.atp_spent == 15
            and resp_valid.deliberation_iters == 4
            and resp_unfeasible.status == "refused_sla"
        )
        invariants.append(
            InvariantResult(
                name="synaptic_serving_engine_sla",
                passed=serving_ok,
                observed={
                    "valid_status": resp_valid.status,
                    "valid_atp_spent": resp_valid.atp_spent,
                    "valid_delib_iters": resp_valid.deliberation_iters,
                    "unfeasible_status": resp_unfeasible.status,
                },
                detail=(
                    f"Valid request executed within SLA ({resp_valid.status}, atp={resp_valid.atp_spent}); "
                    f"Unfeasible request refused cleanly ({resp_unfeasible.status})"
                ),
            )
        )

        # ===================================================================
        # 6. Conformal Certified Abstention (re4e.10)
        # ===================================================================
        selective = _selective_risk_on_cyclic_language(cfg)
        invariants.append(selective)

        # ===================================================================
        # 7. Speculative Decoder Cheap Path (re4e.7)
        # ===================================================================
        def _mock_draft(prefix: list[int]) -> list[int]:
            # Cheap draft generates [101, 102, 103]
            return [101, 102, 103]

        def _mock_verify(prefix: list[int], drafts: list[int]) -> list[bool]:
            # Target model accepts [101, 102] but rejects [103]
            return [d in [101, 102] for d in drafts]

        spec_decoder = SpeculativeDecoder(
            draft_model_fn=_mock_draft,
            verify_model_fn=_mock_verify,
        )
        spec_res = spec_decoder.decode_step(prefix=[1, 2], k_draft=3)

        spec_ok = (
            tuple(spec_res.tokens) == (1, 2, 101, 102)
            and spec_res.drafted_count == 3
            and spec_res.accepted_count == 2
            and abs(spec_res.accept_rate - 2.0 / 3.0) < 1e-4
        )
        invariants.append(
            InvariantResult(
                name="speculative_decode_cheap_path",
                passed=spec_ok,
                observed={
                    "drafted_count": spec_res.drafted_count,
                    "accepted_count": spec_res.accepted_count,
                    "accept_rate": spec_res.accept_rate,
                    "final_tokens": spec_res.tokens,
                },
                detail=(
                    f"Speculative decode drafted {spec_res.drafted_count} tokens, "
                    f"accepted {spec_res.accepted_count} (accept_rate={spec_res.accept_rate:.2%}); "
                    f"Generated tokens: {spec_res.tokens}"
                ),
            )
        )

        for inv in invariants:
            run_logger.event("e2e_invariant", **asdict(inv))

        all_passed = all(inv.passed for inv in invariants)
        report = Wave2CompositionsReport(
            run_id=run_id,
            config=cfg,
            passed=all_passed,
            invariants=invariants,
            summary={
                "self_correct_iters": corr_res.iterations,
                "metacognition_verdict": score_known.verdict,
                "energy_search_reduction": search_res.energy_reduction,
                "lifelong_recalled_sim": sim_alice,
                "serving_valid_status": resp_valid.status,
                "conformal_q_hat": selective.observed["threshold"],
                "speculative_accept_rate": spec_res.accept_rate,
            },
        )

        if verbose:
            table = Table(title="Wave-2 Capability-Frontier Compositions Verification Battery")
            table.add_column("Composition Invariant", style="cyan")
            table.add_column("Status", style="bold")
            table.add_column("Detail", style="dim")
            for inv in invariants:
                status = "[green]PASS[/green]" if inv.passed else "[red]FAIL[/red]"
                table.add_row(inv.name, status, inv.detail)
            console.print(table)

        return report

    finally:
        run_logger.close()
        if clean_tmp:
            shutil.rmtree(base_dir, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Wave-2 Capability Frontier Compositions E2E battery")
    parser.add_argument("--run-dir", type=str, default=None, help="Directory to save E2E traces and logs")
    parser.add_argument("--delib-iters", type=int, default=5, help="Max deliberation iters for self-correction")
    parser.add_argument("--conformal-alpha", type=float, default=0.15, help="Conformal target alpha error")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args(argv)

    cfg = Wave2CompositionsConfig(
        deliberation_max_iters=args.delib_iters,
        conformal_target_alpha=args.conformal_alpha,
        seed=args.seed,
    )
    report = run_wave2_compositions_e2e(cfg, run_dir=args.run_dir, verbose=True)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
