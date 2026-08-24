"""Unit and systematic generalization tests for Cellular Sheaf Binding (Thrust G, 0642.8)."""

from __future__ import annotations

from pathlib import Path
import torch

from bio_inspired_nanochat.sheaf_binding import (
    BindingCertificate,
    OperadicSNAREMatcher,
    SheafConsistencyMonitor,
    SheafDiffusionLayer,
    log_sheaf_audit,
)


def test_sheaf_laplacian_nullspace_and_certificate():
    """Sheaf Laplacian on a tree graph has dim(ker(L)) = stalk_dim and positive spectral gap."""
    stalk_dim = 4
    num_nodes = 3
    edges = [(0, 1), (1, 2)]

    laplacian = SheafConsistencyMonitor.build_sheaf_laplacian(
        edges=edges, num_nodes=num_nodes, stalk_dim=stalk_dim
    )

    cert = SheafConsistencyMonitor.evaluate_certificate(laplacian, stalk_dim=stalk_dim)

    assert cert.is_certified
    assert cert.dimension_kernel == stalk_dim
    assert cert.spectral_gap > 0.0


def test_sheaf_diffusion_convergence_reduces_dirichlet_energy():
    """Sheaf diffusion exponentially decreases Dirichlet energy towards ker(L)."""
    stalk_dim = 4
    num_nodes = 3
    edges = [(0, 1), (1, 2)]
    laplacian = SheafConsistencyMonitor.build_sheaf_laplacian(
        edges=edges, num_nodes=num_nodes, stalk_dim=stalk_dim
    )

    layer = SheafDiffusionLayer(d_model=stalk_dim, num_diffusion_steps=10, diffusion_rate=0.1)

    # Corrupted initial state
    x_init = torch.randn(1, num_nodes, stalk_dim)
    e_init = SheafConsistencyMonitor.compute_obstruction_energy(x_init, laplacian)

    x_diffused = layer(x_init, laplacian)
    e_final = SheafConsistencyMonitor.compute_obstruction_energy(x_diffused, laplacian)

    assert e_final < e_init
    assert e_final < 0.2 * e_init, f"Expected >80% energy reduction, got {e_final:.4f} vs {e_init:.4f}"


def test_operadic_snare_docking_affinity():
    """Operadic SNARE matcher computes high docking score for matched binary codes."""
    code_dim = 8
    matcher = OperadicSNAREMatcher(code_dim=code_dim)

    v_code = torch.randn(2, code_dim)
    t_code_matched = v_code.clone()
    t_code_orthogonal = -v_code

    score_matched = matcher.compute_docking_score(v_code, t_code_matched)
    score_unmatched = matcher.compute_docking_score(v_code, t_code_orthogonal)

    assert (score_matched > score_unmatched).all()


def test_sheaf_obstruction_auroc_on_compositional_battery(tmp_path: Path):
    """Sheaf obstruction energy achieves high AUROC (>= 0.95) on corrupted bindings."""
    stalk_dim = 4
    num_nodes = 4
    edges = [(0, 1), (1, 2), (2, 3)]
    laplacian = SheafConsistencyMonitor.build_sheaf_laplacian(
        edges=edges, num_nodes=num_nodes, stalk_dim=stalk_dim
    )

    clean_energies = []
    corrupted_energies = []

    for _ in range(50):
        # Clean binding: constant along edges (in kernel)
        base = torch.randn(1, 1, stalk_dim)
        x_clean = base.repeat(1, num_nodes, 1)
        clean_energies.append(SheafConsistencyMonitor.compute_obstruction_energy(x_clean, laplacian))

        # Corrupted binding: random inconsistent assignment
        x_corrupted = torch.randn(1, num_nodes, stalk_dim)
        corrupted_energies.append(SheafConsistencyMonitor.compute_obstruction_energy(x_corrupted, laplacian))

    auroc = SheafConsistencyMonitor.compute_binding_auroc(clean_energies, corrupted_energies)
    assert auroc >= 0.95, f"Expected AUROC >= 0.95, got {auroc:.4f}"

    # Verify audit logging
    cert = BindingCertificate(
        is_certified=True,
        h1_obstruction=0.0,
        spectral_gap=0.5,
        dimension_kernel=stalk_dim,
        step=100,
    )
    log_path = tmp_path / "audit.jsonl"
    log_sheaf_audit(cert, jsonl_path=log_path)
    assert log_path.exists()
