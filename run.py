import numpy as np
import re

from three_body_collision import ThreeHardParticlesRing, animate_three_particles_on_ring
from position_prob import time_weighted_position_hist_per_particle, plot_position_histograms
from momentum_state_prob import time_weighted_momentum_hist_per_particle, plot_momentum_histograms


def _parse_csv_floats(s, n=None):
    """
    Parse comma/space-separated floats.
    Example: "1, 0.9, 1.1" or "1 0.9 1.1"
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty input.")

    parts = [p for p in re.split(r"[,\s]+", s) if p]
    vals = np.array([float(p) for p in parts], dtype=float)

    if n is not None and len(vals) != n:
        raise ValueError(f"Expected {n} values, got {len(vals)}.")

    return vals


def _prompt(msg, default=None):
    if default is None:
        return input(msg + ": ").strip()
    out = input(f"{msg} [{default}]: ").strip()
    return out if out else default


def _prompt_int(msg, default):
    while True:
        try:
            return int(_prompt(msg, str(default)))
        except ValueError:
            print("Please enter an integer.")


def _prompt_float(msg, default):
    while True:
        try:
            return float(_prompt(msg, str(default)))
        except ValueError:
            print("Please enter a number.")


def _prompt_vec3(msg, default):
    while True:
        try:
            return _parse_csv_floats(_prompt(msg, default), n=3)
        except ValueError as e:
            print(f"Invalid input: {e}")


def _prompt_yesno(msg, default=True):
    d = "y" if default else "n"
    while True:
        s = _prompt(msg + " (y/n)", d).lower()
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False
        print("Please type y or n.")


def _auto_gaps(L_free):
    """
    Deterministic default gaps summing exactly to L_free.
    """
    w = np.array([0.2, 0.35, 0.45], dtype=float)
    w = w / w.sum()
    h = L_free * w
    h[-1] = L_free - (h[0] + h[1])  # enforce exact sum
    return h



def _prompt_simulation_inputs():
    """
    Prompt for parameters common to animation and histogram runs.
    Returns dict of kwargs for ThreeHardParticlesRing.
    """
    # --- ask SR first ---
    use_SR = _prompt_yesno("Use special relativity?", default=False)

    if use_SR:
        c = _prompt_float("Speed of light c (use 1 for natural units)", 1.0)
        K_rel = _prompt_float("Relativistic internal kinetic energy K_rel", 1.0)
    else:
        # dummy placeholders (not used when use_SR=False)
        c = 1.0
        K_rel = 0.0

    L = _prompt_float("Ring length L", 1.0)
    rod_length = _prompt_float("Rod length a (0 for point particles)", 0.0)
    x1 = _prompt_float("Initial x1 in [0, L)", 0.5)

    m = _prompt_vec3("Masses m1,m2,m3", "1.01, 1.0, 0.99")
    v = _prompt_vec3("Velocities v1,v2,v3", "0.7, -0.2, 0.1")

    L_free = L - 3.0 * rod_length
    if L_free < 0:
        raise ValueError("Invalid geometry: L - 3a must be >= 0.")

    print(f"Gaps h0,h1,h2 must sum to L_free = L - 3a = {L_free:g}")

    h = _auto_gaps(L_free)
    print(f"Auto gaps: {h[0]:.12g}, {h[1]:.12g}, {h[2]:.12g} (sum={h.sum():.12g})")

    if _prompt_yesno("Override gaps manually?", default=False):
        while True:
            h = _prompt_vec3("Enter h0,h1,h2", f"{h[0]}, {h[1]}, {h[2]}")
            if np.isclose(h.sum(), L_free, rtol=0.0, atol=1e-12):
                break
            print(f"h.sum() = {h.sum():.15g} but must equal {L_free:.15g}. Please re-enter.")

    return {
        "L": L,
        "m": m,
        "v": v,
        "h": h,
        "rod_length": rod_length,
        "x1": x1,
        "use_SR": use_SR,
        "c": c,
        "K_rel": K_rel,
    }



def run_animation(sim_kwargs):
    print("\n--- Animation setup ---")
    t_end = _prompt_float("Animation duration t_end", 30.0)
    dt_sample = _prompt_float("Sampling time step dt_sample", 0.005)
    interval_ms = _prompt_int("Animation frame interval (ms)", 15)

    sim = ThreeHardParticlesRing(**sim_kwargs)
    sim.normalise_com_energy()

    times, xs = sim.sample_for_animation(t_end=t_end, dt_sample=dt_sample)
    animate_three_particles_on_ring(times, xs, L=sim.L, interval_ms=interval_ms)


def run_position_hist(sim_kwargs):
    print("\n--- Position histogram setup ---")
    n_collisions = _prompt_int("Number of collisions", 1_000_000)
    burn_in = _prompt_int("Burn-in collisions", 10_000)
    n_bins_x = _prompt_int("Number of position bins", 1000)

    sim = ThreeHardParticlesRing(**sim_kwargs)

    edges_x, prob_x = time_weighted_position_hist_per_particle(
        sim,
        n_collisions=n_collisions,
        n_bins=n_bins_x,
        burn_in=burn_in
    )

    print("Position histogram completed.")
    plot_position_histograms(edges_x, prob_x, L=sim.L)


def run_momentum_hist(sim_kwargs):
    print("\n--- Momentum histogram setup ---")
    n_collisions = _prompt_int("Number of collisions", 10_000_000)
    burn_in = _prompt_int("Burn-in collisions", 10_000)
    n_bins_p = _prompt_int("Number of momentum bins", 1000)

    sim = ThreeHardParticlesRing(**sim_kwargs)

    edges_p, prob_p, total_time, p_max = time_weighted_momentum_hist_per_particle(
        sim,
        n_collisions=n_collisions,
        n_bins=n_bins_p,
        burn_in=burn_in
    )

    print("Momentum histogram completed.")
    plot_momentum_histograms(sim, edges_p, prob_p, p_max)


def main():
    print("=== Three hard particles on a ring: runner ===")
    print("Choose what to run:")
    print("  1) Animation")
    print("  2) Position histogram")
    print("  3) Momentum histogram")

    choice = _prompt("Enter 1/2/3")

    if choice not in {"1", "2", "3"}:
        print("Invalid choice. Exiting.")
        return

    sim_kwargs = _prompt_simulation_inputs()

    if choice == "1":
        run_animation(sim_kwargs)
    elif choice == "2":
        run_position_hist(sim_kwargs)
    elif choice == "3":
        run_momentum_hist(sim_kwargs)

    print("All done.")


if __name__ == "__main__":
    main()