import numpy as np
import re

from three_body_collision import HardParticlesRing
from animation import sample_for_animation, animate_particles_on_ring
from position_prob import time_weighted_position_hist_per_particle, plot_position_histograms, plot_position_histograms_overlay_K, plot_position_histograms_overlay_particles
from momentum_state_prob import time_weighted_momentum_hist_per_particle, plot_momentum_histograms, plot_particle1_histogram, plot_particle1_histograms_overlay, plot_momentum_histograms_overlay_K, plot_momentum_histograms_overlay_particles


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


def _prompt_yesno(msg, default=True):
    d = "y" if default else "n"
    while True:
        s = _prompt(msg + " (y/n)", d).lower()
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False
        print("Please type y or n.")


def _prompt_vecN(msg, N, default_text):
    while True:
        try:
            return _parse_csv_floats(_prompt(msg, default_text), n=N)
        except ValueError as e:
            print(f"Invalid input: {e}")


def _auto_gaps(L_free, N):
    """
    Deterministic default gaps summing exactly to L_free.
    Returns an array of length N.
    """
    if N <= 0:
        raise ValueError("N must be positive.")

    w = np.arange(1, N + 1, dtype=float)
    w /= w.sum()

    h = L_free * w
    h[-1] = L_free - np.sum(h[:-1])
    return h


def _default_masses_text(N):
    if N == 3:
        return "1.01, 1.0, 0.99"
    return ", ".join(["1.0"] * N)


def _default_velocities_text(N):
    if N == 3:
        return "0.2, -0.4, 0.2"
    vals = np.zeros(N, dtype=float)
    if N >= 2:
        vals[0] = 0.2
        vals[1] = -0.2
    return ", ".join(f"{x:g}" for x in vals)


def _special_mass_ratios(N):
    ratios = np.arange(1, N + 1, dtype=float)
    return ratios


def _special_velocity_pattern(N):
    base = np.array([0.2, 1.0, -1.0, -0.2, 0.6, -0.6, 0.9, -0.9], dtype=float)
    vals = np.resize(base, N).astype(float)
    return vals


def _random_values(N, low, high, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(low, high, size=N)


def _format_vec(vec):
    return ", ".join(f"{x:.12g}" for x in vec)


def _prompt_parameter_vector(name, N, *, equal_value, random_low, random_high,
                             special_generator=None, custom_default_text=None):
    """
    Build an N-component vector for masses or velocities.

    Modes:
      1) equal values
      2) random values in a specified range
      3) special preset pattern
      4) manual entry
    """
    print(f"\nChoose {name} input mode:")
    print(f"  1) Equal {name}")
    print(f"  2) Random {name} in [{random_low}, {random_high}]")
    if special_generator is not None:
        print(f"  3) Special {name} pattern")
        print(f"  4) Enter {name} manually")
        valid = {"1", "2", "3", "4"}
    else:
        print(f"  3) Enter {name} manually")
        valid = {"1", "2", "3"}

    while True:
        choice = _prompt("Enter choice", "2")
        if choice not in valid:
            print("Invalid choice.")
            continue

        if choice == "1":
            value = _prompt_float(f"Enter common {name[:-2] if name.endswith('es') else name[:-1]} value", equal_value)
            return np.full(N, value, dtype=float)

        if choice == "2":
            seed_text = _prompt("Random seed (blank for fresh random draw)", "")
            rng = np.random.default_rng(None if seed_text == "" else int(seed_text))
            vals = _random_values(N, random_low, random_high, rng=rng)
            print(f"Generated {name}: {_format_vec(vals)}")
            return vals

        if special_generator is not None and choice == "3":
            vals = np.asarray(special_generator(N), dtype=float)
            print(f"Special {name}: {_format_vec(vals)}")
            return vals

        default_text = custom_default_text or _format_vec(np.full(N, equal_value, dtype=float))
        return _prompt_vecN(f"Enter {N} {name}", N, default_text)


def _prompt_simulation_inputs(fixed_N=None, fixed_use_SR=None, fixed_K_rel=None):
    """
    Prompt for parameters common to animation and histogram runs.
    Returns dict of kwargs for HardParticlesRing.
    """
    if fixed_N is None:
        number_of_particles = _prompt_int("How many particles in the system?", 3)
    else:
        number_of_particles = int(fixed_N)
        print(f"\nUsing fixed particle number N = {number_of_particles}")

    if fixed_use_SR is None:
        use_SR = _prompt_yesno("Use special relativity?", default=False)
    else:
        use_SR = bool(fixed_use_SR)
        print(f"Using fixed special relativity setting: {use_SR}")

    if use_SR:
        if fixed_K_rel is None:
            K_rel = _prompt_float("Relativistic internal kinetic energy K_rel", 1.0)
        else:
            K_rel = float(fixed_K_rel)
            print(f"Using fixed relativistic internal kinetic energy K_rel = {K_rel}")
    else:
        K_rel = 0.0

    L = _prompt_float("Ring length L", 1.0)
    rod_length = _prompt_float("Rod length a (0 for point particles)", 0.0)
    x1 = _prompt_float("Initial x1 in [0, L)", 0.5)

    m = _prompt_parameter_vector(
        "masses",
        number_of_particles,
        equal_value=1.0,
        random_low=0.8,
        random_high=1.2,
        special_generator=_special_mass_ratios,
        custom_default_text=_default_masses_text(number_of_particles),
    )

    v = _prompt_parameter_vector(
        "velocities",
        number_of_particles,
        equal_value=0.0,
        random_low=-0.9999999,
        random_high=0.9999999,
        special_generator=_special_velocity_pattern,
        custom_default_text=_default_velocities_text(number_of_particles),
    )

    L_free = L - number_of_particles * rod_length
    if L_free < 0:
        raise ValueError(f"Invalid geometry: L - N*a must be >= 0, but got {L_free}.")

    use_auto_gaps = _prompt_yesno("Auto-generate gaps summing to L - N*a?", default=True)
    if use_auto_gaps:
        h = _auto_gaps(L_free, number_of_particles)
        print(f"Generated gaps: {_format_vec(h)}")
    else:
        while True:
            h = _auto_gaps(L_free, number_of_particles)
            h = _prompt_vecN(
                f"Enter {number_of_particles} gaps",
                number_of_particles,
                ", ".join(f"{x:.12g}" for x in h)
            )
            if np.isclose(h.sum(), L_free, rtol=0.0, atol=1e-12):
                break
            print(f"h.sum() = {h.sum():.15g} but must equal {L_free:.15g}. Please re-enter.")

    return {
        "N": number_of_particles,
        "L": L,
        "m": m,
        "v": v,
        "h": h,
        "rod_length": rod_length,
        "x1": x1,
        "use_SR": use_SR,
        "K_rel": K_rel,
    }

def run_animation(sim_kwargs):
    print("\n--- Animation setup ---")
    t_end = _prompt_float("Animation duration t_end", 30.0)
    dt_sample = _prompt_float("Sampling time step dt_sample", 0.001)
    interval_ms = _prompt_int("Animation frame interval (ms)", 1)

    sim = HardParticlesRing(**sim_kwargs)
    sim.normalise_COM_energy()

    times, xs = sample_for_animation(sim, t_end=t_end, dt_sample=dt_sample)
    animate_particles_on_ring(times, xs, L=sim.L, interval_ms=interval_ms)


def run_position_hist(sim_kwargs):
    print("\n--- Position histogram setup ---")
    n_collisions = _prompt_int("Number of collisions", 1_000_000)
    burn_in = _prompt_int("Burn-in collisions", 10_000)
    n_bins_x = _prompt_int("Number of position bins", 1000)

    sim = HardParticlesRing(**sim_kwargs)

    edges_x, prob_x = time_weighted_position_hist_per_particle(
        sim,
        n_collisions=n_collisions,
        n_bins=n_bins_x,
        burn_in=burn_in
    )

    print("Position histogram completed.")
    plot_position_histograms(sim, edges_x, prob_x, L=sim.L)
    plot_position_histograms_overlay_particles(sim, edges_x, prob_x, L=sim.L)


def run_momentum_hist(sim_kwargs):
    print("\n--- Momentum histogram setup ---")
    n_collisions = _prompt_int("Number of collisions", 10_000_000)
    burn_in = _prompt_int("Burn-in collisions", 10_000)
    n_bins_p = _prompt_int("Number of momentum bins", 1000)

    sim = HardParticlesRing(**sim_kwargs)

    edges_p, prob_p, total_time, p_max = time_weighted_momentum_hist_per_particle(
        sim,
        n_collisions=n_collisions,
        n_bins=n_bins_p,
        burn_in=burn_in
    )

    print("Momentum histogram completed.")
    plot_momentum_histograms(sim, edges_p, prob_p, p_max)
    plot_momentum_histograms_overlay_particles(sim, edges_p, prob_p, p_max)
    plot_particle1_histogram(sim, edges_p, prob_p, p_max)


def run_both_hists(sim_kwargs):
    print("\n--- Combined histogram setup ---")
    n_collisions_x = _prompt_int("Number of collisions for position histogram", 1_000_000)
    n_collisions_p = _prompt_int("Number of collisions for momentum histogram", 10_000_000)
    burn_in_x = _prompt_int("Burn-in collisions for position histogram", 10_000)
    burn_in_p = _prompt_int("Burn-in collisions for momentum histogram", 10_000)
    n_bins_x = _prompt_int("Number of position bins", 1000)
    n_bins_p = _prompt_int("Number of momentum bins", 1000)

    sim_pos = HardParticlesRing(**sim_kwargs)
    sim_mom = HardParticlesRing(**sim_kwargs)

    print("Running position histogram...")
    edges_x, prob_x = time_weighted_position_hist_per_particle(
        sim_pos,
        n_collisions=n_collisions_x,
        n_bins=n_bins_x,
        burn_in=burn_in_x
    )
    print("Position histogram completed.")

    print("Running momentum histogram...")
    edges_p, prob_p, total_time, p_max = time_weighted_momentum_hist_per_particle(
        sim_mom,
        n_collisions=n_collisions_p,
        n_bins=n_bins_p,
        burn_in=burn_in_p
    )
    print("Momentum histogram completed.")

    print("Displaying position histogram...")
    plot_position_histograms(sim_pos, edges_x, prob_x, L=sim_pos.L)
    plot_position_histograms_overlay_particles(sim_pos, edges_x, prob_x, L=sim_pos.L)

    print("Displaying momentum histogram...")
    plot_momentum_histograms(sim_mom, edges_p, prob_p, p_max)
    plot_momentum_histograms_overlay_particles(sim_mom, edges_p, prob_p, p_max)
    plot_particle1_histogram(sim_mom, edges_p, prob_p, p_max) 



def run_particle1_overlay_fixed_sizes():
    print("\n--- Particle 1 overlay histogram setup (SR, K_rel = 10) ---")
    sizes = [4, 5, 8, 10, 30]

    n_collisions = _prompt_int("Number of collisions", 10_000_000)
    burn_in = _prompt_int("Burn-in collisions", 10_000)
    n_bins_p = _prompt_int("Number of momentum bins", 1000)

    histogram_data = []

    for N in sizes:
        print(f"\n=== Configure system for N = {N} ===")
        sim_kwargs = _prompt_simulation_inputs(
            fixed_N=N,
            fixed_use_SR=True,
            fixed_K_rel=10.0,
        )
        sim = HardParticlesRing(**sim_kwargs)

        edges_p, prob_p, total_time, p_max = time_weighted_momentum_hist_per_particle(
            sim,
            n_collisions=n_collisions,
            n_bins=n_bins_p,
            burn_in=burn_in
        )

        histogram_data.append((N, edges_p, prob_p, p_max))
        print(f"Completed momentum histogram for N = {N}")

    plot_particle1_histograms_overlay(histogram_data)



def run_three_particle_K_overlays():
    print("\n--- SR position overlay setup for N = 3 and K = 0.00001, 0.0001, 0.0001, 0.001, 0.01, 1.0, 100.0, 10000.0 ---")

    K_values = [0.00001, 0.001, 1.0, 10000.0]
    N = 3

    n_collisions_x = _prompt_int("Number of collisions for position histogram", 1_000_000)
    burn_in_x = _prompt_int("Burn-in collisions for position histogram", 10_000)
    n_bins_x = _prompt_int("Number of position bins", 1000)

    print("\nUsing fixed settings: N = 3, special relativity = True")

    L = _prompt_float("Ring length L", 1.0)
    rod_length = _prompt_float("Rod length a (0 for point particles)", 0.0)
    x1 = _prompt_float("Initial x1 in [0, L)", 0.5)

    m = _prompt_parameter_vector(
        "masses",
        N,
        equal_value=1.0,
        random_low=0.8,
        random_high=1.2,
        special_generator=_special_mass_ratios,
        custom_default_text=_default_masses_text(N),
    )

    v = _prompt_parameter_vector(
        "velocities",
        N,
        equal_value=0.0,
        random_low=-0.9999999,
        random_high=0.9999999,
        special_generator=_special_velocity_pattern,
        custom_default_text=_default_velocities_text(N),
    )

    L_free = L - N * rod_length
    if L_free < 0:
        raise ValueError(f"Invalid geometry: L - N*a must be >= 0, but got {L_free}.")

    print(f"Gaps must sum to L_free = L - N*a = {L_free:g}")

    h = _auto_gaps(L_free, N)
    print(f"Auto gaps: {', '.join(f'{x:.12g}' for x in h)} (sum={h.sum():.12g})")

    if _prompt_yesno("Override gaps manually?", default=False):
        while True:
            h = _prompt_vecN(
                f"Enter {N} gaps",
                N,
                ", ".join(f"{x:.12g}" for x in h)
            )
            if np.isclose(h.sum(), L_free, rtol=0.0, atol=1e-12):
                break
            print(f"h.sum() = {h.sum():.15g} but must equal {L_free:.15g}. Please re-enter.")

    base_sim_kwargs = {
        "N": N,
        "L": L,
        "m": m,
        "v": v,
        "h": h,
        "rod_length": rod_length,
        "x1": x1,
        "use_SR": True,
    }

    position_data = []

    for K_rel in K_values:
        print(f"\nRunning K_rel = {K_rel:g}")

        sim_pos = HardParticlesRing(**base_sim_kwargs, K_rel=K_rel)

        edges_x, prob_x = time_weighted_position_hist_per_particle(
            sim_pos,
            n_collisions=n_collisions_x,
            n_bins=n_bins_x,
            burn_in=burn_in_x
        )

        position_data.append((K_rel, edges_x, prob_x))

    print("Displaying position overlays...")
    plot_position_histograms_overlay_K(position_data, L)


def main():
    print("=== Hard particles on a ring: runner ===")
    print("Choose what to run:")
    print("  1) Animation")
    print("  2) Position histogram")
    print("  3) Momentum histogram")
    print("  4) Both position and momentum histograms")
    print("  5) Overlay particle 1 momentum histograms for N = 4, 5, 8, 10, 30 (SR, K=10)")
    print("  6) SR overlays for N = 3 and K = 0.00001, 0.001, 1.0, 10000.0")

    choice = _prompt("Enter 1/2/3/4/5/6")

    if choice not in {"1", "2", "3", "4", "5", "6"}:
        print("Invalid choice. Exiting.")
        return

    if choice == "5":
        run_particle1_overlay_fixed_sizes()
        return

    if choice == "6":
        run_three_particle_K_overlays()
        return

    sim_kwargs = _prompt_simulation_inputs()

    if choice == "1":
        run_animation(sim_kwargs)
    elif choice == "2":
        run_position_hist(sim_kwargs)
    elif choice == "3":
        run_momentum_hist(sim_kwargs)
    elif choice == "4":
        run_both_hists(sim_kwargs)


if __name__ == "__main__":
    main()
