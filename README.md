# Three Hard Particles on a Ring — Event-Driven Simulation + Ergodicity Diagnostics

This project implements an **event-driven (collision-to-collision)** simulator for **three 1D hard particles (rods)** moving on a **periodic ring** of length `L`, plus tools to compute **time-weighted (residence-time)** histograms for:
- **positions** (centre coordinates on `[0, L)`), and
- **momenta** (per-particle momentum distributions after normalisation to `E=1` and `P=0`).

---

## Repository Contents

### `three_body_collision.py`
Core physics + animation utilities.

What it provides (high level):
- `ThreeHardParticlesRing`: simulator class. State is stored in a reduced form:
  - `x1`: absolute position of particle 1 (wrapped mod `L`)
  - `h`: neighbour gaps (particle 2 and 3 positions are reconstructed from `x1` and `h`)
  - `v`: velocities
  - `m`: masses
  - `rod_length`: rod length `a`
- Event-driven stepping:
  - `dt, k = sim.next_event()` — time to next collision and which pair collides
  - `sim.advance(dt)` — advance state to the collision time
  - `sim.collide(k)` — update velocities for an elastic collision
- Normalisation:
  - `sim.normalise_com_energy()` — transform to COM frame and rescale so total kinetic energy `E=1`
- Animation helpers:
  - `times, xs = sim.sample_for_animation(t_end, dt_sample)`
  - `animate_three_particles_on_ring(times, xs, L, interval_ms=...)`

**Matplotlib note:** if you edit the animation code, make sure `FuncAnimation(...)` uses the keyword `init_func=...` (the keyword must be `init_func`, the function name itself can be `initialise`, `_init_`, etc.).

---

### `position_erg.py`
Computes **time-weighted position histograms** (residence time) for each particle.

Key functions:
- `time_weighted_position_hist_per_particle(sim, n_collisions, n_bins, burn_in)`
  - Runs the event-driven simulation for `n_collisions` collision events.
  - Skips the first `burn_in` collisions.
  - Accumulates **exact time spent in each position bin** for each particle.
  - Returns `(edges, prob)` where `prob.shape == (3, n_bins)`.

- `plot_position_histograms(edges, prob, L)`
  - Step plots of the three per-particle position histograms.

Implementation notes:
- Uses **uniform bins**: `edges = linspace(0, L, n_bins+1)`.
- Reconstructs particle positions at the start of each collision-to-collision segment using gaps:
  - `x2 = x1 + (a + h0)`, `x3 = x2 + (a + h1)` (all wrapped mod `L`)
- The segment accumulator splits each segment into:
  - **initial partial bin**, **middle full bins**, **final partial bin**

**Assumption (important):**
- The current “first/middle/last” bin logic assumes a particle does **not** traverse more than one full lap in a single collision-to-collision segment, i.e. effectively `|v| * dt < L`.
- If you ever allow `|v| * dt >= L`, you must extend the accumulator to handle multi-wrap segments.

---

### `momentum_erg.py`
Computes **time-weighted momentum histograms** for each particle.

Key functions:
- `make_momentum_edges_per_particle(m, n_bins)`
  - Uses the bound after normalisation to `E=1`:
    - `|p_i| <= sqrt(2 m_i E)` → with `E=1`: `pmax_i = sqrt(2 m_i)`
  - Returns `(edges, p_max)`.

- `time_weighted_momentum_hist_per_particle(sim, n_collisions, n_bins, burn_in)`
  - During each collision-to-collision segment, momenta are constant.
  - Adds the segment duration `dt` to the momentum bin containing each `p_i = m_i v_i`.
  - Returns `(edges, probability, time_after_burn_in, p_max)`.

- `plot_momentum_histograms(edges, prob, pmax)`
  - Per-particle momentum histogram plots.

---

### `run.py`
Interactive runner script (CLI) to run **one** task at a time:
- Animation **or**
- Position histogram **or**
- Momentum histogram

It prompts for:
- Simulation inputs: `L`, `rod_length`, `x1`, masses `m`, velocities `v`
- Gaps `h0,h1,h2` (must satisfy the constraint below)
- Histogram inputs (only when doing histograms): `n_collisions`, `burn_in`, and number of bins
- Animation inputs (only when doing animation): `t_end`, `dt_sample`, `interval_ms`

Gaps are auto-generated to satisfy the constraint exactly, with an optional prompt to override manually.

---

## Physics / Parameter Constraints

### Gap constraint (hard rods on a ring)
Let rod length be `a`. The three gaps must sum to the free length:

\[
h_0 + h_1 + h_2 = L - 3a
\]

This must be non-negative, so you must also have:

\[
L - 3a \ge 0
\]

---

## Installation

Create and activate a conda environment (example):

```bash
conda create -n three-ring python=3.11 numpy matplotlib
conda activate three-ring
