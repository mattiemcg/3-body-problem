# Three Hard Particles on a Ring  
## Event-Driven Simulation with Optional Special Relativity + Ergodicity Diagnostics

This project implements an **event-driven (collision-to-collision)** simulator for **three 1D hard particles (rods)** moving on a **periodic ring** of length `L`.

The code supports:

- Newtonian dynamics  
- Special relativistic dynamics (exact 1D elastic collisions via 4-momentum conservation)

It also provides tools to compute **time-weighted (residence-time) histograms** for:

- Positions (centre coordinates on `[0, L)`)  
- Momenta (per-particle momentum distributions in the COM frame)

The simulation is fully event-driven: collisions are computed exactly (no time discretisation).

---

# Repository Contents

## `three_body_collision.py`

Core physics engine and animation utilities.

### Special Relativity Support

When `use_SR=True`, the simulator uses:

- Lorentz factor  

\[
\gamma = \frac{1}{\sqrt{1 - v^2/c^2}}
\]

- Exact 1D elastic collision via:
  1. Compute total lab-frame energy \(E\) and momentum \(P\)
  2. Boost to COM frame \(V = c^2 P / E\)
  3. Reverse velocities in COM frame
  4. Boost back to lab frame

This guarantees exact conservation of:

- Total energy  
- Total momentum  
- On-shell mass condition  

No approximations are used.

---

## Simulator Class

### `ThreeHardParticlesRing`

State is stored in reduced form:

- `x1` — absolute position of particle 1 (wrapped mod `L`)
- `h[0], h[1], h[2]` — neighbour gaps
- `v[0], v[1], v[2]` — velocities
- `m[0], m[1], m[2]` — masses
- `rod_length = a`
- `use_SR` — toggle relativistic dynamics
- `c` — speed of light
- `K_rel` — relativistic internal kinetic energy (used only when `use_SR=True`)

---

## Gap Convention

\[
h_0 + h_1 + h_2 = L - 3a
\]

where:

- `h[0]` = gap from particle 1 → 2  
- `h[1]` = gap from particle 2 → 3  
- `h[2]` = gap from particle 3 → 1  

Constraint:

\[
L - 3a \ge 0
\]

---

## Event-Driven Stepping

- `dt, k = sim.next_event()`  
  Time to next collision and which gap closes.

- `sim.advance(dt)`  
  Advance positions and gaps exactly.

- `sim.collide(k)`  
  Perform elastic collision:
  - Newtonian formula (if `use_SR=False`)
  - Exact relativistic 4-momentum solution (if `use_SR=True`)

---

## COM + Energy Normalisation

### Newtonian Case

1. Shift to COM frame  
2. Rescale velocities so total kinetic energy:

\[
E = 1
\]

---

### Relativistic Case

1. Boost to relativistic COM frame  
2. Target total COM energy:

\[
E_{\text{target}} = \sum_i m_i c^2 + K_{\text{rel}}
\]

3. Solve for scaling factor \( \alpha \) via bisection such that:

\[
\sum_i \gamma(\alpha v_i) m_i c^2 = E_{\text{target}}
\]

This guarantees:

- \( P_{\text{COM}} = 0 \)
- Total relativistic energy equals target

---

## Animation

```python
times, xs = sim.sample_for_animation(t_end, dt_sample)
animate_three_particles_on_ring(times, xs, L, interval_ms)
```

Note: `FuncAnimation(...)` must use the keyword `init_func=...`.

---

# `position_prob.py`

Computes **exact time-weighted position histograms**.

## Main Function

```python
time_weighted_position_hist_per_particle(
    sim,
    n_collisions,
    n_bins,
    burn_in
)
```

Features:

- Exact segment-by-segment time accumulation  
- No time discretisation  
- Burn-in period support  
- Uniform bins on `[0, L]`

---

## Important Assumption

The accumulator assumes:

\[
|v| \cdot dt < L
\]

i.e. a particle does **not complete multiple full laps** in one collision-to-collision segment.

If ultra-relativistic speeds make multi-wrap segments possible, the accumulator must be extended.

---

# `momentum_state_prob.py`

Computes **time-weighted momentum histograms**.

Momentum is constant between collisions.

---

## Momentum Bounds

### Newtonian (after normalisation to \(E=1\))

\[
|p_i| \le \sqrt{2 m_i}
\]

---

### Relativistic

Given target COM energy:

\[
E_{\text{target}}
\]

Maximum allowed momentum per particle:

\[
p_{\max,i} =
\frac{1}{c}
\sqrt{E_{\text{target}}^2 - (m_i c^2)^2}
\]

Momentum is computed as:

\[
p_i = \gamma_i m_i v_i
\]

---

## Main Function

```python
time_weighted_momentum_hist_per_particle(
    sim,
    n_collisions,
    n_bins,
    burn_in
)
```

Returns:

- `edges`
- `probability`
- `total_time`
- `p_max`

---

# `run.py`

Interactive CLI runner.

You can run exactly one task at a time:

1. Animation  
2. Position histogram  
3. Momentum histogram  

---

## Simulation Inputs

The CLI prompts for:

- Use special relativity? (y/n)
- If yes:
  - Speed of light `c`
  - Relativistic internal kinetic energy `K_rel`
- Ring length `L`
- Rod length `a`
- Initial `x1`
- Masses `m1,m2,m3`
- Velocities `v1,v2,v3`
- Gaps `h0,h1,h2`

Gaps must satisfy:

\[
h_0 + h_1 + h_2 = L - 3a
\]

Auto-generated gaps are provided by default and can be overridden.

---

# Installation

```bash
conda create -n three-ring python=3.11 numpy matplotlib
conda activate three-ring
```

Run:

```bash
python run.py
```

---

# Physics Summary

This system is a minimal nontrivial model of:

- Hard-core many-body dynamics  
- Deterministic chaos  
- Microcanonical ergodicity  
- Relativistic few-body scattering (when enabled)

It preserves:

- Exact collision times  
- Exact conservation laws  
- Exact time-weighted observables  
