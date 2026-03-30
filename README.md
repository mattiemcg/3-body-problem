# 3-body-problem

Event-driven simulation of hard particles on a 1D ring, with support for both non-relativistic and special-relativistic elastic collisions.

The code can:
- simulate hard-particle dynamics on a periodic ring,
- animate particle motion,
- compute exact time-weighted position histograms,
- compute time-weighted momentum histograms,
- compare behaviour across different system sizes and relativistic energy scales.

Although the repository name refers to the 3-body problem, the code is written more generally for N hard particles on a ring.

Project structure

- three_body_collision.py
  Core simulation code. Defines the HardParticlesRing class, collision rules, centre-of-momentum normalisation, and both Newtonian and relativistic collision updates.

- animation.py
  Samples trajectories uniformly in time and animates particles moving on the ring.

- position_prob.py
  Computes exact time-weighted position histograms for each particle.

- momentum_state_prob.py
  Computes time-weighted momentum histograms for each particle.

- run.py
  Interactive entry point for running animations, histograms, and comparison/overlay studies.

Features

1. Event-driven hard-particle dynamics
The simulation evolves the system collision-to-collision rather than using a fixed timestep for dynamics.
Between collisions, particles move at constant velocity and gaps evolve linearly.

2. Periodic ring geometry
Particles move on a ring of length L, so positions are wrapped into the interval [0, L).

3. Hard rods or point particles
You can choose:
- rod_length = 0 for point particles,
- rod_length > 0 for finite-size hard rods.

4. Newtonian or special-relativistic collisions
The code supports:
- exact 1D elastic collisions in the Newtonian case,
- exact 1D elastic collisions using relativistic 4-momentum conservation in the SR case.

5. Histogram analysis
The project includes tools to measure long-time statistical behaviour:
- position probability distributions,
- momentum probability distributions,
- overlay plots for comparing particles, system sizes, or relativistic energies.

Requirements

This project uses:
- Python 3
- NumPy
- Matplotlib

Install dependencies with:

pip install numpy matplotlib

How to run

The main entry point is:

python run.py

This opens an interactive menu with the following options:

1. Animation
2. Position histogram
3. Momentum histogram
4. Both position and momentum histograms
5. Overlay particle 1 momentum histograms for fixed system sizes
6. Special-relativistic position overlays for fixed N=3 and multiple K values

You will then be prompted to enter simulation parameters such as:
- number of particles,
- ring length,
- rod length,
- initial position,
- masses,
- velocities,
- gap sizes,
- whether to use special relativity.

Simulation parameters

Important inputs include:
- N — number of particles
- L — ring length
- m — particle masses
- v — initial velocities
- h — free gaps between neighbouring particles
- rod_length — particle diameter / rod length
- x1 — initial position of particle 1
- use_SR — whether to use special relativity
- K_rel — relativistic internal kinetic energy target

Geometric constraint

The gaps must satisfy:

sum(h) = L - N * rod_length

This ensures the particles fit consistently on the ring.

Physics implemented

Non-relativistic case
For Newtonian dynamics, collisions are updated using the exact 1D elastic collision formulas, and the total centre-of-mass kinetic energy is normalised to 1.

Relativistic case
For special relativity, the code:
- boosts into the centre-of-momentum frame,
- rescales internal momenta to match a chosen target energy,
- performs exact two-body elastic collisions using relativistic velocity transformations.

Outputs

Depending on the selected mode, the code can produce:
- an animation of particles moving around the ring,
- per-particle position histograms,
- per-particle momentum histograms,
- overlay plots comparing:
  - all particles in a single run,
  - particle 1 across multiple system sizes,
  - N=3 systems at different relativistic energy scales.

Example workflow

To generate a momentum histogram:

python run.py

Then choose:

3

and enter the requested parameters.

To generate an animation, run the same script and choose:

1

Notes

- The code is general in particle number N, despite the repository name.
- The simulation is event-driven, so collision times are computed exactly from gap-closing times.
- Histogram probabilities are time-weighted rather than simple counts, so they reflect the fraction of time spent in each bin.

Possible future improvements

- Save plots automatically to files
- Add command-line arguments instead of interactive prompts
- Add unit tests for collision and normalisation routines
- Add support for exporting animation files
- Add documentation for the statistical and physical interpretation of the histograms

Author

Mattie McG
