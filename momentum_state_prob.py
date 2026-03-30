import numpy as np
import matplotlib.pyplot as plt

from three_body_collision import HardParticlesRing



def make_momentum_edges_per_particle(sim, m, n_bins):

    if sim.use_SR:
        p_max = (1.0 / sim.c) * np.sqrt(np.maximum(sim.E_target**2 - (m * sim.c**2)**2, 0.0))
    else:
        p_max = np.sqrt(2.0 * m)   #this outputs a 3 element array

    edges = np.array([np.linspace(-p, p, n_bins + 1) for p in p_max])
    return edges, p_max



def time_weighted_momentum_hist_per_particle(sim, n_collisions, n_bins, burn_in):
    """calculates the probability for each particle to be within a certain bin"""

    sim.normalise_COM_energy()   #put in COM frame and set E=target

    edges, p_max = make_momentum_edges_per_particle(sim, sim.m, n_bins)   #fixed bin edges (same for all initial conditions) 

    time_in_each_bin = np.zeros((sim.N, n_bins), dtype=float)   #array stores the amount of time for each bin for each particle
    time_after_burn_in = 0.0

    for n in range(n_collisions):
        dt_col, k_star = sim.next_event()   #time left before next collision and index of the smallest entry in the times array

        if n >= burn_in:   #only start adding weighting after the burn in period
            
            if sim.use_SR:
                g = 1.0 / np.sqrt(1.0 - (sim.v * sim.v) / (sim.c * sim.c))
                p = g * sim.m * sim.v
            else:
                p = sim.m * sim.v

            for i in range(sim.N):   #0, 1, 2
                index_of_interval = np.searchsorted(edges[i], p[i], side="right") - 1
                """
                returns index of which bin the momentum falls in 
                minus 1 to get the index of the correct interval
                """
                index_of_interval = int(np.clip(index_of_interval, 0, n_bins - 1))   #if due to rounding it was outwith the range it is clipped back into
                time_in_each_bin[i, index_of_interval] += dt_col

            time_after_burn_in += dt_col   #add time passed to total time

        sim.advance(dt_col)   #this updates all gaps, absolute position of mass 1 and total sim time
        sim.collide(k_star)   #this updates velocities post collision

    probability = time_in_each_bin / time_after_burn_in   #this normalises the time spent in each bin

    return edges, probability, time_after_burn_in, p_max



def plot_momentum_histograms(sim, edges, prob, pmax):
    """plot discrete probability for momentum bins"""

    fig, axes = plt.subplots(sim.N, 1, figsize=(7, 9), sharex=False)
    '''
    sharex means each particle can have a different momentum range as each particle has a different pmax
    3,1 means 3 rows, 1 columns - so 3 plots stacked on top of each other
    '''

    for i in range(sim.N):   #for each particles histogram
    
        centres = 0.5 * (edges[i, :-1] + edges[i, 1:])   #converts bin edges to midpoints
        #---first term removes last edge bin, second term removes first edge - for the j'th you sum left and right

        axes[i].plot(centres, prob[i], drawstyle="steps-mid")   #step-mid for histogram
        axes[i].set_xlim(-pmax[i], pmax[i])
        axes[i].set_ylabel(f"Particle {i+1}\nTime fraction")

        #axes[i].grid(alpha=0.3)

    axes[-1].set_xlabel("Momentum p")
    if sim.use_SR:
        E_rest = float(np.sum(sim.m) * sim.c**2)
        E_target = E_rest + float(sim.K_rel)
        fig.suptitle(f"Time-weighted momentum histograms (SR, P=0, E_COM,target={E_target:.6g})")
    else:
        fig.suptitle("Time-weighted momentum histograms (P=0, E=1)")
    plt.tight_layout()
    plt.show()


def plot_particle1_histogram(sim, edges, prob, pmax):
    """plot discrete probability for momentum bins for particle 1 only"""

    i = 0   # particle 1 has index 0

    fig, ax = plt.subplots(figsize=(7, 4))

    centres = 0.5 * (edges[i, :-1] + edges[i, 1:])   # bin midpoints

    ax.plot(centres, prob[i], drawstyle="steps-mid")
    ax.set_xlim(-pmax[i], pmax[i])
    ax.set_xlabel("Momentum p")
    ax.set_ylabel("Particle 1\nTime fraction")

    if sim.use_SR:
        E_rest = float(np.sum(sim.m) * sim.c**2)
        E_target = E_rest + float(sim.K_rel)
        ax.set_title(f"Particle 1 momentum histogram (SR, P=0, E_COM,target={E_target:.6g})")
    else:
        ax.set_title("Particle 1 momentum histogram (P=0, E=1)")

    plt.tight_layout()
    plt.show()



def plot_particle1_histograms_overlay(histogram_data):
    """
    Overlay particle 1 momentum histograms for multiple system sizes.

    Parameters
    ----------
    histogram_data : list of tuples
        Each entry should be:
        (N, edges, prob, pmax)
        where edges/prob/pmax are the outputs from
        time_weighted_momentum_hist_per_particle(...)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    max_abs_p = 0.0

    for N, edges, prob, pmax in histogram_data:
        centres = 0.5 * (edges[0, :-1] + edges[0, 1:])
        ax.plot(centres, prob[0], drawstyle="steps-mid", label=f"N={N}")
        max_abs_p = max(max_abs_p, float(pmax[0]))

    ax.set_xlim(-max_abs_p, max_abs_p)
    ax.set_xlabel("Momentum p")
    ax.set_ylabel("Particle 1\nTime fraction")
    ax.set_title("Particle 1 momentum histograms for N = 4, 5, 8, 10")
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_momentum_histograms_overlay_K(histogram_data):
    """
    Overlay momentum histograms for N=3 at multiple relativistic K values.

    Parameters
    ----------
    histogram_data : list of tuples
        Each entry should be:
        (K_rel, edges, prob, pmax)
    """
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=False)

    max_p = np.zeros(3, dtype=float)

    for K_rel, edges, prob, pmax in histogram_data:
        for i in range(3):
            centres = 0.5 * (edges[i, :-1] + edges[i, 1:])
            axes[i].plot(centres, prob[i], drawstyle="steps-mid", label=f"K={K_rel:g}")
            axes[i].set_ylabel(f"Particle {i+1}\nTime fraction")
            max_p[i] = max(max_p[i], float(pmax[i]))

    for i in range(3):
        axes[i].set_xlim(-max_p[i], max_p[i])

    axes[-1].set_xlabel("Momentum p")
    axes[0].legend()
    fig.suptitle("Time-weighted momentum histograms (SR, P=0, N=3)")

    plt.tight_layout()
    plt.show()


def plot_momentum_histograms_overlay_particles(sim, edges, prob, pmax):
    """Overlay all particle momentum histograms on one plot."""

    fig, ax = plt.subplots(figsize=(8, 5))

    max_abs_p = float(np.max(pmax))

    for i in range(sim.N):
        centres = 0.5 * (edges[i, :-1] + edges[i, 1:])
        ax.plot(centres, prob[i], drawstyle="steps-mid", label=f"Particle {i+1}")

    ax.set_xlim(-max_abs_p, max_abs_p)
    ax.set_xlabel("Momentum p")
    ax.set_ylabel("Time fraction")
    ax.legend()

    if sim.use_SR:
        E_rest = float(np.sum(sim.m) * sim.c**2)
        E_target = E_rest + float(sim.K_rel)
        ax.set_title(f"Time-weighted momentum histograms (overlay, SR, P=0, E_COM,target={E_target:.6g})")
    else:
        ax.set_title("Time-weighted momentum histograms (overlay, non-relativistic, P=0, E=1)")

    plt.tight_layout()
    plt.show()