import numpy as np
import matplotlib.pyplot as plt

from three_body_collision import HardParticlesRing   



def accumulate_time_for_particle(bin_time_weights, x, v, dt_col, L, edges, n_bins):
    """
    accumulates how much time is spent by one particle in each position bin in the time until any two particles collide
    this function updates one row at a time of the array time_in_each_bin
    
    bin_time_weights is now only a 1d array in this function - total time that the particle has spent in each bin
    x is the initial position of the particle after the last collision
    v is the speed of the particle for this segment

    due to periodic boundary conditions, we will always track position in bin relative to left edge
    """


    def what_bin_your_in(position):
        '''
        this function identifies which bin you are in
        edges[k] <= x < edges[k+1]
        '''

        bin_index = int(np.searchsorted(edges, position, side="right") - 1)
        '''
        returns the index of the edge greater than x
        subtracting by 1 returns the bin index since edges = n_bins + 1
        '''

        bin_index = int(np.clip(bin_index, 0, n_bins - 1))   #due to numerical rounding we could get indices -1 or n_bins so must clip if this happens
        
        return bin_index


    def partial_bins(current_bin, left_edge_distance, bin_time_weights, bin_width, v):
        '''
        this is to accumulate the time spent in the start and finish bins of the segment
        where the particle wont have spent the full bin time = width / velocity
        '''

        if v > 0:
            time_to_right_edge = (bin_width - left_edge_distance) / v
            bin_time_weights[current_bin] += time_to_right_edge 
    
        else:
            time_to_left_edge =  (left_edge_distance) / -v
            bin_time_weights[current_bin] += time_to_left_edge


    initial_bin = what_bin_your_in(x)   #bin index for bin your in at position x
    bin_width = L / n_bins

    final_position = (x + (dt_col * v)) % L
    final_bin = what_bin_your_in(final_position)

    #if it has stayed in the same bin add the whole time contribution to that bin
    if initial_bin == final_bin:
        bin_time_weights[initial_bin] += dt_col
        return

    else:
        left_edge_initial = edges[initial_bin]   #position of the left edge of the initial bin
        left_edge_initial_distance = x - left_edge_initial   #distance from left edge of initial bin
        left_edge_final = edges[final_bin]   #position of the left edge of the final bin
        left_edge_final_distance = final_position - left_edge_final   #distance from left edge of final bin

        time_full_bin = bin_width / abs(v)   #time to cross full bin

        if v > 0.0:
            n_bins_inbetween = (final_bin - initial_bin) % n_bins - 1
            bins_inbetween = [ (initial_bin + i) % n_bins for i in range(1, n_bins_inbetween + 1) ]   #creates list of indices of middle bins

        else:       #for negative velocity
            n_bins_inbetween = (initial_bin - final_bin) % n_bins - 1
            bins_inbetween = [ (initial_bin - i) % n_bins for i in range(1, n_bins_inbetween + 1) ]   #creates list of indices of middle bins

        '''
        add full bin crossing time to each bin inbetween
        if bins_inbetween is empty this still works
        '''
        for i in bins_inbetween:  
            bin_time_weights[i] += time_full_bin

        partial_bins(initial_bin, left_edge_initial_distance, bin_time_weights, bin_width, v)
        partial_bins(final_bin, left_edge_final_distance, bin_time_weights, bin_width, -v)   #negative velocity required to get time spent in final bin

    return bin_time_weights
    
 

def time_weighted_position_hist_per_particle(sim, n_collisions, n_bins, burn_in):
    """
    calculates the probability for each particle to be within a certain bin

    dt_col is the time between the last collision to the next collision
    k_star is the index of the shortest time until collision 
    """

    sim.normalise_COM_energy()   #put in COM frame and set E=1 

    L = sim.L
    edges = np.linspace(0.0, L, n_bins + 1)   #n_bins+1 edges including 0 and L

    bin_time_weights = np.zeros((sim.N, n_bins), dtype=float)   #array storing how much time is spent in each bin for each particle
    time_after_burn_in = 0.0   #initialising the total simulation time excluding burn in period


    for n in range(n_collisions):
        dt_col, k_star = sim.next_event()   #returns time for the next collision and index for that time in the times array

        if n >= burn_in:
            '''
            this loop calls accumulator function, to measure time spent by eacj particle in each bin for all the segments

            this block is skipped for burn in period
            first get positions of all particles at the beginning of this segment
            reconstruct positions of particle 2 and 3 from position of 1 and gaps
            '''

            xs = np.empty(sim.N, dtype=float)
            xs[0] = sim.x1   #position of particle 1

            for i in range(1, sim.N):   #starts at 1 because we already have xs[0] = x1
                distance = sim.rod_length + float(sim.h[i - 1])   #the distance between particles accounting for rod length
                xs[i] = (xs[i - 1] + distance) % L

            '''
            d12 = sim.rod_length + float(sim.h[0])
            d23 = sim.rod_length + float(sim.h[1])
            x2 = (x1 + d12) % L
            x3 = (x2 + d23) % L
            xs = (x1, x2, x3)   #initial positions of all particles for the beginning of this segment
            '''

            #---accumulate exact time in bins for each particle over this segment
            for j in range(sim.N):
                accumulate_time_for_particle(bin_time_weights[j], xs[j], sim.v[j], dt_col, L, edges, n_bins)

            time_after_burn_in += dt_col   #adds the time it took for the next collision to the total time excluding burn in

        sim.advance(dt_col)   #updates all gaps, absolute position of particle 1 and total sim time
        sim.collide(k_star)   #this updates velocities post collision

    prob = bin_time_weights / time_after_burn_in   #gives probability each particle is in each position bin
 
    return edges, prob



def plot_position_histograms(sim, edges, prob, L):
    """plot time-fraction per position bin for each particle"""

    centres = 0.5 * (edges[:-1] + edges[1:])   #computes the midpoint of each bin
    #---edges[:-1] is every element apart from the last element and edges[1:] is everything apart from the first

    fig, axes = plt.subplots(sim.N, 1, figsize=(7, 9), sharex=True)

    for i in range(sim.N):
        axes[i].plot(centres, prob[i], drawstyle="steps-mid")
        axes[i].set_ylabel(f"Particle {i+1}\nTime fraction")
        axes[i].set_xlim(0.0, L)
        #axes[i].grid(alpha=0.3)

    axes[-1].set_xlabel("Position x on ring")
    fig.suptitle("Exact time-weighted position histograms")
    plt.tight_layout()
    plt.show()


def plot_position_histograms_overlay_K(histogram_data, L):
    """
    Overlay position histograms for N=3 at multiple relativistic K values.

    Parameters
    ----------
    histogram_data : list of tuples
        Each entry should be:
        (K_rel, edges, prob)
    L : float
        Ring length
    """
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    for K_rel, edges, prob in histogram_data:
        centres = 0.5 * (edges[:-1] + edges[1:])

        for i in range(3):
            axes[i].plot(centres, prob[i], drawstyle="steps-mid", label=f"K={K_rel:g}")
            axes[i].set_ylabel(f"Particle {i+1}\nTime fraction")
            axes[i].set_xlim(0.0, L)

    axes[-1].set_xlabel("Position x on ring")
    axes[0].legend()
    fig.suptitle("Exact time-weighted position histograms (SR, N=3)")

    plt.tight_layout()
    plt.show()


def plot_position_histograms_overlay_particles(sim, edges, prob, L):
    """Overlay all particle position histograms on one plot."""

    centres = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(8, 5))

    for i in range(sim.N):
        ax.plot(centres, prob[i], drawstyle="steps-mid", label=f"Particle {i+1}")

    ax.set_xlim(0.0, L)
    ax.set_xlabel("Position x on ring")
    ax.set_ylabel("Time fraction")
    ax.legend()

    if sim.use_SR:
        ax.set_title("Exact time-weighted position histograms (overlay, SR)")
    else:
        ax.set_title("Exact time-weighted position histograms (overlay, non-relativistic)")

    plt.tight_layout()
    plt.show()