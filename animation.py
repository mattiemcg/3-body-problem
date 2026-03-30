import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  

from three_body_collision import HardParticlesRing, calculate_gamma_factor 



def sample_for_animation(sim, t_end, dt_sample):
    """
    uniform-in-time sampling of n positions for animation        
    this outputs all the n times of each sample and all N particles positions at each time

    n positions are gathered for all N particles uniformly between t_initial = 0 and t_end
    n = t_end / dt_sample

    Returns:
    times: (n, 1)
    xs   : (n, N) positions in [0, L)
    """

    t_end = float(t_end)   #converts input into float, length of animation
    dt_sample = float(dt_sample)   #frame size of animation

    times = []   #creates a list of times
    xs = []   #creates a list

    t_next = sim.t   #next time you record the frame


    while t_next < t_end: 
        dt_col, k_star = sim.next_event()   #dt_col is the time until the next collision, and not going to use k_star
        t_col = sim.t + dt_col if np.isfinite(dt_col) else np.inf   #calculates absolute time of collision


        while t_next < min(t_col, t_end):   #sample frames at each time until you either reach the collision time or reach the end
        #---at the end, the end time will be less than the next collision time

            '''
            for each run on the loop, t_next updates by dt_sample, whilst self.t only updates in blocks of dt_col
            dt is the offset of next sample time from the current sim time
            '''

            dt = t_next - sim.t   #first run, dt=0, then dt=dt_sample, then dt=2dt_sample
            
            #record current sample time
            times.append(t_next)   #add new times to the list

            #evolve gaps to this sample time
            dh = sim.gap_rates()    #array of gap rates
            h_tmp = sim.h + dh * dt   #temporary gap sizes array from time difference
            
            '''
            print(h_tmp[-1])   #diagnostic to check they are the same
            #enforce total free-length constraint using the last gap
            h_tmp[-1] = sim.L_free - np.sum(h_tmp[:-1])
            print(h_tmp[-1])
            '''

            #evolve particle 1
            x1_tmp = (sim.x1 + sim.v[0] * dt) % sim.L

            #reconstruct all particle positions from x1 and the first N-1 gaps
            x_tmp = np.empty(sim.N, dtype=float)
            x_tmp[0] = x1_tmp

            for i in range(1, sim.N):
                centre_to_centre = sim.rod_length + float(h_tmp[i - 1])
                x_tmp[i] = (x_tmp[i - 1] + centre_to_centre) % sim.L

            xs.append(x_tmp)
            


            '''            
            h_tmp[2] = sim.L_free - h_tmp[0] - h_tmp[1]   #enforces constraint of gaps

            x1_tmp = (sim.x1 + sim.v[0] * dt) % sim.L   #updates new mass 1 position

            a = sim.rod_length   #diameter of particle

            '''
            #2 degrees of freedom for centre to centre displacement between particles
            #this is required to calculate positions
            '''
            d12 = a + float(h_tmp[0]) 
            d23 = a + float(h_tmp[1])

            '''
            #updates positions of particles 2 and 3
            '''
            x2_tmp = (x1_tmp + d12) % sim.L 
            x3_tmp = (x2_tmp + d23) % sim.L
                
            xs.append((x1_tmp, x2_tmp, x3_tmp))   #adds tuple of new positions to list of tuples
            '''

            t_next += dt_sample
                
        if t_col >= t_end or not np.isfinite(t_col):
            break

        #diagnostics for centre of momentum frame
        if sim.use_SR:
            P_before = sum(calculate_gamma_factor(sim.v[i], sim.c) * sim.m[i] * sim.v[i] for i in range(sim.N))
            print("before collide:", P_before)  

        sim.advance(dt_col)   #updates the instance of the objects attributes
        sim.collide(k_star) 

        #diagnostics for centre of momentum frame
        if sim.use_SR:
            P_after = sum(calculate_gamma_factor(sim.v[i], sim.c) * sim.m[i] * sim.v[i] for i in range(sim.N))
            print("after collide:", P_after)

    return np.asarray(times, dtype=float), np.asarray(xs, dtype=float)   #converts list of times into a 1D array of floats and list of tuples (x1,x2,x3) into an nx3 array



def animate_particles_on_ring(times, xs, L, interval_ms):
    """
    Animate N particles moving on a ring.

    Parameters
    ----------
    times : array, shape (n_frames,)
        Sample times
    xs : array, shape (n_frames, N)
        Particle positions in [0, L)
    L : float
        Ring length
    interval_ms : int
        Delay between animation frames in milliseconds
    """
    theta = 2.0 * np.pi * (xs / L)

    # convert angles to Cartesian coordinates on a unit circle
    X = np.cos(theta)   # shape (n_frames, N)
    Y = np.sin(theta)   # shape (n_frames, N)

    n_frames, N = xs.shape

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")

    ring = plt.Circle((0, 0), 1.0, fill=False)
    ax.add_patch(ring)

    # one artist per particle
    particles = [ax.plot([], [], marker="o", linestyle="None")[0] for i in range(N)]

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def initialise():
        for p in particles:
            p.set_data([], [])
        time_text.set_text("")
        return tuple(particles) + (time_text,)

    def update(i):
        for j, p in enumerate(particles):
            p.set_data([X[i, j]], [Y[i, j]])
        time_text.set_text(f"t = {times[i]:.3f}")
        return tuple(particles) + (time_text,)

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=initialise, interval=interval_ms, blit=True)

    plt.show()
    return anim