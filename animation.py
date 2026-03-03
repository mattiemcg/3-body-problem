import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  

from three_body_collision import ThreeHardParticlesRing   



def sample_for_animation(sim, t_end, dt_sample):
    """
    uniform-in-time sampling of n positions for animation        
    this outputs all the n times of each sample and all 3 particles positions at each time

    n positions are gathered for all 3 particles uniformly between t_initial = 0 and t_end
    n = t_end / dt_sample

    Returns:
    times: (n, 1)
    xs   : (n, 3) positions in [0, L)
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
            t_next += dt_sample
            times.append(t_next)   #add new times to the list

            dh = sim.gap_rates()    #array of gap rates
            h_tmp = sim.h + dh * dt   #temporary gap sizes array from time difference
            h_tmp[2] = sim.L_free - h_tmp[0] - h_tmp[1]   #enforces constraint of gaps

            x1_tmp = (sim.x1 + sim.v[0] * dt) % sim.L   #updates new mass 1 position

            a = sim.rod_length   #diameter of particle

            '''
            2 degrees of freedom for centre to centre displacement between particles
            this is required to calculate positions
            '''
            d12 = a + float(h_tmp[0]) 
            d23 = a + float(h_tmp[1])

            '''
            updates positions of particles 2 and 3
            '''
            x2_tmp = (x1_tmp + d12) % sim.L 
            x3_tmp = (x2_tmp + d23) % sim.L
                
            xs.append((x1_tmp, x2_tmp, x3_tmp))   #adds tuple of new positions to list of tuples

                
        if t_col >= t_end or not np.isfinite(t_col):
            break

        sim.advance(dt_col)   #updates the instance of the objects attributes
        sim.collide(k_star) 

    return np.asarray(times, dtype=float), np.asarray(xs, dtype=float)   #converts list of times into a 1D array of floats and list of tuples (x1,x2,x3) into an nx3 array



def animate_three_particles_on_ring(times, xs, L, interval_ms):

    theta = 2.0 * np.pi * (xs / L)   #maps each position into an angle
      
    #---converts angles into cartesian coordinates
    X = np.cos(theta)   #outputs array nx3
    Y = np.sin(theta)   #outputs array nx3
    
    '''
    the frame is the row and the column indicates which particle
    X[i, j], Y[i, j] give the x,y position of particle j at frame i.
    '''

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")

    
    #---need to change this to work for any size of circle
    ax.set_xlim(-1.2, 1.2) 
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")

    
    #---size of circle needs to have input
    ring = plt.Circle((0, 0), 1.0, fill=False)
    ax.add_patch(ring)

    p1, = ax.plot([], [], marker="o", linestyle="None")
    p2, = ax.plot([], [], marker="o", linestyle="None")
    p3, = ax.plot([], [], marker="o", linestyle="None")
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)


    def initialise():
        p1.set_data([], [])
        p2.set_data([], [])
        p3.set_data([], [])
        time_text.set_text("")
        return p1, p2, p3, time_text


    def update(i):
        p1.set_data([X[i, 0]], [Y[i, 0]])
        p2.set_data([X[i, 1]], [Y[i, 1]])
        p3.set_data([X[i, 2]], [Y[i, 2]])
        time_text.set_text(f"t = {times[i]:.3f}")
        return p1, p2, p3, time_text


    anim = FuncAnimation(fig, update, frames=len(times), init_func=initialise, interval=interval_ms, blit=True)
    plt.show()
    return anim