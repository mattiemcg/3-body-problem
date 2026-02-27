import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  



'''
special relativity stuff
'''
def calculate_gamma_factor(v, c):
    '''does what it says on the tin'''

    return 1.0 / np.sqrt(1.0 - (v*v)/(c*c))

def vel_add(u, V, c):
    '''
    relativistic velocity addition: u' = (u - V)/(1 - uV/c^2)
    where V is the speed of the primed frame relative to the unprimed frame
    '''

    return (u - V) / (1.0 - (u*V)/(c*c))

def relativistic_elastic_collision_1d(v_i, v_j, m_i, m_j, c):
    """
    Exact 1D *elastic* collision using 4-momentum conservation.

    Method:
      1) Compute lab-frame total energy E and momentum P.
      2) Boost to COM frame where total momentum is zero.
      3) In 1D elastic scattering, velocities reverse in COM: v' -> -v'.
      4) Boost back to lab frame.
    """

    #---gamma factors for both particles
    g_i = calculate_gamma_factor(v_i, c)
    g_j = calculate_gamma_factor(v_j, c)

    #---lab frame energy and momentum
    E_i = g_i * m_i * c*c
    E_j = g_j * m_j * c*c
    P_i = g_i * m_i * v_i
    P_j = g_j * m_j * v_j

    E_tot = E_i + E_j
    P_tot = P_i + P_j

    #---COM velocity: V = c^2 P / E
    V = (c*c) * P_tot / E_tot

    #---boost to COM frame
    v_i_com = vel_add(v_i, V, c)
    v_j_com = vel_add(v_j, V, c)

    #---elastic collision in COM: reverse velocities
    v_i_com_after = -v_i_com
    v_j_com_after = -v_j_com

    #---boost back to lab frame (inverse boost uses -V)
    v_i_after = vel_add(v_i_com_after, -V, c)
    v_j_after = vel_add(v_j_com_after, -V, c)

    return v_i_after, v_j_after



'''
non relativistic
'''
def elastic_collision_1d(v_i, v_j, m_i, m_j):
    """exact 1D elastic collision update for two masses"""

    v_i_new = ((m_i - m_j) / (m_i + m_j)) * v_i + (2.0 * m_j / (m_i + m_j)) * v_j
    v_j_new = (2.0 * m_i / (m_i + m_j)) * v_i + ((m_j - m_i) / (m_i + m_j)) * v_j
    return v_i_new, v_j_new




class ThreeHardParticlesRing:
    """
    Three hard particles (or rods) on a 1D ring, simulated event-by-event using GAP tracking.

    State:
      - h[0], h[1], h[2]: free gaps between neighbours around the ring (>=0),
        with sum(h) = L_free where:
           L_free = L           (point particles, rod_length=0)
           L_free = L - 3a      (rods of length a)

        Convention:
          h[0] is free gap from particle 1 -> 2
          h[1] is free gap from particle 2 -> 3
          h[2] is free gap from particle 3 -> 1 (wrap-around)

      - v[0], v[1], v[2]: velocities of particles 1,2,3 in an inertial frame
      - x1: an absolute coordinate of particle 1 in [0, L) for animation/visualisation
            (particle 2 and 3 are reconstructed from x1 and gaps)

    Collisions:
      - occur when some h[k] hits 0
      - k=0 -> collision between (1,2)
      - k=1 -> collision between (2,3)
      - k=2 -> collision between (3,1) across the periodic boundary

    This is event-driven (exact collision times for hard collisions).
    """


    def __init__(self, L, m, v, h, rod_length, x1, c, use_SR, K_rel):
        self.L = float(L)
        self.rod_length = float(rod_length)   #for point particles this is zero
        self.t = 0.0    #start time
        self.L_free = self.L - 3.0 * self.rod_length

        #---relativistic dynamics variables
        self.use_SR = bool(use_SR)
        self.c = float(c)
        self.K_rel = float(K_rel)   #this determines how relativistic the system is

        self.m = np.asarray(m, dtype=float).reshape(3,)   #creates 1D 3 entry array of floats for masses
        self.v = np.asarray(v, dtype=float).reshape(3,)   #creates 1D 3 entry array of floats for velocities
        self.h = np.asarray(h, dtype=float).reshape(3,)   #creates 1D 3 entry array of floats for gaps

        self.x1 = float(x1) % self.L   #outputs the remainder, effectively applying periodic boundary conditions



    def COM_frame_non_relativistic(self):
        '''velocities relative to the COM frame'''
        
        M = self.m.sum()   #total mass
        V_cm = float((self.m * self.v).sum() / M)   #velocity of COM frame
        self.v = self.v - V_cm   #shifts every velocity into the COM frame



    def COM_frame_relativistic(self):
        '''velocities relative to the COM frame for relativistic'''

        g = calculate_gamma_factor(self.v, self.c)   #find gamma value for each particle
        P = float(np.sum(g * self.m * self.v))   #calculate momentum for each particle
        E = float(np.sum(g * self.m * self.c**2))   #calculate energy of each particle
        V = (self.c**2) * P/E   #COM frame boost speed
        self.v = vel_add(self.v, V, self.c)   #put velocities in this frame using the boost velocity



    def gap_rates(self):
        """
        calculates rate the gaps close or expand

        dh/dt for each gap:
          dh0/dt = v2 - v1
          dh1/dt = v3 - v2
          dh2/dt = v1 - v3
        """

        v1, v2, v3 = self.v[0], self.v[1], self.v[2]
        return np.array([v2 - v1, v3 - v2, v1 - v3], dtype=float)   #speed of each gap, negative for gap getting smaller, positive for gap getting better



    def next_event(self):
        """
        return (dt_star, k_star) where dt_star is the time until the next collision
        and k_star is the gap index that closes next (0,1,2)
        if no collision is possible (all closing times infinite), returns (np.inf, -1)
        """

        dh = self.gap_rates()   #dh is an array of all the gap rates

        times = np.full(3, np.inf, dtype=float)   #creating a 3 entry 1D array with all elements infinity
        for k in range(3):   #k values 0,1,2
            if dh[k] < 0.0:   #if gap rate is negative, the gap is closing
                times[k] = self.h[k] / (-dh[k])   #update the times array, giving the time until the gap is zero
        #---if gap is not closing the element of the array is left unchanged as infinity---

        k_star = int(np.argmin(times))   #identifies the index of the smallest entry in the times array
        dt_star = float(times[k_star])   #the time left for the next collision

        return dt_star, k_star



    def advance(self, dt):
        """advance gaps and x1 with constant velocities between collisions"""

        dt = float(dt)
        if not np.isfinite(dt):   #if not true - as isfinite is true if dt is finite and not inf or nan
            return   #code exits if the time between the collisions is infinite

        dh = self.gap_rates()   #call method of the class for array of gap rates
        self.h = self.h + dh * dt   #updates all gaps
        self.x1 = (self.x1 + self.v[0] * dt) % self.L   #updating absolute position of mass 1 and then wrapping it into [0,L)
        self.t += dt   #simulation clock is updated by dt



    def collide(self, k):
        """
        resolve the collision corresponding to gap k becoming 0
        k is the index of the gap array
          k=0: particles (1,2)
          k=1: particles (2,3)
          k=2: particles (3,1)
        
        the k value called is the one that is 0
        """

        if k == 0:
            i, j = 0, 1   #gap between masses 1 and 2 is closed
        elif k == 1:
            i, j = 1, 2
        else:
            i, j = 2, 0

        if self.use_SR:
            self.v[i], self.v[j] = relativistic_elastic_collision_1d(self.v[i], self.v[j], self.m[i], self.m[j], self.c)
        else:
            self.v[i], self.v[j] = elastic_collision_1d(self.v[i], self.v[j], self.m[i], self.m[j])



    def sample_for_animation(self, t_end, dt_sample):
        """
        uniform-in-time sampling of n positions for animation
        this outputs all the n times of each sample and all 3 particles positions at each time

        Returns:
        times: (n, 1)
        xs   : (n, 3) positions in [0, L)
        """

        t_end = float(t_end)   #converts input into float, length of animation
        dt_sample = float(dt_sample)   #frame size of animation

        times = []   #creates a list of times
        xs = []   #creates a list

        t_next = self.t   #next time you record the frame


        while t_next < t_end: 
            dt_col, k_star = self.next_event()   #dt_col is the time until the next collision, and not going to use k_star
            t_col = self.t + dt_col if np.isfinite(dt_col) else np.inf   #calculates absolute time of collision


            while t_next < min(t_col, t_end):   #sample frames at each time until you either reach the collision time or reach the end
            #---at the end, the end time will be less than the next collision time

                '''
                for each run on the loop, t_next updates by dt_sample, whilst self.t only updates in blocks of dt_col
                dt is the offset of next sample time from the current sim time
                '''

                dt = t_next - self.t   #first run, dt=0, then dt=dt_sample, then dt=2dt_sample
                t_next += dt_sample
                times.append(t_next)   #add new times to the list

                dh = self.gap_rates()    #array of gap rates
                h_tmp = self.h + dh * dt   #temporary gap sizes array from time difference
                h_tmp[2] = self.L_free - h_tmp[0] - h_tmp[1]   #enforces constraint of gaps

                x1_tmp = (self.x1 + self.v[0] * dt) % self.L   #updates new mass 1 position

                a = self.rod_length   #diameter of particle

                '''
                2 degrees of freedom for centre to centre displacement between particles
                this is required to calculate positions
                '''
                d12 = a + float(h_tmp[0]) 
                d23 = a + float(h_tmp[1])

                '''
                updates positions of particles 2 and 3
                '''
                x2_tmp = (x1_tmp + d12) % self.L 
                x3_tmp = (x2_tmp + d23) % self.L
                
                xs.append((x1_tmp, x2_tmp, x3_tmp))   #adds tuple of new positions to list of tuples

                
            if t_col >= t_end or not np.isfinite(t_col):
                break

            self.advance(dt_col)   #updates the instance of the objects attributes
            k_close = int(np.argmin(self.h))   #identify index of the smallest gap, this is always the closing gap since we have advanced by dt_col
            self.collide(k_close) 

        return np.asarray(times, dtype=float), np.asarray(xs, dtype=float)   #converts list of times into a 1D array of floats and list of tuples (x1,x2,x3) into an nx3 array



    def normalise_COM_energy(self):
        """shift to COM frame (P = 0) and rescale velocities so total kinetic energy = E_target"""

        if not self.use_SR:
            #---Newtonian COM
            self.COM_frame_non_relativistic()   #now the velocities of the instance are in the COM frame
            E_COM = 0.5 * float(np.sum(self.m * self.v**2))
            alpha = np.sqrt(1 / E_COM)
            self.v = self.v * alpha   #now the COM energy is normalised to 1
            return

        
        #---for relativistic energy
        self.COM_frame_relativistic()   #velocities are in relativistic COM frame      
        E_rest = np.sum(self.m) * self.c**2
        self.E_target = np.sum(self.m * self.c**2) + self.K_rel


        def E_com_of(alpha):
            v_scaled = alpha * self.v
            if np.any(np.abs(v_scaled) >= self.c):
                return np.inf
            g = calculate_gamma_factor(v_scaled, self.c)
            return float(np.sum(g * self.m * self.c**2))

        # Bracket alpha in [0, alpha_max)
        alpha_lo = 0.0
        alpha_hi = 0.999999 * float(np.min(self.c / np.maximum(np.abs(self.v), 1e-300)))

        # If all v ~ 0
        if not np.isfinite(alpha_hi) or alpha_hi <= 0:
            if self.E_target > E_rest:
                raise ValueError("All v=0 in COM; cannot reach higher SR energy by scaling.")
            return

        # Bisection solve E_com_of(alpha) = E_target
        lo, hi = alpha_lo, alpha_hi
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            Emid = E_com_of(mid)
            if abs(Emid - self.E_target) < 1e-12 * max(1.0, self.E_target):
                self.v = mid * self.v
                return
            if Emid < self.E_target:
                lo = mid
            else:
                hi = mid

        self.v =  (0.5 * (lo + hi)) * self.v


def animate_three_particles_on_ring(times, xs, L, interval_ms=20):
    
    L = float(L)

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

