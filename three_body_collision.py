import numpy as np



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
      1) Compute lab-frame (global centre momentum frame) total energy E and momentum P.
      2) Boost to COM frame of the two particles where total momentum is zero.
      3) In 1D elastic scattering, velocities reverse in COM frame: v' -> -v'.
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

    #---boost to COM frame of two particles
    v_i_com_before = vel_add(v_i, V, c)
    v_j_com_before = vel_add(v_j, V, c)

    #---elastic collision in COM: reverse velocities
    v_i_com_after = -v_i_com_before
    v_j_com_after = -v_j_com_before

    #---boost back to global COM frame (inverse boost uses -V)
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


    def __init__(self, L, m, v, h, rod_length, x1, use_SR, K_rel):
        self.L = float(L)
        self.rod_length = float(rod_length)   #for point particles this is zero
        self.t = 0.0    #start time
        self.L_free = self.L - 3.0 * self.rod_length

        #---relativistic dynamics variables
        self.use_SR = bool(use_SR)
        self.c = 1.0
        self.K_rel = float(K_rel)   #this determines how relativistic the system is

        self.m = np.asarray(m, dtype=float).reshape(3,)   #creates 1D 3 entry array of floats for masses
        self.v = np.asarray(v, dtype=float).reshape(3,)   #creates 1D 3 entry array of floats for velocities
        self.h = np.asarray(h, dtype=float).reshape(3,)   #creates 1D 3 entry array of floats for gaps

        self.x1 = float(x1) % self.L   #outputs the remainder, effectively applying periodic boundary conditions



    def COM_frame_non_relativistic(self):
        '''
        velocities relative to the COM frame
        in Newtonian COM frame is where the total momentum is zero
        '''
        
        M = self.m.sum()   #total mass
        V_cm = float((self.m * self.v).sum() / M)   #velocity of COM frame
        self.v = self.v - V_cm   #shifts every velocity into the COM frame



    def centre_momentum_frame_relativistic(self):
        '''
        velocities relative to the COM frame for relativistic
        where sum of relativistic momentum = 0
        '''

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
        dt_col = float(times[k_star])   #the time left for the next collision

        return dt_col, k_star



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
        gets called at the end of sample_for_animation when the gap is zero
        it then calls one of the post collision velocity calc

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



    def normalise_COM_energy(self):
        """
        shift to frame where (P = 0)
        then rescale velocities so total kinetic energy = E_target
        E_target is always 1 for Newtonian since different energies don't change the dynamics
        """

        if not self.use_SR:
            #---Newtonian COM
            self.COM_frame_non_relativistic()   #now the velocities of the instance are in the COM frame
            E_COM = 0.5 * float(np.sum(self.m * self.v**2))
            alpha = np.sqrt(1 / E_COM)
            self.v = self.v * alpha   #now the COM energy is normalised to 1
            print(0.5 * float(np.sum(self.m * self.v**2)))
            return

        
        #---for relativistic energy
        self.centre_momentum_frame_relativistic()   #velocities are in frame where the total momentum is zero      

        E_rest = float(np.sum(self.m) * self.c**2)   #1D array of the energies of each particle in its own rest frame
        self.E_target = E_rest + float(self.K_rel)   #what value of energy we are normalising to

        g_before = calculate_gamma_factor(self.v, self.c)   #1D array of gamma factors for each particle in the centre of momentum frame
        p_before = g_before * self.m * self.v   #1D array of momentums of each particle in the centre of momentum frame
        total_momentum_before = np.sum((p_before))
        print("Total relativistic momentum before rescaling for energy =", total_momentum_before)

        #---if there is no relative motion between the particles
        if np.all(np.abs(p_before) < 1e-14):
            if self.E_target > E_rest + 1e-14:
                raise ValueError("All internal momenta are zero in COM; cannot reach higher SR energy.")
            return


        def total_energy_of(lam):
            '''E_i = sqrt(m_i^2 c^4 + p_i^2 c^2)'''

            p_after = lam * p_before
            return float(np.sum(np.sqrt((self.m**2) * self.c**4 + (p_after**2) * self.c**2)))


        #E_target must be greater than rest energy
        if self.E_target < E_rest:
            raise ValueError("Target energy is below total rest energy.")

        # Bracket lambda
        lam_lo = 0.0
        lam_hi = 1.0

        while total_energy_of(lam_hi) < self.E_target:
            lam_hi *= 2.0   #find the upper boundary for the scaling factor
            if lam_hi > 1e12:   #if lambda is getting too high
                raise RuntimeError("Failed to bracket relativistic momentum scale.")

        # Bisection: energy only approximate is fine
        for _ in range(100):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            E_mid = total_energy_of(lam_mid)
            if E_mid < self.E_target:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid

        lamda = 0.5 * (lam_lo + lam_hi)

        p_after = lamda * p_before   #scale momenta exactly

        E_i = np.sqrt((self.m**2) * self.c**4 + (p_after**2) * self.c**2)
        self.v = p_after * self.c**2 / E_i

        # Diagnostics
        g_after = calculate_gamma_factor(self.v, self.c)
        total_momentum_after = float(np.sum(g_after * self.m * self.v))
        total_energy = float(np.sum(g_after * self.m * self.c**2))

        print("Total relativistic momentum after rescaling =", total_momentum_after)
        print("Total relativistic energy after rescaling   =", total_energy)
        print("Energy error =", total_energy - self.E_target)