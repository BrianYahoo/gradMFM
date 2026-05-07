import brainpy as bp
import brainpy.math as bm

class outBalloon(bp.DynamicalSystemNS):
    def __init__(
        self,
        size: int,
        batch_size: int = 1,
        rou: float = 0.34, # ρ = 0.34 is the resting oxygen extraction fraction, the p_constant
        tau: float = 0.98, # hemodynamic transit time τ = 0.98 s
        kappa: float = 0.65, # The kinetic parameters rate of signal decay κ
        gamma: float = 0.41, # rate of elimination
        alpha: float = 0.32, # Grubb’s exponent
        v_0: float = 0.02,# resting blood volume fraction
        B_0: float = 3.0, # B0, which is 3T in the HCP dataset
        r_0: float = 110, # the intravascular relaxation rate is r0 = 110 Hz
        e: float = 0.47, # the ratio between intravascular and extravascular MR signal is e = 0.47
        TE: float = 0.0331, # The echo time TE = 33.1 ms in the HCP dataset
        ):

        super(outBalloon, self).__init__()
        '''
        Output-nonlinear-scalling-layer (Balloon dynamics) of S from RNN-layer DecoModel, i.e., turning postsynaptic gating variable S to BOLD signal in an element-wise fashion.
        Balloon-Windkessel Hemodynamic model is used, equations with default parameters are from https://www.science.org/doi/10.1126/sciadv.aat7854
        
        size = num, i.e., number of network size (# of node)
        '''

        self.num = size

        self.rou = rou
        self.tau = tau
            
        self.kappa = kappa
        self.gamma = gamma
        self.alpha = alpha
 
        self.v_0 = v_0

        eta_0 = 28.265 * B_0 # ϑ0 = 28.265B0 is the frequency offset at the outer surface of magnetized vessels and depends on the main magnetic field strength B0
        self.k_1 = 4.3 * eta_0 * rou * TE
        self.k_2 = e * r_0 * rou * TE
        self.k_3 = 1-e
            
        self.F_0 = bm.Variable(0.0*bm.ones((batch_size,self.num)), batch_axis = 0)
        self.F_1 = bm.Variable(1.0*bm.ones((batch_size,self.num)), batch_axis = 0) 
        self.F_2 = bm.Variable(1.0*bm.ones((batch_size,self.num)), batch_axis = 0) 
        self.F_3 = bm.Variable(1.0*bm.ones((batch_size,self.num)), batch_axis = 0) 

    def reset_state(self, batch_size=1): # this function defines how to reset the mode states
        self.F_0.value = 0.0*bm.ones((batch_size,self.num)) 
        self.F_1.value = 1.0*bm.ones((batch_size,self.num)) 
        self.F_2.value = 1.0*bm.ones((batch_size,self.num)) 
        self.F_3.value = 1.0*bm.ones((batch_size,self.num)) 

    def update(self,S=0):
        '''
        input   S:  batch_size*size matrix represents synaptic gating variable
                batch_size is the number of batch
                size is the number of nodes

        output  S_BOLD: same shape matrix for BOLD signal output (element-wise Hemodynamic transformation)
        '''
                
        # gating2bold_derivative：
        dF_0 = S - self.kappa * self.F_0 - self.gamma * (self.F_1 -1)
        dF_1 = self.F_0
        dF_2 = 1 / self.tau * (self.F_1 - self.F_2**(1/self.alpha))
        dF_3 = 1 / self.tau * (self.F_1 / self.rou * (1-(1-self.rou)**(1/self.F_1)) - self.F_3 * self.F_2 **(1/self.alpha - 1))

        # forward-eular
        self.F_0.value = self.F_0 + dF_0*bm.dt
        self.F_1.value = self.F_1 + dF_1*bm.dt
        self.F_2.value = self.F_2 + dF_2*bm.dt
        self.F_3.value = self.F_3 + dF_3*bm.dt
        
        # caculate and return BOLD signal
        v_t = self.F_2
        q_t = self.F_3
        return self.v_0 * (self.k_1 * (1 - q_t) + self.k_2 * (1 - q_t / v_t) + self.k_3 * (1 - v_t))
        # return 100/self.rou*self.v_0 * (self.k_1 * (1 - q_t) + self.k_2 * (1 - q_t / v_t) + self.k_3 * (1 - v_t))

class outVolterra(bp.DynamicalSystemNS):
    def __init__(
        self,
        size: int,
        batch_size: int = 1,
        tau_s: float = 2.3,   # synaptic decay time constant
        tau_f: float = 3.9,   # frequency/inertia time constant
        B_0: float = 3.0,     # B0, which is 3T in the HCP dataset
        rou: float = 0.34,    # ρ = 0.34 is the resting oxygen extraction fraction, the p_constant
        TE: float = 0.0331,  # The echo time TE = 33.1 ms in the HCP dataset
        V_0: float = 0.0118,     # resting blood volume fraction
    ):
        super(outVolterra, self).__init__()
        '''
        First-Order Volterra hemodynamic kernel implemented as a 2-state linear ODE:
            tau_f * h'' + tau_s * h' + h = a * S(t)
        turned into a state-space system with states [h, h_dot].

        Args:
            size: Number of nodes (dimensions).
            batch_size: Batch size for simulation.
            tau_s: Damping (synaptic) time constant.
            tau_f: Inertia (frequency) time constant.
            a:     Input scaling (kernel amplitude).
        '''
        self.num = size
        self.tau_s = tau_s
        self.tau_f = tau_f
        self.tau_f_inv = 1 / self.tau_f
        eta_0 = 28.265 * B_0
        self.k_1 = 4.3 * eta_0 * rou * TE
        self.V_0 = V_0
        self.out_scale = self.k_1 * self.V_0

        # State variables: h (hemodynamic response), h_dot (its derivative)
        self.h = bm.Variable(bm.zeros((batch_size, self.num)), batch_axis=0)
        self.h_dot = bm.Variable(bm.ones((batch_size, self.num)), batch_axis=0)

    def reset_state(self, batch_size=1):
        self.h.value = bm.zeros((batch_size, self.num))
        self.h_dot.value = bm.zeros((batch_size, self.num))

    def update(self, S=0):
        '''
        Update the states given input S (synaptic gating variable), using forward Euler.

        Args:
            S: shape (batch_size, size), synaptic input.
        Returns:
            h: shape (batch_size, size), current hemodynamic response.
        '''
        d_h      = self.h_dot
        d_h_dot  = self.tau_f_inv * (S - self.tau_s * self.h_dot - self.h)

        # Forward Euler integration
        self.h.value     = self.h + d_h * bm.dt
        self.h_dot.value = self.h_dot + d_h_dot * bm.dt

        # return self.out_scale * self.h
        return self.h
    
class outGamma(bp.DynamicalSystemNS):
    def __init__(
        self,
        size: int,
        batch_size: int = 1,
        tau: float = 1.08,
        n: float = 3.0,
        factorial: float = 2.0,
        B_0: float = 3.0,     # B0, which is 3T in the HCP dataset
        rou: float = 0.34,    # ρ = 0.34 is the resting oxygen extraction fraction, the p_constant
        TE: float = 0.0331,  # The echo time TE = 33.1 ms in the HCP dataset
        V_0: float = 0.0118,     # resting blood volume fraction
    ):
        super(outGamma, self).__init__()
        '''
        Gamma HRF implemented as a 2-state linear ODE:
            tau * dh/dt = (n - 1 - t / tau) * h(t)
        turned into a state-space system with states [h, h1].
            dh_1/dt = (S(t) - h_1) / tau
            dh/dt = (n - 1) * h_1 / tau - h / tau

        Args:
            size: Number of nodes (dimensions).
            batch_size: Batch size for simulation.
            tau: Damping (synaptic) time constant.
            n: Inertia (frequency) time constant.
            factorial: Input scaling (kernel amplitude).
        '''

        self.num = size
        self.tau = tau
        self.n = n
        self.factorial = factorial
        eta_0 = 28.265 * B_0
        self.k_1 = 4.3 * eta_0 * rou * TE
        self.V_0 = V_0
        self.out_scale = self.k_1 * self.V_0

        # State variables
        self.h1 = bm.Variable(bm.zeros((batch_size, self.num)), batch_axis=0)  # intermediate state
        self.h = bm.Variable(bm.zeros((batch_size, self.num)), batch_axis=0)   # HRF output

    def reset_state(self, batch_size=1):
        self.h1.value = bm.zeros((batch_size, self.num))
        self.h.value = bm.zeros((batch_size, self.num))

    def update(self, S=0):
        d_h1 = (S - self.h1) / self.tau
        d_h = ((self.n - 1) * self.h1 / self.tau - self.h) / self.tau

        self.h1.value = self.h1 + d_h1 * bm.dt
        self.h.value = self.h + d_h * bm.dt

        # return self.out_scale * self.h
        return self.h