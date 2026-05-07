import brainpy as bp
import brainpy.math as bm
from jax import custom_jvp,vmap
from typing import Union,Callable
from brainpy.types import ArrayType

# Neural-mass dynamics for the Deco reduced Wong-Wang model.
# This module keeps the activation and state-update code differentiable in JAX.

def get_AbbottChance(method = 'bm.ifelse', a=270, b=108, d=0.154, epsilon_inner=1e-1, epsilon_outer=5e2):
    ''' 
    Abbott-Chance transfer function refined with bm.ifelse for numerical
    stability and automatic differentiation:
        (-epsilon_inner, +epsilon_inner) is replaced by a quadratic approximation
        (-inf, -epsilon_outer) and (+epsilon_outer, +inf) are replaced by a ReLU approximation
    '''
    def AbbottChance_ifelse(x):

        y = bm.ifelse( 
            operands=x, 
            conditions=(
                bm.abs(x)<epsilon_inner,
                bm.abs(x)<epsilon_outer,
                ),
            branches=(
                lambda x: d/12*x**2 + 0.5*x + 1/d, # Quadratic approximation near zero.
                lambda x: x/(1 - bm.exp(-d*x)),    # Original Abbott-Chance transfer function.
                lambda x: bm.maximum(0,x),         # ReLU approximation in the numerical tail.
                ),
        )
        
        return y


    ''' 
    Abbott-Chance transfer function refined with jax.custom_jvp and a
    hand-crafted gradient for numerical stability and automatic differentiation:
        (-epsilon_inner, +epsilon_inner) is replaced by a quadratic approximation
    '''
    @custom_jvp
    def AbbottChance_customjvp(x):

        conditions = [
            (bm.abs(x) <= epsilon_inner),# Near-zero approximation region.
            ((bm.abs(x) > epsilon_inner)&(bm.abs(x) < epsilon_outer)),# Original-function region.
            (bm.abs(x) >= epsilon_outer),# Tail approximation region.
        ]

        choices = [
            (d/12*x**2 + 0.5*x + 1/d),
            (x/(1 - bm.exp(-d*x))),
            (bm.maximum(0,x))
        ]

        return bm.select(conditions, choices)


    @AbbottChance_customjvp.defjvp
    def AbbottChance_customjvp_jvp(primals, tangents):
        x, = primals
        x_dot, = tangents
        y = AbbottChance_customjvp(x)

        conditions = [
            (bm.abs(x) <= epsilon_inner),# Near-zero approximation region.
            ((bm.abs(x) > epsilon_inner)&(bm.abs(x) < epsilon_outer)),# Original-function region.
            (bm.abs(x) >= epsilon_outer),# Tail approximation region.
        ]

        x1= bm.where( conditions[0], 
            x, # Active value in this region.
            epsilon_inner, # Safe fallback used only outside the active region.
        )

        x2= bm.where( conditions[1], 
            x, # Active value in this region.
            epsilon_inner, # Safe fallback used only outside the active region.
        )

        x3= bm.where( conditions[2], 
            x, # Active value in this region.
            epsilon_inner, # Safe fallback used only outside the active region.
        )

        choices = [
            ( d/6*x1 + 0.5 ) * x_dot,
            ( 1 + (1 - d*x2)/(bm.exp(d*x2) - 1) - d*x2/(bm.exp(d*x2) - 1)**2 ) * x_dot,
            ( bm.heaviside(x3,0.5) ) * x_dot,
        ]

        y_dot = bm.select(conditions, choices)

        return y, y_dot
    
    
    # Return the requested differentiable Abbott-Chance implementation.
    if method == 'bm.ifelse':
        AbbottChance_ifelse_m = vmap(vmap(AbbottChance_ifelse, out_axes=0,in_axes=0), out_axes=0,in_axes=0) # Element-wise operator for 2-D arrays.
        def AbbottChance(x):
            return AbbottChance_ifelse_m(a*x-b)  
         
    elif method == 'custom_jvp':
        def AbbottChance(x):
            return AbbottChance_customjvp(a*x-b)
        
    else:
        raise ValueError("method from ['bm.ifelse','custom_jvp']")

    return AbbottChance

class DecoModel(bp.DynamicalSystemNS):
    """
    Reduced Wong-Wang-Deco model implemented with BrainPy.

    tau_S, w, and I may be floats, scalar-like arrays, or node-wise arrays. Float
    values are expanded to homogeneous node-wise arrays when registered as
    trainable variables.

    TrainVar_list specifies which parameters are optimized, for example
    ['tau_S', 'G', 'w', 'I'].
    
    S_init and H_init may be floats or node-wise arrays. S (synaptic gating
    state) and H (firing rate) are broadcast to (batch_size, num) during state
    initialization.

    """
    def __init__(
        self,
        size: int,
        struc_conn_matrix: ArrayType,
        batch_size: int = 1,
        gamma: float = 0.641, # Kinetic parameter.
        J: float = 0.2609, # Synaptic coupling.
        tau_S: Union[float, ArrayType] = 0.1, # Synaptic gating time constant.
        G: float = 1.0, # Global coupling weight.
        w: Union[float, ArrayType] = 0.9, # Regional recurrent weights.
        I: Union[float, ArrayType] = 0.3, # Regional background inputs.
        TrainVar_list = ['G','w','I'],
        H_x_act: Union[str, Callable] = 'Softplus', # Element-wise firing-rate transfer function.
        S_init: Union[float, ArrayType] = None, # Initial synaptic gating state.
        H_init: Union[float, ArrayType] = None, # Initial firing rate.
        CST: bool = False,
        rng: bm.random.RandomState = None,
    ):
        

        super(DecoModel, self).__init__()

        # Random state controls initial conditions and stochastic inputs.
        if rng is None:
            self.rng = bm.random.RandomState(42)
        else:
            self.rng = rng

        # Fixed model parameters.
        self.num = size # Number of network nodes.
        self.gamma = gamma
        self.J = J

        # Register selected parameters as BrainPy train variables.
        if 'tau_S' in TrainVar_list:
            if isinstance(tau_S, float):
                self.tau_S = bm.TrainVar(tau_S * bm.ones(self.num)) # Homogeneous node-wise initialization.
            else:
                self.tau_S = bm.TrainVar(tau_S) # Scalar-like or node-wise array.
        else:
            self.tau_S = tau_S # Fixed scalar-like or node-wise value.

        if 'G' in TrainVar_list:
            self.G = bm.TrainVar(G)
        else:
            self.G = G # Fixed global coupling.
        
        if 'w' in TrainVar_list:
            if isinstance(w, float):
                self.w = bm.TrainVar(w * bm.ones(self.num)) # Homogeneous node-wise initialization.
            else:
                self.w = bm.TrainVar(w) # Scalar-like or node-wise array.
        else:
            self.w = w # Fixed scalar-like or node-wise value.

        if 'I' in TrainVar_list:
            if isinstance(I, float):
                self.I = bm.TrainVar(I * bm.ones(self.num)) # Homogeneous node-wise initialization.
            else:
                self.I = bm.TrainVar(I) # Scalar-like or node-wise array.
        else:
            self.I = I # Fixed scalar-like or node-wise value.

        if 'SC' in TrainVar_list:
            self.struc_conn_matrix = bm.TrainVar(struc_conn_matrix)
        else:
            self.struc_conn_matrix = bm.asarray(struc_conn_matrix)
            
        # Select the input-output transfer function for firing rates.
        if callable(H_x_act):
            self.H_x_act = H_x_act
        elif H_x_act == 'Softplus':
            self.H_x_act = lambda x: bm.logaddexp( 0.154*bm.log(2)*(270*x-108), 0 )/( 0.154*bm.log(2) ) # Numerically stable Softplus form of the Abbott-Chance gain.
        elif H_x_act == 'AbbottChance': # Original Abbott-Chance transfer function.
            self.H_x_act = lambda x: bm.nan_to_num( (270*x-108)/(1-bm.exp(-0.154*(270*x-108))) , nan = 1/0.154 )
        elif H_x_act == 'RefinedAbbottChance':
            self.H_x_act = get_AbbottChance(method = 'custom_jvp')

        # Optionally constrain the effective structural connectivity to be non-negative.
        if CST:
            self.sc_mapping = lambda x: bm.relu(x)
        else:
            self.sc_mapping = lambda x: x

        # Initialize gating-state and firing-rate variables for batched simulation.
        if S_init is None:
            self.S_init = self.rng.rand(self.num)
        elif isinstance(S_init, float):
            self.S_init = bm.asarray(S_init * bm.ones(self.num))
        else:
            self.S_init = bm.asarray(S_init) # Node-wise initial gating state.
        

        if H_init is None:
            self.H_init = bm.zeros(self.num)
        elif isinstance(H_init, float):
            self.H_init = bm.asarray(H_init * bm.ones(self.num))
        else:
            self.H_init = bm.asarray(H_init) # Node-wise initial firing rate.
        

        self.S = bm.Variable(self.S_init*bm.ones((batch_size,self.num)), batch_axis = 0) 
        self.H = bm.Variable(self.H_init*bm.ones((batch_size,self.num)), batch_axis = 0) 
    
    
    def reset_state(self, batch_size=1): # Reset dynamic states for a new batch.
        self.S.value = self.S_init*bm.ones((batch_size,self.num))
        self.H.value = self.H_init*bm.ones((batch_size,self.num))
    

    def reset_init(self,):
        # Refresh initial conditions before warm-up to reduce dependence on a single state.
        self.S_init.value = self.rng.rand(self.num)
        self.H_init.value = self.rng.rand(self.num)


    def update(self, inp = 0):
        # Integrate synaptic gating; stochastic drive is passed through inp.
        self.S.value = self.S + ( - self.S / self.tau_S + self.gamma * (1-self.S) * self.H) * bm.dt + inp
        
        # Keep gating variables within the biophysical interval [0, 1].
        self.S.value = bm.minimum(bm.maximum(self.S, 0), 1)
        
        # Compute recurrent, long-range, and background input currents.
        x = self.J * bm.multiply(self.w, self.S) + self.J * self.G * bm.matmul(self.S, self.sc_mapping(self.struc_conn_matrix)) + self.I
        
        # Map input currents to firing rates.
        self.H.value = self.H_x_act(x) 
        
        return self.S
