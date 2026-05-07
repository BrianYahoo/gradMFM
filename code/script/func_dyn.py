import brainpy as bp
import brainpy.math as bm
from jax import custom_jvp,vmap
from typing import Union,Callable
from brainpy.types import ArrayType

def get_AbbottChance(method = 'bm.ifelse', a=270, b=108, d=0.154, epsilon_inner=1e-1, epsilon_outer=5e2):
    ''' 
    AbbottChance refined using bm.ifelse for numerical stability and auto-grad :
        (-epsilon_inner, +epsilon_inner) section is replaced by a quadratic approximation 
        (-inf, -epsilon_outer) and (+epsilon_outer, +inf) sections are replaced by a ReLU approximation 
    '''
    def AbbottChance_ifelse(x):

        y = bm.ifelse( 
            operands=x, 
            conditions=(
                bm.abs(x)<epsilon_inner,
                bm.abs(x)<epsilon_outer,
                ),
            branches=(
                lambda x: d/12*x**2 + 0.5*x + 1/d, # case1 0点附近二次近似
                lambda x: x/(1 - bm.exp(-d*x)),    # case2 原始AbbottChance
                lambda x: bm.maximum(0,x),         # else  外围ReLU近似
                ),
        )
        
        return y


    ''' 
    AbbottChance refined using jax.custom_jvp with handcrafted gradiant for numerical stability and auto-grad :
        (-epsilon_inner, +epsilon_inner) section is replaced by a quadratic approximation 
    '''
    @custom_jvp
    def AbbottChance_customjvp(x):

        conditions = [
            (bm.abs(x) <= epsilon_inner),# section 1
            ((bm.abs(x) > epsilon_inner)&(bm.abs(x) < epsilon_outer)),# section 2
            (bm.abs(x) >= epsilon_outer),# section 3
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
            (bm.abs(x) <= epsilon_inner),# section 1
            ((bm.abs(x) > epsilon_inner)&(bm.abs(x) < epsilon_outer)),# section 2
            (bm.abs(x) >= epsilon_outer),# section 3
        ]

        x1= bm.where( conditions[0], 
            x, #True
            epsilon_inner, #False,一个对所有情况都不会爆nan得安全的数即可
        )

        x2= bm.where( conditions[1], 
            x, #True
            epsilon_inner, #False
        )

        x3= bm.where( conditions[2], 
            x, #True
            epsilon_inner, #False
        )

        choices = [
            ( d/6*x1 + 0.5 ) * x_dot,
            ( 1 + (1 - d*x2)/(bm.exp(d*x2) - 1) - d*x2/(bm.exp(d*x2) - 1)**2 ) * x_dot,
            ( bm.heaviside(x3,0.5) ) * x_dot,
        ]

        y_dot = bm.select(conditions, choices)

        return y, y_dot
    
    
    # return the AbbottChance we want
    if method == 'bm.ifelse':
        AbbottChance_ifelse_m = vmap(vmap(AbbottChance_ifelse, out_axes=0,in_axes=0), out_axes=0,in_axes=0) # 2-D matrix element-wise operator
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
    reduced-wong-wang-deco model using BrainPy 

    tau_S, w and I should be float or (1,) or (num,) bm.array, if float then it will be initialized as float*(num,) bm.array
    TrainVar_list should be a list such as ['tau_S','G','w','I'] to specify if this parameter is trainable
    
    S_init and H_init should be float or (num,) bm.array, if float then it will be initialized as float*(num,) bm.array
    S(gating state of model) and H(firing rate) will be initialized by S_init and H_init into (batch_size,num) with broadcasting along batch axis  

    """
    def __init__(
        self,
        size: int,
        struc_conn_matrix: ArrayType,
        batch_size: int = 1,
        gamma: float = 0.641, # kinetic parameter
        J: float = 0.2609, # synaptic coupling
        tau_S: Union[float, ArrayType] = 0.1, # time constant
        G: float = 1.0, # global coupling weight
        w: Union[float, ArrayType] = 0.9, # recurrent weights
        I: Union[float, ArrayType] = 0.3, # background inputs (intercepts)
        TrainVar_list = ['G','w','I'],
        H_x_act: Union[str, Callable] = 'Softplus', # callable element-wise activation function or just typing 'Softplus' or 'AbbottChance'
        S_init: Union[float, ArrayType] = None, # initial S
        H_init: Union[float, ArrayType] = None, # initial H (firing rate)
        CST: bool = False,
        rng: bm.random.RandomState = None,
    ):
        

        super(DecoModel, self).__init__()

        #>>> random state
        if rng is None:
            self.rng = bm.random.RandomState(42)
        else:
            self.rng = rng

        #>>> fixed parameters:
        self.num = size # number of network size (# of node)
        # self.struc_conn_matrix = bm.asarray(struc_conn_matrix) # (num,num)
        self.gamma = gamma
        self.J = J

        #>>> Trainable weights
        if 'tau_S' in TrainVar_list:
            if isinstance(tau_S, float):
                self.tau_S = bm.TrainVar(tau_S * bm.ones(self.num)) # (num,) with same initialization
            else:
                self.tau_S = bm.TrainVar(tau_S) # (1,) or (num,)
        else:
            self.tau_S = tau_S # float or (1,) (num,) bm.array

        if 'G' in TrainVar_list:
            self.G = bm.TrainVar(G)
        else:
            self.G = G # float
        
        if 'w' in TrainVar_list:
            if isinstance(w, float):
                self.w = bm.TrainVar(w * bm.ones(self.num)) # (num,) with same initialization
            else:
                self.w = bm.TrainVar(w) # (1,) or (num,)
        else:
            self.w = w # float or (1,) (num,) bm.array

        if 'I' in TrainVar_list:
            if isinstance(I, float):
                self.I = bm.TrainVar(I * bm.ones(self.num)) # (num,) with same initialization
            else:
                self.I = bm.TrainVar(I) # (1,) or (num,)
        else:
            self.I = I # float or (1,) (num,) bm.array

        if 'SC' in TrainVar_list:
            self.struc_conn_matrix = bm.TrainVar(struc_conn_matrix)
        else:
            self.struc_conn_matrix = bm.asarray(struc_conn_matrix)
            
        #>>> activation function
        if callable(H_x_act):
            self.H_x_act = H_x_act
        elif H_x_act == 'Softplus':
            self.H_x_act = lambda x: bm.logaddexp( 0.154*bm.log(2)*(270*x-108), 0 )/( 0.154*bm.log(2) ) # bp.dnn.Softplus(beta=0.154*bm.log(2))(270*x-108) 目前有错，更新后可再使用
        elif H_x_act == 'AbbottChance': # the original AbbottChance
            self.H_x_act = lambda x: bm.nan_to_num( (270*x-108)/(1-bm.exp(-0.154*(270*x-108))) , nan = 1/0.154 )
        elif H_x_act == 'RefinedAbbottChance':
            self.H_x_act = get_AbbottChance(method = 'custom_jvp')

        #>>> sc mapping
        if CST:
            self.sc_mapping = lambda x: bm.relu(x)
        else:
            self.sc_mapping = lambda x: x

        #>>> Variables:
        if S_init is None:
            self.S_init = self.rng.rand(self.num)
        elif isinstance(S_init, float):
            self.S_init = bm.asarray(S_init * bm.ones(self.num))
        else:
            self.S_init = bm.asarray(S_init) # (num,)
        

        if H_init is None:
            self.H_init = bm.zeros(self.num)
        elif isinstance(H_init, float):
            self.H_init = bm.asarray(H_init * bm.ones(self.num))
        else:
            self.H_init = bm.asarray(H_init) # (num,)
        

        self.S = bm.Variable(self.S_init*bm.ones((batch_size,self.num)), batch_axis = 0) 
        self.H = bm.Variable(self.H_init*bm.ones((batch_size,self.num)), batch_axis = 0) 
    
    
    def reset_state(self, batch_size=1): # this function defines how to reset the mode states
        self.S.value = self.S_init*bm.ones((batch_size,self.num))
        self.H.value = self.H_init*bm.ones((batch_size,self.num))
    

    def reset_init(self,):
        self.S_init.value = self.rng.rand(self.num)
        self.H_init.value = self.rng.rand(self.num)


    def update(self, inp = 0):
        # update S based on H and input, noise is integrated into the input
        self.S.value = self.S + ( - self.S / self.tau_S + self.gamma * (1-self.S) * self.H) * bm.dt + inp
        
        # hard threshold of S
        self.S.value = bm.minimum(bm.maximum(self.S, 0), 1)
        
        # get x based on S
        x = self.J * bm.multiply(self.w, self.S) + self.J * self.G * bm.matmul(self.S, self.sc_mapping(self.struc_conn_matrix)) + self.I
        
        # get firing rate H(x) based on its input x
        self.H.value = self.H_x_act(x) 
        
        return self.S