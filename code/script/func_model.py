import brainpy as bp
import brainpy.math as bm
from func_dyn import DecoModel
from func_out import outBalloon, outVolterra

class MFM(bp.DynamicalSystemNS):
    def __init__(self, N, Batch_size, struc_conn_matrix, G, w, I, TrainVar_list, CST=True, rng=None):
        super(MFM, self).__init__()
        self.rnnLayer = DecoModel(size=N, struc_conn_matrix=bm.asarray(struc_conn_matrix), 
                                batch_size=Batch_size,
                                G=G, w=w, I=I, TrainVar_list=TrainVar_list,
                                H_x_act = 'Softplus', 
                                CST=CST, rng=rng)

    def reset_state(self, Batch_size): # this function defines how to reset the mode states
        self.rnnLayer.reset_state(Batch_size)
        
    def reset_init(self,):
        self.rnnLayer.reset_init()

    def update(self, x = 0):
        return self.rnnLayer(x)

class MFMVolterra(bp.DynamicalSystemNS):
    def __init__(self, N, Batch_size, struc_conn_matrix, G, w, I, TrainVar_list, CST=True, rng=None):
        super(MFMVolterra, self).__init__()
                                         
        self.rnnLayer = DecoModel(size=N, struc_conn_matrix=struc_conn_matrix, 
                                  batch_size=Batch_size,
                                  G=G, w=w, I=I, TrainVar_list=TrainVar_list,
                                  H_x_act = 'Softplus', 
                                  CST=CST, rng=rng)
        
        self.outLayer = outVolterra(size=N,batch_size=Batch_size)

    def reset_state(self, Batch_size): # this function defines how to reset the mode states
        self.rnnLayer.reset_state(Batch_size)
        self.outLayer.reset_state(Batch_size)

    def reset_init(self,):
        self.rnnLayer.reset_init()

    def update(self, x = 0):
        return self.outLayer(self.rnnLayer(x))
    
class MFMBalloon(bp.DynamicalSystemNS):
    def __init__(self, N, Batch_size, struc_conn_matrix, G, w, I, TrainVar_list, CST=True, rng=None):
        super(MFMBalloon, self).__init__()
                                         
        self.rnnLayer = DecoModel(size=N, struc_conn_matrix=struc_conn_matrix, 
                                  batch_size=Batch_size,
                                  G=G, w=w, I=I, TrainVar_list=TrainVar_list,
                                  H_x_act = 'AbbottChance', 
                                  CST=CST, rng=rng)
        
        self.outLayer = outBalloon(size=N,batch_size=Batch_size,)

    def reset_state(self, Batch_size): # this function defines how to reset the mode states
        self.rnnLayer.reset_state(Batch_size)
        self.outLayer.reset_state(Batch_size)
        
    def reset_init(self,):
        self.rnnLayer.reset_init()

    def update(self, x = 0):
        return self.outLayer(self.rnnLayer(x))