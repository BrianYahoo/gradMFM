import brainpy as bp
import brainpy.math as bm

from func_metrics import compute_FC, compute_sliced_trFCD, correlation_similarity_batch

class loss_function_fc(bp.BrainPyObject):
    def __init__(self, model, N, sigma, TrainVar_list, ZSC=False):

        super(loss_function_fc, self).__init__()

        self.model = model
        self.ZSC = ZSC

        if 'sigma' in TrainVar_list:
            if isinstance(sigma, float):
                self.sigma = bm.TrainVar(sigma*bm.ones(N))  
            
            else:
                self.sigma = bm.TrainVar(sigma)
        else:
            if isinstance(sigma, float):
                self.sigma = bm.asarray(sigma*bm.ones(N))  
            
            else:
                self.sigma = bm.asarray(sigma)

    def __call__(self, inputs, tr_tuple):
        
        FC_target, mask = tr_tuple

        inputs = inputs*bm.abs(self.sigma)
        runner = bp.DSTrainer(self.model, progress_bar=False, numpy_mon_after_run=False)
        output = runner.predict(inputs, reset_state=False)

        FC_predict = compute_FC(output, self.ZSC) * mask

        mse = bp.losses.mean_squared_error(FC_predict , FC_target)
        cor = bm.mean(correlation_similarity_batch(FC_predict, FC_target))

        loss = 1 - cor + mse

        return  loss, (mse, cor)

class loss_function_fcd(bp.BrainPyObject):
    def __init__(self, model, N, sigma, TrainVar_list=[], ZSC=False):

        super(loss_function_fcd, self).__init__()

        self.model = model
        self.ZSC = ZSC

        if 'sigma' in TrainVar_list:
            if isinstance(sigma, float):
                self.sigma = bm.TrainVar(sigma*bm.ones(N))
            else:
                self.sigma = bm.TrainVar(sigma)
        else:
            if isinstance(sigma, float):
                self.sigma = bm.asarray(sigma*bm.ones(N))  
            else:
                self.sigma = bm.asarray(sigma)

    def __call__(self, inputs, tr_tuple):

        FC_target, mask, trFCD_target_mean, trFCD_target_std, slices = tr_tuple

        inputs = inputs*bm.abs(self.sigma)
        runner = bp.DSTrainer(self.model, progress_bar=False, numpy_mon_after_run=False)
        output = runner.predict(inputs, reset_state=False)

        FC_predict = compute_FC(output, self.ZSC) * mask

        mse = bp.losses.mean_squared_error(FC_target, FC_predict)
        cor = bm.mean(correlation_similarity_batch(FC_target, FC_predict))

        trFCD_predict = compute_sliced_trFCD(output, slices, self.ZSC)

        loss_mean = bm.abs(bm.mean(trFCD_predict) - trFCD_target_mean)
        loss_std = bm.abs(bm.std(trFCD_predict) - trFCD_target_std)

        loss = 1-cor + mse + loss_mean + loss_std

        return  loss, (mse, cor, loss_mean, loss_std)