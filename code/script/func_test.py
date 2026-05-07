import os
import sys
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import brainpy as bp
import brainpy.math as bm

from func_settings import get_save_path, get_fig_dir, load_data
from func_metrics import z_score, compute_fc_np, compute_fcd_np
from func_model import MFM, MFMVolterra, MFMBalloon

from tqdm import tqdm

def test(step, settings_list):
    test = Test(step, settings_list)
    test.run()

class Test:
    def __init__(self, step, settings_list):
        bm.set_mode(bm.batching_mode)
        self.step = step
        self.settings_list = settings_list
        self.settings_dict = settings_list[step]
        self.dt = self.settings_dict['dt']
        self.TR = self.settings_dict['TR']
        self.rand_seed = self.settings_dict['random seed']
        self.batch_size = self.settings_dict['batch size']
        self.window = int(self.settings_dict['window time']/self.TR)
        self.warmup_time_seconds = self.settings_dict['warm-up epoch long']
        self.run_time_seconds = self.settings_dict['fMRI time']
        self.downsample_rate = int(np.round(self.TR/self.dt, 0))
        self.warmation = int(np.round(self.warmup_time_seconds/self.dt, 0))
        self.duration = int(np.round(self.run_time_seconds/self.dt, 0))
        self.model_dict = {'Activity': MFM, 'Volterra': MFMVolterra, 'Balloon': MFMBalloon}
        self.train_var_list = self.settings_dict['training variables']
        self.CST = self.settings_dict['contrain non-negative SC']
        self.PROG = False

    def progress_bar(self, list_input, desc=None):
        if self.PROG:
            return tqdm(list_input, desc=desc)
        else:
            print(desc+'...')
            return list_input

    def set_seed(self):
        bm.random.seed(int(self.rand_seed + self.step*1000))
        np.random.seed(int(self.rand_seed + self.step*1000))
        self.rng = bm.random.RandomState(int(self.rand_seed + self.step*1000))

    def set_save_dir(self):
        self.save_dir0, self.save_dir1 = get_save_path(self.settings_dict)
        self.save_dir = os.path.join(self.save_dir0, self.save_dir1)
        os.makedirs(self.save_dir, exist_ok=True)
        if os.path.exists(os.path.join(self.save_dir, 'signals_run_downsampled.npz')):
            print('Test of seed {} has already been completed!'.format(self.rand_seed))
            sys.exit()
        
    def set_fig_dir(self):
        self.fig_dir = get_fig_dir(self.settings_dict)
        self.fig_dir_dict = {'average': os.path.join(self.fig_dir, 'average')}
        for batch_idx in range(self.batch_size):
            self.fig_dir_dict[batch_idx] = os.path.join(self.fig_dir, 'run{}'.format(batch_idx+1))
        for key, dir_tmp in self.fig_dir_dict.items():
            os.makedirs(dir_tmp, exist_ok=True)

    def load_vali_states(self):
        last_step = self.settings_dict['training steps'][-1]
        load_dir, load_file = get_save_path(settings_dict=self.settings_list[last_step])
        load_path_vali = os.path.join(load_dir, load_file+'_vali.bp')
        if not os.path.exists(load_path_vali):
            print('Validation of seed {} has not been completed!'.format(self.rand_seed))
            sys.exit()
            
        self.states = bp.checkpoints.load_pytree(load_path_vali)
        fc_corr = np.array(self.states['epoch_FCcorr_vali'])
        fcd_ks = np.array(self.states['epoch_FCDks_vali'])
        loss = 1 - fc_corr + fcd_ks
        self.best_epoch = np.argmin(loss)
        print('Best epoch:', self.best_epoch)
        print('Best loss:', loss[self.best_epoch])
        print('Best FC correlation:', fc_corr[self.best_epoch])
        print('Best FCD KS distance:', fcd_ks[self.best_epoch])

    def get_data(self):
        self.n_roi, _, self.fc_emp, self.biomarkers, _, _, _, _ = load_data(self.settings_dict)
        np.fill_diagonal(self.fc_emp, 1.)
        ###########################################################################################################
        self.fcd_cdf_emp = self.biomarkers['fcd cdf']
        self.fcd_cdf_vmin = 0.
        self.fcd_cdf_vmax = 1.
        self.fcd_cdf_n_bins = 10000

        epoch_G = np.array(self.states['epoch_G'])
        epoch_w = np.array(self.states['epoch_w'])
        epoch_I = np.array(self.states['epoch_I'])
        epoch_sigma = np.array(self.states['epoch_sigma'])
        epoch_SC = np.array(self.states['epoch_SC'])
        self.parameter_dict = {'G': epoch_G, 'w': epoch_w, 'I': epoch_I, 'sigma': epoch_sigma, 'SC': epoch_SC}

        self.G_np = epoch_G[self.best_epoch].repeat(self.batch_size, axis=0).reshape(self.batch_size, 1)
        self.w_np = np.expand_dims(epoch_w[self.best_epoch], axis=0).repeat(self.batch_size, axis=0)
        self.I_np = np.expand_dims(epoch_I[self.best_epoch], axis=0).repeat(self.batch_size, axis=0)
        self.sigma_np = np.expand_dims(epoch_sigma[self.best_epoch], axis=(0, 1)).repeat(self.batch_size, axis=0)
        self.struc_conn_matrix_np = np.array(bm.relu(epoch_SC[self.best_epoch]))
    
    def generate_noise(self):
        print('Generate noise...')
        tic = time.time()
        self.noise_wm_np = self.rng.randn(self.batch_size, self.warmation, self.n_roi)*np.sqrt(self.dt)*np.abs(self.sigma_np)
        self.noise_run_np = self.rng.randn(self.batch_size, self.duration, self.n_roi)*np.sqrt(self.dt)*np.abs(self.sigma_np)
        toc = time.time()
        print('Generation time: {:.2f}s'.format(toc-tic))

    def simulate_signal_gpu(self, model):
        ###############################################################################################
        print('Move data to GPU...')
        bm.dt = self.dt
        G = bm.asarray(self.G_np).cuda()
        w = bm.asarray(self.w_np).cuda()
        I = bm.asarray(self.I_np).cuda()
        struc_conn_matrix = bm.asarray(self.struc_conn_matrix_np).cuda()
        noise_wm = bm.asarray(self.noise_wm_np).cuda()
        noise_run = bm.asarray(self.noise_run_np).cuda()
        ###############################################################################################
        model_run = model(self.n_roi, self.batch_size, struc_conn_matrix, G, w, I, 
                          TrainVar_list=self.train_var_list, CST=self.CST, rng=self.rng)
        print('Warmup...')
        runner_warmup = bp.DSTrainer(model_run, progress_bar=self.PROG, numpy_mon_after_run=False)
        signal_warmup = np.array(runner_warmup.predict(noise_wm, reset_state=True))
        noise_wm = None
        print('Run...')
        runner_run = bp.DSTrainer(model_run, progress_bar=self.PROG, numpy_mon_after_run=False)
        signal_run = np.array(runner_run.predict(noise_run, reset_state=False))
        ###############################################################################################
        bm.clear_buffer_memory()
        return signal_warmup, signal_run
    
    def simulate_signal_cpu(self, model):
        bm.dt = self.dt
        G = bm.asarray(self.G_np)
        w = bm.asarray(self.w_np)
        I = bm.asarray(self.I_np)
        struc_conn_matrix = bm.asarray(self.struc_conn_matrix_np)
        noise_wm = bm.asarray(self.noise_wm_np)
        noise_run = bm.asarray(self.noise_run_np)
        ###############################################################################################
        model_run = model(self.n_roi, self.batch_size, struc_conn_matrix, G, w, I,
                          TrainVar_list=self.train_var_list, CST=self.CST, rng=self.rng)
        print('Warmup...')
        runner_warmup = bp.DSTrainer(model_run, progress_bar=self.PROG, numpy_mon_after_run=False)
        signal_warmup = np.array(runner_warmup.predict(noise_wm, reset_state=True))
        print('Run...')
        runner_run = bp.DSTrainer(model_run, progress_bar=self.PROG, numpy_mon_after_run=False)
        signal_run = np.array(runner_run.predict(noise_run, reset_state=False))
        ###############################################################################################
        return signal_warmup, signal_run
    
    def simulate_signal_all(self):
        self.signal_warmup_dict = {}
        self.signal_warmup_downsampled_dict = {}
        self.signal_run_dict = {}
        self.signal_run_downsampled_dict = {}
        for model_name, model_class in self.model_dict.items():
            print('#############Simulating {} for seed={}#############'.format(model_name, self.rand_seed))
            tic = time.time()
            signal_warmup, signal_run = self.simulate_signal_cpu(model_class)
            self.signal_warmup_dict[model_name] = signal_warmup
            self.signal_warmup_downsampled_dict[model_name] = signal_warmup[:, ::self.downsample_rate, :]
            self.signal_run_dict[model_name] = signal_run
            self.signal_run_downsampled_dict[model_name] = signal_run[:, ::self.downsample_rate, :]
            toc = time.time()
            print('Simulation time: {:.2f}s'.format(toc-tic))

    def compute_metrics(self):
        self.fc_sim_dict = {}
        self.fc_ave_dict = {'Empirical': self.fc_emp}
        self.fcd_sim_dict = {}
        self.fcd_cdf_dict = {'Empirical': self.fcd_cdf_emp}
        self.fcd_pdf_dict = {'Empirical': np.diff(self.fcd_cdf_emp) / ((self.fcd_cdf_vmax-self.fcd_cdf_vmin)/self.fcd_cdf_n_bins)}
        for key, signal in self.signal_run_downsampled_dict.items():
            fc_list = []
            fcd_list = []
            fcd_entries = []
            for batch_idx in self.progress_bar(range(signal.shape[0]), desc=f'Computing FC and FCD for {key}'):
                signal_tmp = signal[batch_idx, :, :].T
                fc = compute_fc_np(signal_tmp)
                fc_list.append(fc)
                fcd = compute_fcd_np(signal_tmp, window=int(self.window))
                fcd_list.append(fcd)
                fcd_entries.extend(fcd[np.triu_indices_from(fcd, k=1)])
            ###########################################################################################
            self.fc_sim_dict[key] = np.array(fc_list)
            self.fcd_sim_dict[key] = np.array(fcd_list)
            ###########################################################################################
            self.fc_ave_dict[key] = np.mean(self.fc_sim_dict[key], axis=0)
            fcd_hist_sim, _ = np.histogram(np.array(fcd_entries), 
                                           bins=self.fcd_cdf_n_bins, 
                                           range=(self.fcd_cdf_vmin, self.fcd_cdf_vmax))
            fcd_cumsum_sim = np.cumsum(fcd_hist_sim)
            fcd_cdf_sim = fcd_cumsum_sim / fcd_cumsum_sim[-1]
            self.fcd_cdf_dict[key] = fcd_cdf_sim
            self.fcd_pdf_dict[key] = np.diff(fcd_cdf_sim) / ((self.fcd_cdf_vmax-self.fcd_cdf_vmin)/self.fcd_cdf_n_bins)
            ###########################################################################################

    def save_data(self):
        print('Saving data...')
        tic = time.time()
        np.savez(os.path.join(self.save_dir, 'parameter.npz'), **self.parameter_dict)
        np.savez(os.path.join(self.save_dir, 'signals_warmup_downsampled.npz'), **self.signal_warmup_downsampled_dict)
        np.savez(os.path.join(self.save_dir, 'signals_run_downsampled.npz'), **self.signal_run_downsampled_dict)
        np.savez(os.path.join(self.save_dir, 'fc_sim.npz'), **self.fc_sim_dict)
        np.savez(os.path.join(self.save_dir, 'fc_ave.npz'), **self.fc_ave_dict)
        np.savez(os.path.join(self.save_dir, 'fcd_sim.npz'), **self.fcd_sim_dict)
        np.savez(os.path.join(self.save_dir, 'fcd_cdf.npz'), **self.fcd_cdf_dict)
        toc = time.time()
        print('Saving time: {:.2f}s'.format(toc-tic))
    
    def plot_fc_ave(self):
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        for i, key in enumerate(self.fc_ave_dict.keys()):
            ax = axes[i//2, i%2]
            fc_tmp = self.fc_ave_dict[key]
            np.fill_diagonal(fc_tmp, np.nan)
            sns.heatmap(fc_tmp, ax=ax, cmap='coolwarm', cbar=True, square=True)
            ax.set_title(key)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir_dict['average'], 'fc.png'), bbox_inches='tight', dpi=500)
        plt.close()

    def plot_fc_corr(self):
        fc_df = pd.DataFrame()
        for key, fc in self.fc_ave_dict.items():
            fc_df[key] = fc[np.triu_indices_from(fc, k=1)]
        fc_corr = fc_df.corr()

        plt.figure(figsize=(5, 4))
        sns.heatmap(fc_corr, cbar=True, square=True, annot=True, fmt='.3f', cmap='viridis')
        plt.xticks(rotation=0)
        plt.title('Correlation of FC Matrices')
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir_dict['average'], 'fc_corr.png'), bbox_inches='tight', dpi=500)
        plt.close()

    def plot_fc_corr_boxplot(self):
        fc_corr_dict = {}
        for model_name, fc_bacth in self.fc_sim_dict.items():
            fc_corr_list = []
            for batch_idx in range(fc_bacth.shape[0]):
                fc_tmp = fc_bacth[batch_idx, :, :]
                fc_tmp_entries = fc_tmp[np.triu_indices_from(fc_tmp, k=1)]
                fc_emp_entries = self.fc_emp[np.triu_indices_from(self.fc_emp, k=1)]
                fc_corr_list.append(np.corrcoef(fc_tmp_entries, fc_emp_entries)[0, 1])
            fc_corr_dict[model_name] = np.array(fc_corr_list)
        fc_corr_df = pd.DataFrame(fc_corr_dict)

        plt.figure(figsize=(4, 4))
        sns.boxplot(fc_corr_df)
        plt.xlabel('Model')
        plt.ylabel('Correlation')
        plt.title('Correlation with Empirical FC')
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir_dict['average'], 'fc_corr_boxplot.png'), bbox_inches='tight', dpi=500)
        plt.close()

    def plot_fcd_cdf(self):
        df_fcd_cdf = pd.DataFrame(self.fcd_cdf_dict)
        plt.figure(figsize=(4, 3))
        sns.lineplot(data=df_fcd_cdf)
        plt.title('Off-diagonal entries of FCD Matrix')
        plt.xticks(ticks=np.arange(int(self.fcd_cdf_vmin*self.fcd_cdf_n_bins), 
                                   int(self.fcd_cdf_vmax*self.fcd_cdf_n_bins)+1, 
                                   int(self.fcd_cdf_n_bins/5)), 
                   labels=np.round(np.arange(self.fcd_cdf_vmin, 
                                             self.fcd_cdf_vmax+1/self.fcd_cdf_n_bins, 
                                             (self.fcd_cdf_vmax-self.fcd_cdf_vmin)/5), 
                                            1))
        plt.xlabel('Entry Values')
        plt.ylabel('CDF')
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir_dict['average'], 'fcd_cdf.png'), bbox_inches='tight', dpi=500)
        plt.close()

    def plot_fcd_pdf(self):
        df_fcd_pdf = pd.DataFrame(self.fcd_pdf_dict)
        plt.figure(figsize=(4, 3))
        sns.lineplot(data=df_fcd_pdf)
        plt.title('Off-diagonal entries of FCD Matrix')
        plt.xticks(ticks=np.arange(int(self.fcd_cdf_vmin*self.fcd_cdf_n_bins), 
                                   int(self.fcd_cdf_vmax*self.fcd_cdf_n_bins)+1, 
                                   int(self.fcd_cdf_n_bins/5)), 
                   labels=np.round(np.arange(self.fcd_cdf_vmin, 
                                             self.fcd_cdf_vmax+1/self.fcd_cdf_n_bins, 
                                             (self.fcd_cdf_vmax-self.fcd_cdf_vmin)/5), 
                                            1))
        plt.xlabel('Entry Values')
        plt.ylabel('PDF')
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir_dict['average'], 'fcd_pdf.png'), bbox_inches='tight', dpi=500)
        plt.close()

    def plot_fcd_ks(self):
        fcd_ks_df = pd.DataFrame(columns=self.fcd_cdf_dict.keys(), index=self.fcd_cdf_dict.keys())
        for key1, fcd_cdf1 in self.fcd_cdf_dict.items():
            for key2, fcd_cdf2 in self.fcd_cdf_dict.items():
                ks_tmp = np.max(np.abs(fcd_cdf1 - fcd_cdf2))
                fcd_ks_df.loc[key1, key2] = ks_tmp
        fcd_ks_df = fcd_ks_df.astype(float)

        plt.figure(figsize=(5, 4))
        sns.heatmap(fcd_ks_df, cbar=True, square=True, annot=True, fmt='.3f', cmap='viridis')
        plt.xticks(rotation=0)
        plt.title('KS Distance of FCD Entries CDFs')
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir_dict['average'], 'fcd_ks.png'), bbox_inches='tight', dpi=500)
        plt.close()

    def plot_fcd_ks_boxplot(self):
        ks_dict = {}
        for model_name, fcd_bacth in self.fcd_sim_dict.items():
            ks_list = []
            for batch_idx in range(fcd_bacth.shape[0]):
                fcd_tmp = fcd_bacth[batch_idx, :, :]
                fcd_entries = fcd_tmp[np.triu_indices_from(fcd_tmp, k=1)]
                fcd_hist, _ = np.histogram(fcd_entries, 
                                           bins=self.fcd_cdf_n_bins, 
                                           range=(self.fcd_cdf_vmin, self.fcd_cdf_vmax))
                fcd_cumsum = np.cumsum(fcd_hist)
                fcd_cdf = fcd_cumsum / fcd_cumsum[-1]
                ks_list.append(np.max(np.abs(fcd_cdf - self.fcd_cdf_dict['Empirical'])))
            ks_dict[model_name] = np.array(ks_list)
        ks_df = pd.DataFrame(ks_dict)

        plt.figure(figsize=(4, 4))
        sns.boxplot(ks_df)
        plt.xlabel('Model')
        plt.ylabel('KS Distance')
        plt.title('KS Distance of FCD Entries CDFs')
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir_dict['average'], 'fcd_ks_boxplot.png'), bbox_inches='tight', dpi=500)
        plt.close()

    def plot_signal_imshow_batch(self):
        for batch_idx in self.progress_bar(range(self.batch_size), desc='Plotting signal imshow'):
            fig, axes = plt.subplots(3, 1, figsize=(16, 9))
            for key_idx, key in enumerate(self.signal_run_downsampled_dict.keys()):
                signal_tmp = z_score(self.signal_run_downsampled_dict[key][batch_idx, :, :], axis=0)
                axes[key_idx].imshow(signal_tmp.T, vmax=np.max(signal_tmp), vmin=np.min(signal_tmp),
                                     cmap='RdBu_r', aspect='auto', alpha=1)
                axes[key_idx].set_title(key, fontsize=15)
            for ax in axes:
                ax.set_yticklabels([])
                ax.set_ylabel('ROI', fontsize=13)
                ax.tick_params(axis='both', which='both', length=0)
            for ax in axes[:-1]:
                ax.set_xticklabels([])
            axes[-1].set_xlabel('Time (TR={}s)'.format(self.TR), fontsize=13)
            plt.tight_layout()
            plt.savefig(os.path.join(self.fig_dir_dict[batch_idx], 'signal.png'), 
                        bbox_inches='tight', dpi=500)
            plt.close()

    def plot_signal_lineplot_batch(self):
        for batch_idx in self.progress_bar(range(self.batch_size), desc='Plotting signal lineplot'):
            time_show = 300000
            timepoints = np.arange(0, time_show)*self.dt
            fig_dir_tmp = os.path.join(self.fig_dir_dict[batch_idx], 'signal')
            os.makedirs(fig_dir_tmp, exist_ok=True)
            for roi_idx in range(3):
                plt.figure(figsize=(12, 3))
                plt.plot(timepoints, z_score(self.signal_run_dict['Balloon'][batch_idx, :time_show, roi_idx], axis=0), label='Balloon')
                plt.plot(timepoints, z_score(self.signal_run_dict['Volterra'][batch_idx, :time_show, roi_idx], axis=0), label='Volterra')
                plt.title('Simulated BOLD Signals of ROI {}'.format(roi_idx+1))
                plt.xlabel('Time (s)')
                plt.ylabel('Signal (z-score)')
                plt.legend(loc='upper right')
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir_tmp, 'roi={}.png'.format(roi_idx+1)), bbox_inches='tight', dpi=200)
                plt.show()

    def plot_fc_batch(self):
        for batch_idx in self.progress_bar(range(self.batch_size), desc='Plotting FC'):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for key_idx, key in enumerate(self.fc_sim_dict.keys()):
                fc_tmp = self.fc_sim_dict[key][batch_idx, :, :]
                axes[key_idx].imshow(fc_tmp)
                axes[key_idx].set_title(key, fontsize=15)
            for ax in axes:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xlabel('ROI', fontsize=13)
                ax.set_ylabel('ROI', fontsize=13)
                ax.tick_params(axis='both', which='both', length=0)
            plt.tight_layout()
            plt.savefig(os.path.join(self.fig_dir_dict[batch_idx], 'fc.png'), 
                        bbox_inches='tight', dpi=500)
            plt.close()
    
    def plot_fcd_batch(self):
        for batch_idx in self.progress_bar(range(self.batch_size), desc='Plotting FCD'):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for key_idx, key in enumerate(self.fcd_sim_dict.keys()):
                fcd_tmp = self.fcd_sim_dict[key][batch_idx, :, :]
                fcd_tril = fcd_tmp[np.triu_indices_from(fcd_tmp, k=1)]
                fcd_percentile = np.percentile(fcd_tril, 95)
                axes[key_idx].imshow(fcd_tmp, cmap='RdBu_r', vmax=fcd_percentile)
                axes[key_idx].set_title(key, fontsize=15)
            for ax in axes:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xlabel('Time Window', fontsize=13)
                ax.set_ylabel('Time Window', fontsize=13)
                ax.tick_params(axis='both', which='both', length=0)
            plt.tight_layout()
            plt.savefig(os.path.join(self.fig_dir_dict[batch_idx], 'fcd.png'), 
                        bbox_inches='tight', dpi=500)
            plt.close()

    def plot(self):
        tic = time.time()
        self.plot_fc_ave()
        self.plot_fc_corr()

        self.plot_fcd_cdf()
        self.plot_fcd_pdf()
        self.plot_fcd_ks()
        self.plot_fcd_ks_boxplot()

        self.plot_signal_imshow_batch()
        self.plot_signal_lineplot_batch()
        self.plot_fc_batch()
        self.plot_fcd_batch()
        toc = time.time()
        print('Plotting time: {:.2f}s'.format(toc-tic))
        
    def prepare(self):
        self.set_seed()
        self.set_save_dir()
        self.set_fig_dir()
        self.load_vali_states()
        self.get_data()
        self.generate_noise()
        self.simulate_signal_all()
        self.compute_metrics()
        self.save_data()
    
    def run(self):
        self.prepare()
        self.plot()

