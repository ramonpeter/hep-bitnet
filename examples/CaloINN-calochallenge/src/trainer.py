import sys
import time

import torch
import torch.distributions as dist
import numpy as np

import data_util
from model import CINN
import plotting
from plotter import Plotter

import caloch_eval.evaluate as evaluate

class LogUniform(dist.TransformedDistribution):
    def __init__(self, lb, ub):
        super(LogUniform, self).__init__(dist.Uniform(torch.log(lb), torch.log(ub)),
                                            dist.ExpTransform())
class Trainer:
    """ This class is responsible for training and testing the model.  """

    def __init__(self, params, device, doc):
        """
            Initializes train_loader, test_loader, model, optimizer and scheduler.

            Parameters:
            params: Dict containing the network and training parameter
            device: Device to use for the training
            doc: An instance of the documenter class responsible for documenting the run
        """

        self.params = params
        self.device = device
        self.doc = doc

        train_loader, test_loader, layer_boundaries = data_util.get_loaders(
            params.get('data_path'),
            params.get('xml_path'),
            params.get('xml_ptype'),
            params.get('val_frac'),
            params.get('batch_size'),
            params.get('eps'),
            device,
            width_noise=params.get("width_noise", 1e-7),
            energy=params.get("single_energy", None),
            u0up_cut=params.get("u0up_cut", 7.0),
            u0low_cut=params.get("u0low_cut", 0.0),
            rew=params.get("pt_rew", 1.0),
            dep_cut=params.get("dep_cut", 1e10),
        )
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.layer_boundaries = layer_boundaries
        self.single_energy = params.get("single_energy", None)
        self.avg_gen_time = {}

        self.num_dim = train_loader.data.shape[1]

        data = torch.clone(train_loader.data)
        #if self.params.get("extra_dim_w_noise", False):
        #    data[:,:-1] = train_loader.add_noise(data[:,:-1])
        #elif self.params.get("extra_dims_w_noise", False):
        #data[:,-4:] = train_loader.add_noise(data[:,-4:])
        #else:
        if params.get("custom_noise"):
            q = self.eval_quantiles(data)
            self.q = q
            train_loader.set_quantiles(q)
            test_loader.set_quantiles(q)
            data = train_loader.add_noise_v2(data)
        else:
            data = data
            self.q = torch.tensor(self.params.get("width_noise", 1e-7), device=self.device)
        cond = torch.clone(train_loader.cond)

        model = CINN(params, data, cond)
        self.model = model.to(device)
        self.set_optimizer(steps_per_epoch=len(train_loader))

        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        print('model size: {:.3f}MB'.format(size_all_mb))

        self.losses_train = {'inn': [], 'kl': [], 'total': []}
        self.losses_test = {'inn': [], 'kl': [], 'total': []}
        self.learning_rates = []

    def eval_quantiles(self, data: torch.Tensor) -> torch.Tensor:
        cp = torch.clone(data)
        cp[cp==0] = torch.nan
        quantiles = torch.nanquantile(cp, q=0.01, dim=0).reshape(1, -1)
        print(quantiles.shape)
        cp[cp==torch.nan] = 0.0
        return quantiles

    def train(self):
        """ Trains the model. """

        if self.model.bayesian:
            self.model.enable_map()

        self.latent_samples(0)
        N = len(self.train_loader.data)

        self.epoch = 0
        self.save()

        best_test_loss = 1.e6
        best_total_loss = 1.e6
        best_epoch = 0

        i = 0
        for epoch in range(1,self.params['n_epochs']+1):
            self.epoch = epoch
            train_loss = 0
            test_loss = 0
            train_inn_loss = 0
            test_inn_loss = 0
            if self.model.bayesian:
                train_kl_loss = 0
                test_kl_loss = 0
            max_grad = 0.0

            self.model.train()
            for x, c in self.train_loader:
                self.optim.zero_grad()
                inn_loss = - torch.mean(self.model.log_prob(x,c))
                if self.model.bayesian:
                    kl_loss = self.model.get_kl() / N
                    loss = inn_loss + kl_loss
                    self.losses_train['kl'].append(kl_loss.item())
                    train_kl_loss += kl_loss.item()*len(x)
                else:
                    loss = inn_loss
                loss.backward()
                if 'grad_clip' in self.params:
                    torch.nn.utils.clip_grad_norm_(self.model.params_trainable, self.params['grad_clip'], float('inf'))
                self.optim.step()

                self.losses_train['inn'].append(inn_loss.item())
                self.losses_train['total'].append(loss.item())
                train_inn_loss += inn_loss.item()*len(x)
                train_loss += loss.item()*len(x)
                if i < self.scheduler.total_steps-2:
                    self.scheduler.step()
                else:
                    pass
                self.learning_rates.append(self.optim.param_groups[0]['lr'])
                i += 1

                for param in self.model.params_trainable:
                    max_grad = max(max_grad, torch.max(torch.abs(param.grad)).item())

            self.model.eval()
            with torch.no_grad():
                for x, c in self.test_loader:
                    inn_loss = - torch.mean(self.model.log_prob(x,c))
                    if self.model.bayesian:
                        kl_loss = self.model.get_kl() / N
                        loss = inn_loss + kl_loss
                        test_kl_loss += kl_loss.item()*len(x)
                    else:
                        loss = inn_loss
                    test_inn_loss += inn_loss.item()*len(x)
                    test_loss += loss.item()*len(x)

            test_inn_loss /= len(self.test_loader.data)
            test_loss /= len(self.test_loader.data)

            train_inn_loss /= len(self.train_loader.data)
            train_loss /= len(self.train_loader.data)

            self.losses_test['inn'].append(test_inn_loss)
            self.losses_test['total'].append(test_loss)

            if self.model.bayesian:
                test_kl_loss /= len(self.test_loader.data)
                train_kl_loss /= len(self.train_loader.data)
                self.losses_test['kl'].append(test_kl_loss)

            if self.model.bayesian:
                max_bias = 0.0
                max_mu_w = 0.0
                min_logsig2_w = 100.0
                max_logsig2_w = -100.0
                for name, param in self.model.named_parameters():
                    if 'bias' in name:
                        max_bias = max(max_bias, torch.max(torch.abs(param)).item())
                    if 'mu_w' in name:
                        max_mu_w = max(max_mu_w, torch.max(torch.abs(param)).item())
                    if 'logsig2_w' in name:
                        min_logsig2_w = min(min_logsig2_w, torch.min(param).item())
                        max_logsig2_w = max(max_logsig2_w, torch.max(param).item())

            print('')
            print(f'=== epoch {epoch} ===')
            print(f'inn loss (train): {train_inn_loss}')
            if self.model.bayesian:
                print(f'kl loss (train): {train_kl_loss}')
                print(f'total loss (train): {train_loss}')
            print(f'inn loss (test): {test_inn_loss}')
            if self.model.bayesian:
                print(f'kl loss (test): {test_kl_loss}')
                print(f'total loss (test): {test_loss}')
            print(f'lr: {self.optim.param_groups[0]["lr"]}')
            if self.model.bayesian:
                print(f'maximum bias: {max_bias}')
                print(f'maximum mu_w: {max_mu_w}')
                print(f'minimum logsig2_w: {min_logsig2_w}')
                print(f'maximum logsig2_w: {max_logsig2_w}')
            print(f'maximum gradient: {max_grad}')
            sys.stdout.flush()
            
            #if test_inn_loss < best_test_loss:
            #    best_test_loss = test_inn_loss
            #    self.save("_best_inn")

            #ep_diff = epoch - best_epoch
            #if test_loss < best_total_loss and ep_diff>10:
            #    best_epoch = epoch
            #    best_total_loss = test_loss
            #    self.save("_best_total")

            if epoch >= 1:
                plotting.plot_loss(self.doc.get_file('loss.png'), self.losses_train['total'], self.losses_test['total'])
                if self.model.bayesian:
                    plotting.plot_loss(self.doc.get_file('loss_inn.png'), self.losses_train['inn'], self.losses_test['inn'])
                    plotting.plot_loss(self.doc.get_file('loss_kl.png'), self.losses_train['kl'], self.losses_test['kl'])
            plotting.plot_lr(self.doc.get_file('learning_rate.png'), self.learning_rates, len(self.train_loader))

            if epoch%self.params.get("save_interval", 20) == 0 or epoch == self.params['n_epochs']:
                self.save(str(epoch))
                if not epoch == self.params['n_epochs']:
                    gen_data = self.generate(10000)
                else:
                    gen_data = self.generate(100000)

                self.latent_samples(epoch)

                # move everything to CaloChallenge evaluate.py
                if epoch < 70:
                    evaluate.main(f"-i {self.doc.basedir}/samples.hdf5 -r {self.params['val_data_path']} -m all -d {self.params['eval_dataset']} --output_dir {self.doc.basedir}/eval/{epoch}/ --cut 1.0e-3".split())
                else:
                    evaluate.main(f"-i {self.doc.basedir}/samples.hdf5 -r {self.params['val_data_path']} -m all -d {self.params['eval_dataset']} --output_dir {self.doc.basedir}/eval/{epoch}/ --cut 0.0".split())

    def set_optimizer(self, steps_per_epoch=1, no_training=False, params=None):
        """ Initialize optimizer and learning rate scheduling """
        if params is None:
            params = self.params

        self.optim = torch.optim.AdamW(
                self.model.params_trainable,
                lr = params.get("lr", 0.0002),
                betas = params.get("betas", [0.9, 0.999]),
                eps = params.get("eps", 1e-6),
                weight_decay = params.get("weight_decay", 0.)
                )

        if no_training: return

        self.lr_sched_mode = params.get("lr_scheduler", "reduce_on_plateau")
        if self.lr_sched_mode == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optim,
                    step_size = params["lr_decay_epochs"],
                    gamma = params["lr_decay_factor"],
                    )
        elif self.lr_sched_mode == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optim,
                    factor = 0.8,
                    patience = 50,
                    cooldown = 100,
                    threshold = 5e-5,
                    threshold_mode = "rel",
                    verbose=True
                    )
        elif self.lr_sched_mode == "one_cycle_lr":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optim,
                params.get("max_lr", params["lr"]*10),
                epochs = params.get("cycle_epochs") or params["n_epochs"],
                steps_per_epoch=steps_per_epoch,
                )
        elif self.lr_sched_mode == "cycle_lr":
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optim,
                base_lr = params.get("lr", 1.0e-4),
                max_lr = params.get("max_lr", params["lr"]*10),
                step_size_up= params.get("step_size_up", 2000),
                mode = params.get("cycle_mode", "triangular"),
                cycle_momentum = False,
                    )
        elif self.lr_sched_mode == "multi_step_lr":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optim,
                    milestones=[2730, 8190, 13650, 27300],
                    gamma=0.5
                    )

    def save(self, epoch=""):
        """ Save the model, its optimizer, losses, learning rates and the epoch """
        torch.save({#"opt": self.optim.state_dict(),
                    "net": self.model.state_dict(),
                    #"losses": self.losses_test,
                    #"learning_rates": self.learning_rates,
                    }#"epoch": self.epoch}
                    , self.doc.get_file(f"model{epoch}.pt"))

    def load(self, epoch=""):
        """ Load the model, its optimizer, losses, learning rates and the epoch """
        name = self.doc.get_file(f"model{epoch}.pt")
        state_dicts = torch.load(name, map_location=self.device)
        self.model.load_state_dict(state_dicts["net"])

        #self.losses_test = state_dicts.get("losses", {})
        #self.learning_rates = state_dicts.get("learning_rates", [])
        #self.epoch = state_dicts.get("epoch", 0)
        #self.optim.load_state_dict(state_dicts["opt"])
        self.model.to(self.device)

	   
    def generate_Einc_ds1(self, energy=None, sample_multiplier=1000):
        """ generate the incident energy distribution of CaloChallenge ds1 
			sample_multiplier controls how many samples are generated: 10* sample_multiplier for low energies,
			and 5, 3, 2, 1 times sample multiplier for the highest energies
		
        """
        ret = np.logspace(8,18,11, base=2)
        ret = np.tile(ret, 10)
        ret = np.array([*ret, *np.tile(2.**19, 5), *np.tile(2.**20, 3), *np.tile(2.**21, 2), *np.tile(2.**22, 1)])
        ret = np.tile(ret, sample_multiplier)
        if energy is not None:
            ret = ret[ret==energy]
        np.random.shuffle(ret)
        return ret 
    
    def generate(self, num_samples, batch_size = 1000):
        """
            generate new data using the modle and storing them to a file in the run folder.

            Parameters:
            num_samples (int): Number of samples to generate
            batch_size (int): Batch size for samlpling
        """
        self.model.eval()
        #self.model.enable_map()
        with torch.no_grad():
            if self.params.get('eval_dataset') == "2":
                logunif = LogUniform(torch.tensor(1e3), torch.tensor(1e6))
                energies = logunif.sample((num_samples,1))/1e3
            else:
                energies = (torch.tensor(self.generate_Einc_ds1(energy=self.single_energy, sample_multiplier=1000), dtype=torch.float)/1e3).reshape(-1, 1)
            samples = torch.zeros((energies.shape[0],1,self.num_dim))
            num_samples = energies.shape[0]
            times = []
            for batch in range((num_samples+batch_size-1)//batch_size):
                    #self.model.reset_random()
                    start = batch_size*batch
                    stop = min(batch_size*(batch+1), num_samples)
                    energies_l = energies[start:stop].to(self.device)
                    t1 = time.time()
                    samples[start:stop] = self.model.sample(1, energies_l)
                    t_diff = time.time() - t1
                    times.append( t_diff/(stop-start) )
            self.avg_gen_time[str(batch_size)] = np.array(times).mean() 
            samples = samples[:,0,...].cpu().numpy()
            energies = energies.cpu().numpy()
        samples -= self.params.get("width_noise", 1e-7)
        data_util.save_data(
            data = data_util.postprocess(
                samples,
                energies,
                layer_boundaries=self.layer_boundaries,
                threshold=self.params.get("width_noise", 1e-7),
                quantiles=self.q.cpu().numpy(),
                #rew=self.params.get("pt_rew", 1.0)
            ),
            filename = self.doc.get_file('samples.hdf5')
        )
        data = data_util.postprocess(
                samples,
                energies,
                layer_boundaries=self.layer_boundaries,
                threshold=self.params.get("width_noise", 1e-7),
                quantiles=self.q.cpu().numpy(),
                #rew=self.params.get("pt_rew", 1.0)
            )
        return data

    def latent_samples(self, epoch=None):
        """
            Plot latent space distribution. 

            Parameters:
            epoch (int): current epoch
        """
        self.model.eval()
        with torch.no_grad():
            samples = torch.zeros(self.train_loader.data.shape)
            stop = 0
            for x, c in self.train_loader:
                start = stop
                stop += len(x)
                samples[start:stop] = self.model(x,c)[0].cpu()
            samples = samples.numpy()
        plotting.plot_latent(samples, self.doc.basedir, epoch)

    def plot_uncertaintys(self, plot_params, num_samples=100000, num_rand=30, batch_size = 10000):
        """
            Plot Bayesian uncertainties for given observables.

            Parameters:
            plot_params (dict): Parameters for the plots to make
            num_samples (int): Number of samples to draw for each instance of the Bayesian parameters
            num_rand (int): Number of random stats to use for the Bayesian parameters
            batch_size (int): Batch size for samlpling
        """
        l_plotter = Plotter(plot_params, self.doc.get_file('plots/uncertaintys'))
        data = data_util.load_data(
            data_file=self.params.get('train_data_path'),
            mask=self.params.get("mask", 0))
        l_plotter.bin_train_data(data)
        self.model.eval()
        for i in range(num_rand):
            self.model.reset_random()
            with torch.no_grad():
                energies = 1.0e3*torch.rand((num_samples,1)) + 0.256
                samples = torch.zeros((num_samples,1,self.num_dim))
                for batch in range((num_samples+batch_size-1)//batch_size):
                        start = batch_size*batch
                        stop = min(batch_size*(batch+1), num_samples)
                        energies_l = energies[start:stop].to(self.device)
                        samples[start:stop] = self.model.sample(1, energies_l).cpu()
                samples = samples[:,0,...].cpu().numpy()
                energies = energies.cpu().numpy()
            samples -= self.params.get("width_noise", 1e-7)
            data = data_util.postprocess_wenergy(
                    samples,
                    energies,
                    layer_boudaries=self.layer_boundaries,
                    threshold=params.get("width_noise", 1e-7)
                )
            l_plotter.update(data)
        l_plotter.plot()
