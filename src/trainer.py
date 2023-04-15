
import torch
import torch.optim as optim

from ganlib.loggers import Loggers
from ganlib.utils import initialize_experiment, TrainingStats
from datasource import DataSource
from models import create_models
from losses import create_loss



class Trainer:
    def __init__(self, args):       
        self.args = args

        # Initialize experiment
        self.name, self.out_folder = initialize_experiment(args)
        self.stats = TrainingStats(decay=0.995)

        # CUDA / CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Get our data source
        self.ds = DataSource(args, self.device)

        # Models & optimizers
        self.models = create_models(args, self.ds.shape, self.device)
        self.opts = self.create_optimizers(self.models, args)

        # Training loss !
        self.loss = create_loss(args)



    #--------------------------------------------------------------------------
    #   Initialization
    #--------------------------------------------------------------------------

    def create_optimizers(self, models, args):

        result = {
            # Generator optimizer
            "opt_g":    optim.Adam(
                            params=models["gen"].parameters(),
                            lr=args.lr_gen,
                            betas=(0.5, 0.999)
                        ),

            # Discriminator optimizer
            "opt_d":    optim.Adam(
                            params=models["dis"].parameters(),
                            lr=args.lr_dis,
                            betas=(0.5, 0.999)
                        )
        }
        return result


    def setup(self, loggers):
        '''
            Setup our loggers
        '''
        self.log = Loggers(loggers)


    def train(self):
        '''
            Our main training loop
        '''
        self.log.training_start()
        for it in range(self.args.iters):
            stats = self._train_iteration(it)
            self.log.on_iteration(it, stats)

        self.log.training_end()


    def _train_iteration(self, it):

        # Get our stuff
        gen, dis, L = self.models["gen"], self.models["dis"], self.loss
        og, od = self.opts["opt_g"], self.opts["opt_d"]

        #----------------------------------------------------------------------
        #   Forward pass
        #----------------------------------------------------------------------

        # Get a fresh batch of real images
        x = self.ds.get()

        # Sample random noise, and generate fake images
        z = torch.randn(size=(x.size(0), self.args.zdim)).to(self.device)
        x_fake = gen(z)

        #----------------------------------------------------------------------
        #   Train Generator
        #----------------------------------------------------------------------

        og.zero_grad()
        score_fake = dis(x_fake)
        loss_gen = L.G(score_fake)
        lG = loss_gen.cpu().detach().item()
        loss_gen.backward()

        og.step()

        #----------------------------------------------------------------------
        #   Train Discriminator
        #----------------------------------------------------------------------

        od.zero_grad()
        score_real = dis(x)
        score_fake = dis(x_fake.detach())
        loss_dis = (L.D(score_real, True) + L.D(score_fake, False)) * 0.5
        lD = loss_dis.cpu().detach().item()

        # When training with WGAN-GP, we need to compute gradient penalty
        if (self.args.loss == "wgan-gp"):
            loss_dis += 10.0 * L.gradient_penalty(dis, x, x_fake)

        loss_dis.backward()
        od.step()





        # Update statistics
        return self.stats.step(g=lG, d=lD)



