import argparse

import ganlib.loggers as l
from trainer import Trainer

import torch
import lightning as LT
from lightning.fabric.strategies import DDPStrategy


def do_train(args):

    strategy = "auto"
    if args.strategy == "ddp":
        strategy="ddp"

    fabric = LT.Fabric(
        accelerator="cuda",
        devices=torch.cuda.device_count(),
        strategy=strategy,
        precision="32-true",
        num_nodes=args.num_nodes
        )
    fabric.launch()
   
    trainer = Trainer(args, fabric)

    # Setup our logging
    trainer.setup(
        loggers=[
            l.ConsoleLogger(interval=args.log_interval)
        ]
    )
    """
    #l.CheckpointMaker(trainer, interval=10*1000),
    l.ImageSampler(
        trainer, 
        interval=args.log_interval, 
        sinks=[ 
            l.ImageFileSink(trainer, only_current=True) 
        ],
        shape=(3, 6)
        )
    """

    # Commence training
    trainer.train()




if (__name__ == "__main__"):

    p = argparse.ArgumentParser()

    # Command line arguments
    p.add_argument("--name", "-n", default="", type=str, help="Experiment name")
    p.add_argument("--outfolder", "-o", default=".scratch/runs", type=str, help="Output folder for training runs")
    p.add_argument("--strategy", "-s", default="single", type=str, help="Paralelization strategy")
    p.add_argument("--num_nodes", "-nn", default=1, type=int, help="Number of parallel nodes")

    # Basic GAN settings
    p.add_argument("--loss", "-l", default="gan", type=str, help="Loss function to use for training")
    p.add_argument("--dataset", "-d", default="mnist", type=str, help="Dataset")
    p.add_argument("--batch_size", "-b", default=32, type=int, help="Batch size")        
    p.add_argument("--iters", "-it", default=1000*1000, type=int, help="Number of training iterations")
    p.add_argument("--zdim", "-z", default=100, type=int, help="Latent space dimensionality")
    p.add_argument("--lr_gen", "-lrG", default=0.0001, type=float, help="Generator learning rate")
    p.add_argument("--lr_dis", "-lrD", default=0.0002, type=float, help="Discriminator learning rate")

    # Logging
    p.add_argument("--log_interval", "-lit", default=1*1000, type=int, help="Number of training iterations")



    # Execute training
    do_train(p.parse_args())
