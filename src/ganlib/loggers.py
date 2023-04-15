import os
import cv2

import torch
import numpy as np

from pathlib import Path

from .utils import arange_images


#------------------------------------------------------------------------------
#   Logger  & Loggers baseclass
#------------------------------------------------------------------------------

class Logger:
    def training_start(self):
        pass

    def training_end(self):
        pass

    def on_iteration(self, it, stats):
        pass



class Loggers(Logger):
    def __init__(self, loggers=[]):
        self.loggers = loggers

    def training_start(self):
        for l in self.loggers:
            l.training_start()

    def training_end(self):
        for l in self.loggers:
            l.training_end()

    def on_iteration(self, it, stats):
        for l in self.loggers:
            l.on_iteration(it, stats)




#------------------------------------------------------------------------------
#   CheckpointMaker
#------------------------------------------------------------------------------

class CheckpointMaker(Logger):
    def __init__(self, trainer, interval):
        self.trainer = trainer
        self.interval = interval
        self.fn = os.path.join(trainer.out_folder, "checkpoint.pt")
        self.check_dirs = os.path.join(trainer.out_folder, "checkpoints")
        os.makedirs(self.check_dirs, exist_ok=True)
        self.counter = 0

    def on_iteration(self, it, stats):
        if ((it % self.interval) == 0):
            checkpoint = {}
            for k in self.trainer.models.keys():
                checkpoint[k] = self.trainer.models[k].state_dict()
            for k in self.trainer.opts.keys():
                checkpoint[k] = self.trainer.opts[k].state_dict()
            torch.save(checkpoint, self.fn)

            # Checkpoint
            fn = os.path.join(self.check_dirs, "checkpoint-{:04d}.pt".format(self.counter))
            torch.save(checkpoint, fn)
            self.counter += 1


#------------------------------------------------------------------------------
#   ConsoleLogger 
#------------------------------------------------------------------------------

class ConsoleLogger(Logger):
    def __init__(self, interval):
        self.interval = interval

    def on_iteration(self, it, stats):
        if ((it % self.interval) == 0):
            values = [ " {}: {:07f}".format(k, stats[k]) for k in stats.keys() ]
            values = [ "it: {}".format(it) ] + values
            line = ",".join(values)
            print(line)




#------------------------------------------------------------------------------
#   Image Sampler
#------------------------------------------------------------------------------

class ImageFileSink:
    def __init__(self, trainer, only_current=True):
        self.folder = trainer.out_folder
        self.run_name = trainer.name
        self.only_current = only_current

    def write(self, i, name, image):

        # Scale !
        H,W = image.shape[0], image.shape[1]
        SF = 4
        image = cv2.resize(image, (W*SF,H*SF), interpolation=cv2.INTER_NEAREST)

        ifile = i % 100
        idir = i // 100
        fn = "{:05d}/{:03d}-{}.png".format(idir, ifile, name)
        fn = os.path.join(self.folder, "images", fn)
        fnCurrent = os.path.join(self.folder, "{}.png".format(name))
        
        # Store the image
        if (not self.only_current):
            os.makedirs(Path(fn).parent, exist_ok=True)
            cv2.imwrite(fn, image)

        os.makedirs(Path(fnCurrent).parent, exist_ok=True)
        cv2.imwrite(fnCurrent, image)


class Sinks:
    def __init__(self, sinks):
        self.sinks = sinks

    def write(self, idx, name, image):
        for s in self.sinks:
            s.write(idx, name, image)


class ImageSampler(Logger):
    def __init__(self, trainer, interval, sinks, shape):
        self.trainer = trainer
        self.interval = interval
        self.sink = Sinks(sinks)

        # Get our images
        self.shape = shape
        self.nImages = np.prod(self.shape)
        self.counter = 0

        # Random Z vector
        self.z = torch.randn(size=(self.nImages, trainer.args.zdim))
        self.z = self.z.to(trainer.device)


    def on_iteration(self, it, stats):
        if ((it % self.interval) == 0):
            # use this model
            gen = self.trainer.models["gen"]
            self.sample_models(gen, "")
            self.counter += 1

    
    def sample_models(self, m, prefix):
        # Compute images
        with torch.no_grad():
            x = m(self.z)

        # Convert to image ranges
        image = arange_images(x, shape=self.shape)

        # Output images
        self.sink.write(
            idx=self.counter, 
            name="{}{}".format(prefix, "sample"),
            image=image
            )


