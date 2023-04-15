import numpy as np
import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm

from torchsummary import summary


def linear_sn(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))



class Gen_MLP(nn.Module):
    def __init__(self, zdim, out_shape):
        super().__init__()
        self.in_shape = (zdim,)
        self.out_shape = out_shape

        def _linear(chin, chout, bn):
            layers = [ nn.Linear(chin, chout, bias=(not bn)) ]
            if (bn): layers.append(nn.BatchNorm1d(chout))
            layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        use_batchnorm = True

        nin = zdim
        out_units = [256, 512, 1024]
        layers = []
        for nout in out_units:
            layers.append(_linear(nin, nout, bn=use_batchnorm))
            nin = nout

        # Final layer with TanH
        layers += [
            nn.Linear(nin, np.prod(out_shape)),
            nn.Tanh()
        ]

        self.mlp = nn.Sequential(*layers)


    def forward(self, z):
        # Get the in/out shapes
        B,_ = z.shape
        C,H,W = self.out_shape

        # Generate a new image via MLP
        out = self.mlp(z)

        # Reshape flat tensor into correct image shape
        out = out.view(B,C,H,W)
        return out



class Dis_MLP(nn.Module):
    def __init__(
        self, 
        in_shape, 
        use_spectral_norm=False
    ):
        super().__init__()
        self.in_shape = in_shape

        # Assemble the MLP
        nin = np.prod(in_shape)
        out_units = [1024, 512, 256, 1]
        layers = [ nn.Flatten() ]
        for nout in out_units:
            if (use_spectral_norm):
                layers += [ linear_sn(nin, nout) ]
            else:
                layers += [ nn.Linear(nin, nout) ]            
            if (nout > 1): layers.append(nn.ReLU())
            nin = nout

        self.mlp = nn.Sequential(*layers)



    def forward(self, x):
        return self.mlp(x)





def create_models(args, shape, device):

    use_sn = False

    # Spectral normalization with hinge loss
    if (args.loss == "hinge"): use_sn = True
    
    result = {
        # MLP generator
        "gen":  Gen_MLP(zdim=args.zdim, out_shape=shape),

        # MLP discriminator
        "dis":  Dis_MLP(in_shape=shape, use_spectral_norm=use_sn)
    }

    # Move the models to the GPU and print their stats
    for k,v in result.items():
        v.train()
        result[k] = v.to(device)

        print("Summary for: ", k)
        summary(result[k], input_size=result[k].in_shape)
        print("")

    return result
