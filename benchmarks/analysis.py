import os
import numpy as onp
from iminuit import Minuit, minimize

from jax import jit, grad, random
import jax.numpy as np
from jax.scipy import special, stats
from jax import config
config.update("jax_enable_x64", True)
#config.update("jax_platform_name", "CPU")
os.environ["JAX_PLATFORM_NAME"] = "cpu"

@jit
def getAxionTemplate(mass, freqs, v0, vObs):

    v = np.sqrt(np.maximum(0, 2. * (2.*np.pi*freqs-mass) / mass))
    norm = (v > 0)/np.sqrt(np.pi)/v0/vObs
    fSHM_divV = norm*(np.exp(-(v-vObs)**2. / v0**2.) - np.exp(-(v + vObs)**2 / v0**2.))

    template = np.pi * fSHM_divV / mass
    return template

@jit
def NegLL(x, freqs, psd, axionTemp):

    A = x[0]
    x = x[1:]
    lambdaB = x[:len(x)//2]
    slope = x[len(x)//2:]

    template = A*axionTemp[None, :] + lambdaB[:, None]*(1 + slope[:, None]*freqs[None, :])
    std = np.std(psd - template, axis = 1)
    LL = stats.norm.logpdf(psd, loc = template, scale = std[:, None])
    return -2*np.sum(LL)

NegLL_Jac = jit(grad(NegLL, argnums=(0)))
