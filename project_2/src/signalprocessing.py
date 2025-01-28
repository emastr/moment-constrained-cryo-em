from jax import vmap
import jax.numpy as jnp
from jax.numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq



def F(x):
    return fftshift(fft(x))

def get_ks(N):
    return fftshift(fftfreq(N, 1./N))

def iF(x):
    return ifft(ifftshift(x))

def diffuse_ft(x_ft, sigma):
    return x_ft*jnp.exp(-sigma**2*get_ks(len(x_ft))**2/2)

def shift_ft(x_ft, a):
    return x_ft*jnp.exp(-1j*a*get_ks(len(x_ft)))

def dirac_ft(N):
    return jnp.ones((N,))

def smooth_dirac_ft(N, x0, sigma):
    return diffuse_ft(shift_ft(dirac_ft(N), x0), sigma)

def smooth_diracs_ft(N, x0s, sigma):
    return jnp.sum(vmap(lambda x0: smooth_dirac_ft(N, x0, sigma))(x0s), axis=0)

def decimate_ft(x_ft, M, retain_shape=True):
    # Eliminate high frequencies
    N = len(x_ft)
    if retain_shape:
        x_ft = ifftshift(x_ft)
        x_ft = x_ft.at[M:N-M].set(0.) 
        return fftshift(x_ft)
    else:
        return x_ft[N//2-M:N//2+M]
