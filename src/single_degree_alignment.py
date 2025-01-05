import jax
jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
from jax import vmap, grad, jit
from jax.scipy.special import sph_harm
from geometry import cart2sph, sph2cart, rotZYZ, sample_on_sphere

import s2fft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

@jit
def eval_fourier(x, coef, ks):
    return jnp.sum(coef * jnp.exp(1j * x * ks))

@jit
def ifft_shifted(f):
    return jnp.fft.ifft(jnp.fft.ifftshift(f))# * len(f)
    
@jit
def fft_shifted(f):
    return jnp.fft.fftshift(jnp.fft.fft(f))

def get_l_and_m(Lmax):
    ms = jnp.arange(-Lmax, Lmax+1).astype(jnp.int32)
    ls = jnp.ones_like(ms) * Lmax
    return ls, ms

def eval_shell(theta, phi, fm, ls, ms, L_max):
    # sph_harm uses different conventions for theta and phi
    theta, phi = phi, theta
    return jnp.sum(sph_harm(ms, ls, jnp.array([theta]), jnp.array([phi]), n_max=L_max+1) * fm)

@jit
def rot_sph_harm(fm, alpha, d_beta, gamma, ms):
    return jnp.exp(-1j*ms*alpha) * jnp.einsum('ij,j->i', d_beta, jnp.exp(-1j*ms*gamma) * fm)

@jit
def rot_slice_sph(fm, alpha, d_beta, gamma, ms, sph_zero):
    return rot_sph_harm(fm, alpha, d_beta, gamma, ms) * sph_zero


def get_slicing_weights(betas, ms, Lmax):
    d_beta = vmap(lambda beta: s2fft.utils.rotation.generate_rotate_dls(Lmax+1, beta)[-1, :, :])(betas)
    sph_zero = sph_harm(ms, jnp.ones_like(ms)*Lmax, 0., jnp.pi/2)
    return d_beta, sph_zero


def random_signal(Lmax, key):
    key1, key2 = jax.random.split(key)
    ms = jnp.arange(-Lmax, Lmax+1).astype(jnp.int32)
    f = jax.random.normal(key1, (2*Lmax+1,)) + jax.random.normal(key2, (2*Lmax+1,)) * 1j
    f = f + (-1)**(Lmax-ms)*jnp.conjugate(f[::-1])
    return f


@jit
def get_corr(afft, bfft):
    return ifft_shifted(afft * jnp.conjugate(bfft)).real

@jit
def max_corr(afft, bfft):
    corr = get_corr(afft, bfft).real/ jnp.linalg.norm(afft) / jnp.linalg.norm(bfft)
    argmax = jnp.argmax(corr)
    return corr[argmax], argmax

@jit
def max_corr_slices(afft_vec, bfft):
    corr_max, arg_max = vmap(max_corr, (0, None))(afft_vec, bfft)
    meta_argmax = jnp.argmax(corr_max)
    return arg_max[meta_argmax], meta_argmax

@jit
def align_data_to_slices(fm_rot_slices, data, fft_data):
    arg_max, meta_argmax = max_corr_slices(fm_rot_slices, fft_data)
    return jnp.roll(data, arg_max), meta_argmax

@jit
def align_data_to_slices_average(fm_rot_slices, data, fft_data):
    vmap_aligns = vmap(align_data_to_slices, (None, 0, 0))
    aligned_data, slice_idx = vmap_aligns(fm_rot_slices, data, fft_data)
    conditional_mean = lambda idx: jnp.sum(jnp.where(slice_idx[:, None] == idx, aligned_data, jnp.zeros_like(aligned_data)), axis=0)/jnp.sum(slice_idx == idx)
    return vmap(conditional_mean)(jnp.arange(fm_rot_slices.shape[0]))
    
def rigid_body_align(fm, fm_template, ms, Lmax, Alphas, Betas, Gammas):
    d_betas, _ = get_slicing_weights(Betas, ms, Lmax)
    fm_template_rot = vmap(rot_sph_harm, (None, 0, 0, 0, None))(fm_template, Alphas, d_betas, Gammas, ms)
    argmax = jnp.argmax(vmap(lambda a,b: jnp.sum(a * jnp.conjugate(b)).real, (0, None))(fm_template_rot, fm))
    return  fm_template_rot[argmax]

