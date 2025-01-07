import sys 
sys.path.append('../src/')

import jax.numpy as jnp
from jax import vmap, jit
from geometry import *
from single_degree_alignment import *

EPS = 1e-10

def main():
    test_ifft_shifted()
    test_fft_shifted()
    test_eval_shell()
    test_rot_sph_harm()
    test_rot_slice_sph()
    test_get_slicing_weights()
    test_get_corr()
    test_max_corr()
    test_max_corr_slices()
    test_align_data_to_slices()
    test_align_data_to_slices_average()
    test_rigid_body_align()
    print("All tests passed.")
    


def test_ifft_shifted():
    K = 4
    k = 3
    x = jnp.linspace(0, 2*jnp.pi, 2*K+2)[:-1]
    yft = x*0 + 0j
    yft = yft.at[K-k].set(4.5j)
    yft = yft.at[K+k].set(-4.5j)
    y = ifft_shifted(yft) 
    assert jnp.linalg.norm(y-jnp.sin(k*x))<EPS
    
    
def test_fft_shifted():
    K = 4
    k = 3
    x = jnp.linspace(0, 2*jnp.pi, 2*K+2)[:-1]
    y = jnp.sin(k*x)
    yft = fft_shifted(y)
    assert abs(yft[K-k]-4.5j)<EPS
    assert abs(yft[K+k]+4.5j)<EPS
    
    
def test_eval_shell():
    pass


def test_rot_sph_harm():
    alpha = 0.1
    gamma = 0.2
    beta = 0.3
    Lmax = 5
    ms = jnp.arange(-Lmax, Lmax+1).astype(jnp.int32)
    ls = jnp.ones_like(ms)*Lmax
    
    dbeta = s2fft.utils.rotation.generate_rotate_dls(Lmax+1, beta)[-1, :, :]
    fm = random_signal(Lmax, jax.random.PRNGKey(0))
    fm_rot = rot_sph_harm(fm, alpha, dbeta, gamma, ms)
    
    thetas = jax.random.uniform(jax.random.PRNGKey(0), (100,)) * jnp.pi
    phis = jax.random.uniform(jax.random.PRNGKey(0), (100,)) * 2*jnp.pi
    
    thetas_rot, phis_rot = vmap(rot_sph, (0,0,None,None,None))(thetas, phis, -gamma, -beta, -alpha)
    
    vmap_args = (0,0,None,None,None,None)
    f_rot1 = vmap(eval_shell, vmap_args)(thetas, phis, fm_rot, ls, ms, Lmax+1)
    f_rot2 = vmap(eval_shell, vmap_args)(thetas_rot, phis_rot, fm, ls, ms, Lmax+1)
    
    assert jnp.max(jnp.abs(f_rot1-f_rot2))<EPS
    

def test_rot_slice_sph():
    alpha = 0.1
    gamma = 0.2
    beta = 0.3
    Lmax = 5
    ms = jnp.arange(-Lmax, Lmax+1).astype(jnp.int32)
    ls = jnp.ones_like(ms)*Lmax
    
    dbeta, sph_zero = get_slicing_weights(jnp.array([beta]), ms, Lmax)
    fm = random_signal(Lmax, jax.random.PRNGKey(0))
    fm_rot_slice = rot_slice_sph(fm, alpha, dbeta[0], gamma, ms, sph_zero)
    
    thetas = jnp.zeros(100) + jnp.pi/2
    phis = jax.random.uniform(jax.random.PRNGKey(0), (100,)) * 2*jnp.pi
    thetas_rot, phis_rot = vmap(rot_sph, (0,0,None,None,None))(thetas, phis, alpha, beta, gamma)
    
    vmap_args = (0,0,None,None,None,None)
    f_rot1 = vmap(eval_fourier, (0, None, None))(phis, fm_rot_slice, ms)
    f_rot2 = vmap(eval_shell, vmap_args)(thetas_rot, phis_rot, fm, ls, ms, Lmax) # Used to say Lmax + 1
    assert jnp.max(jnp.abs(f_rot1-f_rot2))<EPS


def test_get_slicing_weights():
    pass


def test_get_corr():
    N = 10
    a = jax.random.normal(jax.random.PRNGKey(0), (N,)) + 1j*jax.random.normal(jax.random.PRNGKey(0), (N,))
    b = jax.random.normal(jax.random.PRNGKey(0), (N,)) + 1j*jax.random.normal(jax.random.PRNGKey(0), (N,))
    
    # With FFTs
    afft = fft_shifted(a)
    bfft = fft_shifted(b)
    corr_1 = get_corr(afft, bfft)
    
    #With shifts + inner product
    corr_func = lambda s: jnp.sum(a * jnp.conjugate(jnp.roll(b, -s))).real
    corr_2 = vmap(corr_func)(jnp.arange(N))
    
    assert jnp.max(jnp.abs(corr_1 - corr_2))<EPS
    

def test_max_corr():
    N = 74
    n = 31
    a = jax.random.normal(jax.random.PRNGKey(0), (N,)) + 1j*jax.random.normal(jax.random.PRNGKey(0), (N,))
    b = jnp.roll(a, n)
    _, idx_max = max_corr(fft_shifted(a), fft_shifted(b))
    assert jnp.max(jnp.abs(jnp.roll(b , idx_max) - a)) < EPS


def test_max_corr_slices():
    N = 100
    M = 10
    shifts = jax.random.randint(jax.random.PRNGKey(0), (M,),0, N)
    b = jnp.zeros(N)
    b = b.at[N//2].set(1.)
    a_vec = vmap(lambda s: jnp.roll(b,s))(shifts)
    a_vec = vmap(lambda n,a: a*(M-n) + b*n)(jnp.arange(M), a_vec)
    
    bfft = fft_shifted(b)
    afft_vec = vmap(fft_shifted)(a_vec)
    argmax, meta_argmax = max_corr_slices(afft_vec, bfft)
    assert (argmax==shifts[meta_argmax]) & (meta_argmax==0)
    

def test_align_data_to_slices():
    N = 47
    M = 11
    a = jax.random.normal(jax.random.PRNGKey(0), (M,N)) + 1j*jax.random.normal(jax.random.PRNGKey(0), (M,N))
    shifts = jax.random.randint(jax.random.PRNGKey(0), (M,),0, N)
    classes = jax.random.randint(jax.random.PRNGKey(0), (M,),0, M)
    b = vmap(lambda c,s: jnp.roll(a[c],s))(classes, shifts)
    b_fft = vmap(fft_shifted)(b)
    a_fft = vmap(fft_shifted)(a)
    b_shifted, meta_argmax = vmap(align_data_to_slices, (None, 0, 0))(a_fft, b, b_fft)
    assert jnp.max(jnp.abs(b_shifted - a[meta_argmax, :]))<EPS
    

def test_align_data_to_slices_average():
    Npoint = 47
    Nclass = 11
    Ndata = 201
    a = jax.random.normal(jax.random.PRNGKey(0), (Nclass, Npoint)) + 1j*jax.random.normal(jax.random.PRNGKey(0), (Nclass,Npoint))
    shifts = jax.random.randint(jax.random.PRNGKey(0), (Ndata,), 0, Npoint)
    classes = jax.random.randint(jax.random.PRNGKey(0), (Ndata,),0, Nclass)
    b = vmap(lambda c,s: jnp.roll(a[c],s))(classes, shifts)
    b_fft = vmap(fft_shifted)(b)
    a_fft = vmap(fft_shifted)(a)
    b_shift_avg = align_data_to_slices_average(a_fft, b, b_fft)
    assert jnp.max(jnp.abs(b_shift_avg - a))<EPS
    
    
def test_rigid_body_align():
    pass


if __name__ == "__main__":
    main()