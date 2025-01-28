import jax
jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
from jax import vmap, grad, jit
from jax.scipy.special import sph_harm
from geometry import cart2sph, rot_pts
from single_degree_alignment import eval_shell

import s2fft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from jaxinterp2d import interp2d


def extend_periodic(f, theta, phi):
    # Extend in phi direction
    f = jnp.concatenate([f, f[:, 0:1]], axis=1)
    phi = jnp.concatenate([phi, phi[0:1]+jnp.pi*2], axis=0)
    
    # Extend in theta direction (need an even number of points for this to work)
    f = jnp.concatenate([f[0:1, ::-1], f], axis=0) # Mirrored
    theta = jnp.concatenate([-theta[0:1], theta])
    return f, theta, phi


def fast_sph_harm_to_pts(theta_eval, phi_eval, fm, ms, L, Lmax):
    sampling_grid = "mw"
    phi_grid = s2fft.sampling.s2_samples.phis_equiang(Lmax, sampling_grid)
    theta_grid = s2fft.sampling.s2_samples.thetas(Lmax, sampling_grid)

    fm_full = jnp.zeros((Lmax, 2*Lmax-1), dtype=jnp.complex128)
    fm_full = fm_full.at[L-1, ms+Lmax-1].set(fm) #L-1
    f_grid = s2fft.inverse_jax(fm_full, Lmax, sampling=sampling_grid, L_lower=L-1)
    f_grid, theta_grid, phi_grid = extend_periodic(f_grid, theta_grid, phi_grid)
    f_eval = interp2d(theta_eval, phi_eval, theta_grid, phi_grid, f_grid)
    return f_eval


def sph_harm_to_pts(theta_eval, phi_eval, fm, ms, Lmax):
    ls = jnp.ones_like(ms)*Lmax
    evals = vmap(lambda theta, phi: eval_shell(theta, phi, fm, ls, ms, Lmax), (0, 0))(theta_eval, phi_eval)
    return evals


def precompute_lsq(Alphas, Betas, Gammas, ms, Lmax):    
    # Compute spherical coordinates of the data
    Ncircle = 2*Lmax + 1
    t = jnp.linspace(0, 2*jnp.pi, Ncircle + 1)[:-1]
    x0, y0, z0 = jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)
    XYZ = jnp.array(vmap(rot_pts, (None, None, None, 0, 0, 0))(x0, y0, z0, Alphas, Betas, Gammas))
    XYZ_flat = XYZ.reshape(3,-1).T
    theta_eval, phi_eval = cart2sph(*XYZ_flat.T)

    # Compute matrix A, precompute normal equations A^H A x = A^H b.
    matvec = lambda fm: sph_harm_to_pts(theta_eval, phi_eval, fm, ms, Lmax)    
    A = vmap(matvec, in_axes=0, out_axes=1)(jnp.eye(2*Lmax+1).astype(jnp.complex128))
    lhs = A.T.conj() @ A
    lu_sol_proj = jax.scipy.linalg.lu_factor(lhs)
    solve_func = lambda f_aligned: jax.scipy.linalg.lu_solve(lu_sol_proj, A.T.conj() @ f_aligned)
    #solve_func = lambda f_aligned: jnp.linalg.solve(lhs, A.T.conj() @ f_aligned)
    return solve_func, matvec #, A, theta_eval, phi_eval



def line_search(x, y, loss_fcn, step0=1.0):
    step = step0
    loss_x = loss_fcn(x)
    for i in range(10):
        x_new = (1-step)*x + step*y
        if loss_fcn(x_new) < loss_x:
            return x_new, step
        step *= 0.5
    print("Warning: Line search failed")
    return x_new, x