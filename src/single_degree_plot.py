
import matplotlib.pyplot as plt
from geometry import sph2cart, rot_pts, rot_pt
import jax.numpy as jnp


def colormap(x):
    cmap = plt.get_cmap('viridis', 256)
    return cmap(x)


def normalize(v, vmin=None, vmax=None):
    if vmin is None:
        vmin = jnp.min(v)
    if vmax is None:
        vmax = jnp.max(v)
    return (v - vmin) / (vmax - vmin)


def plot_sph_parity(f, thetas, phis):
    plt.figure(figsize=(15, 5))
    
    ax = plt.subplot(141, projection='3d')
    plt.title("Re F(r)")
    v = colormap(normalize(f.real))
    plot_sph(v, thetas, phis, ax)
    
    ax = plt.subplot(142, projection='3d')
    plt.title("Re F(-r)")
    v = colormap(normalize(f.real))
    plot_sph(v, jnp.pi - thetas, phis + jnp.pi, ax)

    ax = plt.subplot(143, projection='3d')
    plt.title("Im F(r)")
    v = colormap(normalize(f.imag))
    plot_sph(v, thetas, phis, ax)

    ax = plt.subplot(144, projection='3d')
    plt.title("-Im F(-r)")
    v = colormap(normalize(-f.imag))
    plot_sph(v, jnp.pi - thetas, phis + jnp.pi, ax)
    

def plot_sph(f, theta, phi, ax):
    theta, phi = jnp.meshgrid(theta, phi)
    x, y, z = sph2cart(theta, phi)
    ax.plot_surface(x, y, z, facecolors=f, rstride=1, cstride=1, shade=False)


def plot_rot(alphas, betas, gammas, N):
    #gammas = jnp.zeros_like(alphas)
    t = jnp.linspace(0, 2*jnp.pi, N+1)[:-1]
    x0, y0, z0 = jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)
    i1, i2 = 0, 5
    
    # Plot sampling points
    plt.figure(figsize=(15, 5))
    
    ax1 = plt.subplot(141, projection='3d')
    plt.title("Sampled angles")
    ax1.scatter(*sph2cart(betas, alphas, r=1), color='g')
    ax1.axis('equal')
    
    # Scatter rotated pooints for alpha[i], beta[i] rotation
    ax2 = plt.subplot(142, projection='3d')
    plt.title("Example rotation 1")
    ax2.scatter(*sph2cart(betas[i1:i1+1], alphas[i1:i1+1]), color='g')
    ax2.scatter(*rot_pts(x0, y0, z0, alphas[i1], betas[i1], gammas[i1]), color='r')
    ax2.plot(*[[0., p] for p in rot_pt(0, 0, 1., alphas[i1], betas[i1], gammas[i1])], color='b')
    ax2.axis('equal')
    
    ax3 = plt.subplot(143, projection='3d')
    plt.title("Example rotation 2")
    for i2 in range(min(2, len(alphas))):
        ax3.scatter(*sph2cart(betas[i2:i2+1], alphas[i2:i2+1]), color='g')
        ax3.scatter(*rot_pts(x0, y0, z0, alphas[i2], betas[i2], gammas[i2]), color='r')
        ax3.plot(*[[0., p] for p in rot_pt(0, 0, 1., alphas[i2], betas[i2], gammas[i2])], color='b')
    ax3.axis('equal')
    
    ax4 = plt.subplot(144, projection='3d')
    plt.title("Coverage")
    for i in range(len(alphas)):
        ax4.scatter(*rot_pts(x0, y0, z0, alphas[i], betas[i], gammas[i]), color='b')
    #ax.scatter(*sph2cart(betas, alphas, r=1), color='r')
    #ax.view_init(elev=0, azim=0)
    ax4.axis('equal')
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.view_init(elev=0, azim=0)

