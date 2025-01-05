import jax
from jax import vmap, jit  
import jax.numpy as jnp


def rotY(beta):
    return jnp.array([
        [jnp.cos(beta), 0, jnp.sin(beta)],
        [0, 1, 0],
        [-jnp.sin(beta), 0, jnp.cos(beta)]
    ])


def rotZ(gamma):
    return jnp.array([
        [jnp.cos(gamma), -jnp.sin(gamma), 0],
        [jnp.sin(gamma), jnp.cos(gamma), 0],
        [0, 0, 1]
    ])
    

def rotZYZ(alpha, beta, gamma):
    return rotZ(alpha) @ rotY(beta) @ rotZ(gamma)

@jit
def cart2sph(x, y, z):
    r = jnp.sqrt(x**2 + y**2 + z**2)
    theta = jnp.arccos(z/r)
    phi = jnp.arctan2(y, x)
    return theta, phi % (2*jnp.pi)

@jit
def sph2cart(theta, phi, r=1.):
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    return x, y, z

@jit
def rot_sph(theta, phi, alpha, beta, gamma):
    x, y, z = sph2cart(theta, phi)
    x, y, z = rot_pt(x, y, z, alpha, beta, gamma)
    return cart2sph(x, y, z)

@jit
def rot_pt(x, y, z, alpha, beta, gamma):
    R = rotZYZ(alpha, beta, gamma)
    xyz = (R @ jnp.array([[x, y, z]]).T).squeeze()
    return xyz[0], xyz[1], xyz[2]

@jit
def rot_pts(x,y,z, alpha, beta, gamma):
    R = rotZYZ(alpha, beta, gamma)
    xyz = vmap(lambda xi,yi,zi: R @ jnp.array([[xi, yi, zi]]).T)(x, y, z)
    return xyz[:,0,0], xyz[:,1,0], xyz[:,2,0]


@jit
def inv_rot_pts(x,y,z, alpha, beta, gamma):
    return rot_pts(x, y, z, -gamma, -beta, -alpha)

@jit
def inv_rot_pt(x,y,z,alpha, beta, gamma):
    return rot_pt(x, y, z, -gamma, -beta, -alpha)


def sample_on_sphere(key, N):
    return cart2sph(*jax.random.normal(key, (3, N)))
    
    
def random_so3(key, N):
    Betas, Alphas = sample_on_sphere(key, N)
    Gammas = jax.random.uniform(key, (N,)) * 2 * jnp.pi
    return Alphas, Betas, Gammas