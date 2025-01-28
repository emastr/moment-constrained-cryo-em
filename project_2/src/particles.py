
import jax
from jax import vmap
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax.numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
from src.signalprocessing import F, iF, decimate_ft, smooth_diracs_ft

def node_to_array(node):
    (ids, xv, cx, w, data) = node["data"]
    return jnp.concatenate([data, jnp.array([cx, w])])

def nodes_to_array(node_list):
    return jnp.concatenate([node_to_array(node)[None, :] for node in node_list], axis=0)

def array_to_density_feature(feature):
    cx = feature[-2]
    w = feature[-1]
    f = iF(feature[:-2]).real
    return  jnp.concatenate([f, jnp.array([cx, w])])  

def array_to_density_domain_feature(feature):
    cx = feature[-2]
    w = feature[-1]
    f = iF(feature[:-2]).real
    t = jnp.linspace(cx-w, cx+w, len(f)+1)[:-1]
    return  jnp.concatenate([f, point_potential(0.,t)])


def array_to_potential_feature(feature):
    cx = feature[-2]
    w = feature[-1]
    f = iF(feature[:-2]).real
    t = jnp.linspace(cx-w, cx+w, len(f)+1)[:-1]
    
    f = f * point_potential(t, 0.)
    return  jnp.concatenate([f, jnp.array([cx, w])])


def array_to_features(feature_vec, feature_func):
    return vmap(feature_func)(feature_vec)


def point_potential(x0, x1):
    return 1/(jnp.abs(x0 - x1) + 0.01)


def potential_from_points(x0, points, tol=1e-10):
    return jnp.sum(vmap(lambda x1: jax.lax.cond(jnp.abs(x0-x1)>tol, lambda x0: point_potential(x0, x1), lambda x0: 0., x0))(points))


def potential_from_vec(x0, array, tol=1e-10):
    cx = array[-2]
    w = array[-1]
    f = iF(array[:-2]).real
    t = jnp.linspace(cx-w, cx+w, len(f)+1)[:-1]
    return jax.lax.cond(jnp.abs(cx-x0)<tol, lambda x0: 0., lambda x0: jnp.sum(f * point_potential(x0, t)), x0)
    
    
def potential_from_array(x0, array):
    return jnp.sum(vmap(potential_from_vec, (None, 0))(x0, array))

