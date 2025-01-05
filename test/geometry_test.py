import sys 
sys.path.append('../src/')

import jax.numpy as jnp
from jax import vmap, jit
from geometry import *


def main():
    test_rotZYZ()
    test_cart2sph()
    test_sph2cart()
    test_cart2sph_sph2cart()
    test_rot_sph()
    test_rot_pt()
    test_rot_pts()
    test_inv_rot_pts()
    test_inv_rot_pt()
    print("All tests passed!")


def ar(x):
    return jnp.array([x])

def isclose_tpl(a, b, atol=1e-6):
    return jnp.isclose(jnp.array(list(a)).flatten(), jnp.array(list(b)).flatten(), atol=atol).all()

def test_rotZYZ():
    R = rotZYZ(-jnp.pi/2, jnp.pi, jnp.pi/2)
    assert(jnp.isclose(R @ jnp.array([[0.],[1.],[0.]]), jnp.array([[0.], [-1.], [0.]]), atol=1e-6).all())


def test_cart2sph():
    assert(isclose_tpl(cart2sph(1., 0., 0.), (jnp.pi/2., 0.)))
    assert(isclose_tpl(cart2sph(0., 1., 0.), (jnp.pi/2, jnp.pi/2)))
    assert(isclose_tpl(cart2sph(0., 0., 1.), (0., 0.)))


def test_sph2cart():
    assert(isclose_tpl((1., 0., 0.), sph2cart(jnp.pi/2., 0.)))
    assert(isclose_tpl((0., 1., 0.), sph2cart(jnp.pi/2, jnp.pi/2)))
    assert(isclose_tpl((0., 0., 1.), sph2cart(0., 0.)))
    

def test_cart2sph_sph2cart():
    x, y, z = jax.random.normal(jax.random.PRNGKey(0), (3,))
    r = jnp.sqrt(x**2 + y**2 + z**2)
    x, y, z = x / r, y / r, z / r
    assert(isclose_tpl((x, y, z), sph2cart(*cart2sph(x, y, z))))


def test_rot_sph():
    assert(isclose_tpl(rot_sph(jnp.pi/2, jnp.pi/2, -jnp.pi/2, jnp.pi, jnp.pi/2), (jnp.pi/2, 3/2*jnp.pi)))


def test_rot_pt():
    assert isclose_tpl(rot_pt(0., 1., 0., -jnp.pi/2, jnp.pi, jnp.pi/2), (0., -1., 0.))


def test_rot_pts():
    x, y, z = jax.random.normal(jax.random.PRNGKey(0), (3, 10))
    r = jnp.sqrt(x**2 + y**2 + z**2)
    x, y, z = x / r, y / r, z / r
    x1, y1, z1 = rot_pts(x, y, z, 0.2, 0.1, -2)
    for xi, yi, zi, x1i, y1i, z1i in zip(x, y, z, x1, y1, z1):
        assert(isclose_tpl((x1i, y1i, z1i), rot_pt(xi, yi, zi, 0.2, 0.1, -2)))
        

def test_inv_rot_pts():
    x, y, z = jax.random.normal(jax.random.PRNGKey(0), (3, 10))
    assert(isclose_tpl(rot_pts(*inv_rot_pts(x, y, z, 0.2, 0.1, -2), 0.2, 0.1, -2), (x, y, z)))

def test_inv_rot_pt():
    x,y,z = jax.random.normal(jax.random.PRNGKey(0), (3,))
    assert(isclose_tpl(rot_pt(*inv_rot_pt(x, y, z, 0.2, 0.1, -2), 0.2, 0.1, -2), (x, y, z)))


if __name__ == "__main__":
    main()