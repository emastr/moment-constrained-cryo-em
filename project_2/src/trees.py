import jax
from jax import vmap
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax.numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
from src.signalprocessing import F, iF, decimate_ft, smooth_diracs_ft


def unpack(tree):
    return tree['children'], tree['data']


def assign_to_segment(x, center_x):
    """Assigns a point to the left or right segment based on its position relative to center_x."""
    return int(x > center_x)


def split_points_1d(i_list, x_list, center_x):
    """Splits points into left and right segments."""
    segments_i = [[], []]
    segments_x = [[], []]
    for i, x in zip(i_list, x_list):
        idx = assign_to_segment(x, center_x)
        segments_i[idx].append(i)
        segments_x[idx].append(x)
    return segments_i, segments_x


def get_1d_tree(point_ids, points_x, center_x, width, max_points=1, data_func=None):
    """Recursively constructs a 1D tree."""
    if len(points_x) <= max_points:
        children = None
    else:
        segments_i, segments_x = split_points_1d(point_ids, points_x, center_x)
        center_x_list = (center_x - width / 4, center_x + width / 4)
        widths = (width / 2, width / 2)
        children = tuple(
            get_1d_tree(i, x, cx, w, max_points, data_func)
            for i, x, cx, w in zip(segments_i, segments_x, center_x_list, widths)
        )
    if data_func is None:
        data = None
    else:
        data = data_func(point_ids, points_x, center_x, width)
    return {'children': children, 'data': (point_ids, points_x, center_x, width, data)}


def get_data_1d(point_ids, points_x, center_x, width):
    """Computes the data for the 1D tree. In this case, 
    we compute the fft of the empirical density function for the points."""
    
    Kmax = 16
    sigma = 5/(Kmax)#**0.5
    Nsample = 4*Kmax
    x0s = (jnp.array(points_x) - center_x)/width * jnp.pi + jnp.pi
    ft =  decimate_ft(smooth_diracs_ft(Nsample, x0s, sigma), Kmax)
    #ft =  decimate_ft(smooth_diracs_ft(5, x0s, sigma), 3)
    return ft


def is_far(node, x, ratio):
    children, (ids, xs, cx, w, data) = unpack(node)
    return bool(w/jnp.abs(x - cx) < ratio)


def is_node(node):
    return isinstance(node, dict)


def is_leaf(node):
    return isinstance(node, dict) and (node["children"] is None)


def trunc_fcn(node, x0, ratio):
    if is_node(node):
        children, (ids, xs, cx, w, data) = unpack(node)
        if not is_far(node, x0, ratio) and children is None:
            ws = [jnp.abs(x - x0) * ratio * 0.5 for x in xs]
            children = [get_1d_tree([idx], [pt], c, w, max_points=1, data_func=get_data_1d) for idx, pt, c, w in zip(ids, xs, xs, ws)]
            #assert all(is_far(child, x0, ratio) for child in children)
        else:
            children = None
        return {"children": children, "data": (ids, xs, cx, w, data)}
    else:
        return None


def flatten_around_point(x0, ratio, tree_1d):
    new_tree_1d = jax.tree.map(
        f=lambda node: trunc_fcn(node, x0, ratio), 
        tree=tree_1d, 
        is_leaf=lambda node: (is_node(node) and (is_far(node, x0, ratio) or node["children"] is None))
    )
    new_leaves_1d = jax.tree_leaves(new_tree_1d, is_leaf=is_leaf)
    new_leaves_1d = [leaf for leaf in new_leaves_1d if is_node(leaf)]
    return new_leaves_1d


def plot_1d_tree(tree, depth=0, data_plotter=None, **kwargs):
    """Visualizes the 1D tree as a series of line segments."""
    if not isinstance(tree, dict):
        print(tree)
    children, (ids, xs, cx, w, data) = unpack(tree)
    plt.plot([cx-w/2, cx-w/2, cx+w/2, cx + w/2], [-depth-1, -depth, -depth, -depth-1], **kwargs)
    if data_plotter is not None:
        data_plotter(depth, ids, xs, cx, w, data)
    if children is not None:
        for child in children:
            plot_1d_tree(child, depth+1, data_plotter, **kwargs)


def data_plotter(depth, ids, xs, cx, w, data):
    if data is not None:
        N = len(data)
        t = jnp.linspace(cx-w, cx+w, N)
        f = iF(data).real
        #f = (f - jnp.min(f))/(jnp.max(f) - jnp.min(f))*0.1 - depth - 0.5
        f = f/jnp.max(f)*0.1 - depth - 0.5
        #f = f*0.1 - depth - 0.5
        plt.plot(t[N//4:-N//4], f[N//4:-N//4])
        plt.scatter(xs, [-depth-0.6]*len(xs), color='red', s=10)