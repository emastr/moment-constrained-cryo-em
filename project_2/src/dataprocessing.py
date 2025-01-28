
import jax
import jax.numpy as jnp

def make_batches(key, *data, batch_size=16, shuffle=True):
    num_batches = len(data[0]) // batch_size
    if shuffle:
        perm = jax.random.permutation(key, len(data[0]))
        data = [x[perm] for x in data]
    return [tuple(d[i*batch_size:(i+1)*batch_size] for d in data) for i in range(num_batches)]
    
    
def pad_data(data):
    max_len = max(len(x) for x, _ in data)
    padded_x = jnp.stack([jnp.pad(x, ((0, max_len - len(x)), (0,0))) for x, _ in data])
    padded_y = jnp.stack([y for _, y in data])
    return padded_x, padded_y

def create_padding_mask(seq, pad_token=0):
    return (seq != pad_token).astype(jnp.float32)