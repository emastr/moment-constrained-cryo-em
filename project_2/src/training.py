
import jax
import jax.numpy as jnp
import optax
from jax import vmap
from flax.training import train_state
import time

def make_batches(key, *data, batch_size=16, shuffle=True):
    num_batches = len(data[0]) // batch_size
    if shuffle:
        perm = jax.random.permutation(key, len(data[0]))
        data = [x[perm] for x in data]
    return [[d[i*batch_size:(i+1)*batch_size] for d in data] for i in range(num_batches)]
    
    
def compute_loss(params, func, x, y, mask):
    y_est = func(params, x, mask) 
    return jnp.mean((y_est-y)**2)# / jnp.mean((y-jnp.mean(y))**2) #return jnp.mean((y_est - y)**2/y**2)

@jax.jit
def train_step(state, x, y, mask):
    """
    Performs a single optimization step.
    Args:
        state: A TrainState object containing the optimizer state and model parameters.
        batch: A tuple (inputs, targets) for this batch.
        mask: Padding mask for the input sequences.
    Returns:
        Updated TrainState and loss value.
    """
    
    def loss_fn(params):
        return compute_loss(params, vmap(state.apply_fn, (None, 0, 0)), x, y, mask)

    # Compute gradients
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    # Apply gradients
    state = state.apply_gradients(grads=grads)
    return state, loss


# Initialize the optimizer and training state
def create_train_state(rng, model, input_shape, mask_shape, learning_rate, num_epochs):
    """
    Creates the training state, including initialized model parameters and the optimizer.
    Args:
        model: The Transformer model.
        rng: Random number generator.
        input_shape: Shape of the input tensor.
        mask_shape: Shape of the mask tensor.
        learning_rate: Learning rate for the optimizer.
    Returns:
        A TrainState object.
    """
    rng, subrng = jax.random.split(rng)
    
    dummy_inputs = jax.random.normal(rng, input_shape)
    dummy_mask = jnp.ones(mask_shape)
    params = model.init(subrng, dummy_inputs, dummy_mask)
    tx = optax.adam(learning_rate)
    #lr_schedule = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=num_epochs)
    #tx = optax.chain(
    #    optax.clip_by_global_norm(1.0),  # Gradient clipping
    #    optax.adamw(lr_schedule)  # AdamW optimizer with learning rate scheduling
    #)
    
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Example training loop
def train_model(rng, model, x_train, y_train, mask_train, batch_size, num_epochs, learning_rate):
    """
    Trains the Transformer model.
    Args:
        model: The Transformer model.
        train_data: A dataset of (inputs, targets) pairs.
        train_masks: A dataset of masks corresponding to the inputs.
        num_epochs: Number of epochs to train for.
        learning_rate: Learning rate for the optimizer.
        rng: Random number generator.
    Returns:
        Trained model parameters.
    """
    _, seq_len, embed_dim = x_train.shape  # (batch_size, seq_len, embed_dim)
    rng, subrng = jax.random.split(rng)
    state = create_train_state(subrng, model, (seq_len, embed_dim), (seq_len,), learning_rate, num_epochs)

    avg_losses = []
    max_losses = []
    min_losses = []
    
    t = time.time()
    jit_train_step = jax.jit(train_step)
    for epoch in range(num_epochs):
        rng, subrng = jax.random.split(rng)
        batches = make_batches(subrng, x_train, y_train, mask_train, batch_size=batch_size)
        epoch_losses = []
        for (x, y, mask) in batches:
            state, loss = jit_train_step(state, x, y, mask)
            epoch_losses.append(loss**0.5)
        
        epoch_losses = jnp.array(epoch_losses)
        avg_losses.append(jnp.mean(epoch_losses))
        max_losses.append(jnp.max(epoch_losses))
        min_losses.append(jnp.min(epoch_losses))
        t_left = (time.time() - t) * (num_epochs-epoch)/(epoch+1)# in minutes
        t_h = jnp.floor(t_left/3600)
        t_m = jnp.floor((t_left - t_h*3600)/60)
        t_s = t_left - t_m*60 - t_h*3600
        print(f"Epoch {epoch}/{num_epochs}, {t_h:.0f}h{t_m:.0f}m{t_s:.0f}s left, Avg. Loss: {avg_losses[-1]:.2e}, Max Loss: {max_losses[-1]:.2e}, Min Loss: {min_losses[-1]:.2e}", end='\r')

    return state.params, avg_losses, max_losses, min_losses