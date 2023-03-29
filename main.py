import os
import threading
import select
import sys

terminate_program = False
def check_exit():
    global terminate_program
    while not terminate_program:
        ready, _, _ = select.select([sys.stdin], [], [], 1)  # Check for input with a timeout of 1 second
        if ready:
            user_input = sys.stdin.readline()
            if user_input.strip().lower() == "exit":
                terminate_program = True

t = threading.Thread(target=check_exit)
t.daemon = True  # Set the thread as a daemon thread, so it exits when the main program exits
t.start()

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
os.environ['JAX_DEBUG_NANS'] = 'True'

print("Initializing JAX!")
import jax
import jax.numpy as jnp
import jax.random as rand
from jax import jit

print(f"Recognized Devices: {jax.devices()}")

def layer_params(m, n, key, scale=1e-2):
    mKey, bKey = rand.split(key)
    return scale * rand.normal(mKey, (n, m)), scale * rand.normal(bKey, (n,))

def dense_layer(sizes, key):
    keys = rand.split(key, len(sizes))
    return [layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

@jit
def relu(x, alpha=0.01):
  return jnp.maximum(alpha * x, x)

@jit
def propagate(params, input):
    activations = input
    for idx, (m, b) in enumerate(params[:-1]):
        outputs = jnp.dot(m, activations) + b
        if idx % 2 == 1:  # Add a skip connection every 2 layers
            activations = relu(outputs) + activations
        else:
            activations = relu(outputs)

    fm, fb = params[-1]
    result = jnp.dot(fm, activations) + fb
    return result


batched_propagate = jax.vmap(propagate, in_axes=(None, 0))

@jit
def loss(params, input, output):
    pred = batched_propagate(params, input)
    return jnp.mean(jnp.abs(output - pred))

@jit
def update(params, input, output, lr):
    grads = jax.grad(loss)(params, input, output)
    return [(m - lr * dm, b - lr * db) for (m, b), (dm, db) in zip(params, grads)]

def create_batches(xs, ys, batch_size):
    num_batches = len(xs) // batch_size
    xs_batches = jnp.split(xs[:num_batches * batch_size], num_batches)
    ys_batches = jnp.split(ys[:num_batches * batch_size], num_batches)
    return xs_batches, ys_batches

def exponential_decay(initial_lr, decay_rate, epoch, min_lr=1e-6):
    return max(initial_lr * decay_rate ** epoch, min_lr)

def main():
    key = rand.PRNGKey(4)

    print("Initializing Network Parameters!")
    layers = [1, 25, 25, 25, 25, 25, 1]
    parameters = dense_layer(layers, key)

    print("Initializing Training Data Set!")
    key, _ = rand.split(key)
    xs = rand.uniform(key, (1_000_000, 1), minval=-10, maxval=10)
    ys = xs ** 2

    print("Initializing Training Batches!")
    xs_batches, ys_batches = create_batches(xs, ys, 1_000)

    initial_lr = 0.01
    decay_rate = 0.99

    print("Starting Training!")
    for i in range(100000):
        if terminate_program:
            break

        learning_rate = exponential_decay(initial_lr, decay_rate, i)
        for xs_batch, ys_batch in zip(xs_batches, ys_batches):
            parameters = update(parameters, xs_batch, ys_batch, learning_rate)

        key, _ = rand.split(key)
        test = rand.randint(key, minval=0, maxval=1_000_000, shape=(1,))[0]
        print(f"epoch={i:-3d} | lr={learning_rate:5.4f} | x0={xs[test][0]:6.3f} | x0^2={ys[test][0]:6.3f} | ~x^2={propagate(parameters, xs[test])[0]:6.3f} | loss={jnp.mean(loss(parameters, xs, ys)):6.3f}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
