# Neural Squarer
A simple dense neural network written in python using JAX that approximates `x^2`

# Building
run `main.py` in Linux or WSL2 (JAX does not have windows support)

## How it works
The program is a dense neural network made of 5 densely connected 25-parameter layers that work to approximate the
square of values -10 through 10. Type 'exit' to stop training.

## Accuracy
After a few minutes on my RTX 3080, the accuracy of estimation is within `0.012` of the real value.