import tensorflow as tf

# Variables are manipulated via the tf.Variable class.
# A tf.Variable represents a tensor whose value can be changed by running ops on it.
# Specific ops allow you to read and modify the values of this tensor.

## Creaging a Variable

with tf.name_scope("my"):
    variable = tf.Variable(1)

print("value:", variable.numpy())

## Using Variables

# To use the value of a tf.Variable in a TensorFlow graph, simply treat it like a normal tf.Tensor
variable = variable + 1
print("value:", variable.numpy())

# To assign a value to a variable, use the methods assign, assign_add
variable = tf.Variable(3)
variable.assign_add(1)
print("value:", variable.numpy())

for _ in range(0,10):
    print("s")

