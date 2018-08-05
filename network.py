import random
import numpy as np

class network(object):
	def __init__(self, layer_sizes):
		self.num_of_layers = len(layer_sizes)
		self.layer_sizes = layer_sizes
		self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
	
	def forward(self, x):
		for w, b in zip(self.weights, self.biases):
			x = sigmoid(np.dot(w, x) + b)
		return x

	def SGD(self, training_data, epochs, batch_size, eta, test_data = None):
		if test_data: n_test = len(test_data)
		n = len(training_data)
		for j in range(epochs):
			random.shuffle(training_data)
			batches = [training_data[k : k + batch_size] for k in range(0, n, batch_size)]
			for batch in batches: self.update_batch(batch, eta)
			if test_data:
				print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
			else:
				print ("Epoch {0} complete".format(j))

	def update_batch(self, batch, eta):
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		for x, y in batch:
			delta_nabla_w, delta_nabla_b = self.back_propagation(x, y)
			nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
			nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
		self.weights = [w - (eta / len(batch)) * nw for w, nw in zip(self.weights, nabla_w)]	
		self.biases = [b - (eta / len(batch)) * nb for b, nb in zip(self.biases, nabla_b)]

	def back_propagation(self, x, y):
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		activation = x
		activations = [x]
		zs = []
		for w, b in zip(self.weights, self.biases):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		nabla_b[-1] = delta
		for l in range(2, self.num_of_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
			nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
			nabla_b[-l] = delta
		return (nabla_w, nabla_b)

	def evaluate(self, test_data):
		test_result = [(np.argmax(self.forward(x)), y) for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_result)

	def cost_derivative(self, output_activations, y):
		return (output_activations - y)

def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z) * (1 - sigmoid(z))