import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

n = network.network([784, 30, 10])
n.SGD(training_data, 30, 10, 3.0, test_data = test_data)