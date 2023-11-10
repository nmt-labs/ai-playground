import numpy as np
import random

class Perceptron:
    def __init__(self, n):
        self.weights = [random.uniform(-1, 1) for _ in range(n)]
        self.bias = random.uniform(-1, 1)

    def predict(self, inputs):
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return 1 if weighted_sum >= 0 else 0

    def train(self, training_data, epochs, learning_rate):
        for _ in range(epochs):
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction
                for i in range(len(self.weights)):
                    self.weights[i] += learning_rate * error * inputs[i]
                self.bias += learning_rate * error

def train_perceptron_for_logic_gate(gate, n):
    if gate == "AND":
        training_data = [(inputs, int(all(inputs))) for inputs in (zip(*[[0, 1] for _ in range(n)]))]
    elif gate == "OR":
        training_data = [(inputs, int(any(inputs))) for inputs in (zip(*[[0, 1] for _ in range(n)]))]

    perceptron = Perceptron(n)
    perceptron.train(training_data, epochs=1000, learning_rate=0.1)

    return perceptron

def test_perceptron(perceptron, n):
    test_data = [(inputs, int(all(inputs))) for inputs in (zip(*[[0, 1] for _ in range(n)]))]
    for inputs, target in test_data:
        prediction = perceptron.predict(inputs)
        print(f"Entrada: {inputs} => Saída: {prediction} (Esperado: {target})")

n = input("Insert number of entries: ")  # Número de entradas
and_perceptron = train_perceptron_for_logic_gate("AND", n)
print("Perceptron para a função AND:")
test_perceptron(and_perceptron, n)

or_perceptron = train_perceptron_for_logic_gate("OR", n)
print("Perceptron para a função OR:")
test_perceptron(or_perceptron, n)

xor_test_data = [(inputs, int(inputs[0] ^ inputs[1])) for inputs in [(0, 0), (0, 1), (1, 0), (1, 1)]]
print("Perceptron para a função XOR:")
for inputs, target in xor_test_data:
    prediction = and_perceptron.predict(inputs)
    print(f"Entrada: {inputs} => Saída: {prediction} (Esperado: {target})")