"""Проверки численной устойчивости и корректности ручного backpropagation."""

import unittest

import numpy as np

from main import Net


class NetworkTests(unittest.TestCase):
    def test_softmax_and_loss_are_stable(self):
        logits = np.array([[10000.0, 10001.0, 9999.0], [-10000.0, 0.0, 10000.0]])
        probabilities = Net.softmax(logits)
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        np.testing.assert_allclose(probabilities, Net.softmax(logits + 50000))
        loss = Net.cross_entropy(logits, np.array([1, 0]))
        expected = (np.log(np.exp(-1) + 1 + np.exp(-2)) + 20000) / 2
        self.assertAlmostEqual(loss, expected)

    def test_gradients_match_finite_differences(self):
        net = Net(input_size=3, hidden_size=4, output_size=2, seed=7)
        # float64 нужен для точности численного дифференцирования.
        names = ("weight_input_hidden", "bias_hidden",
                 "weight_hidden_output", "bias_output")
        for name in names:
            setattr(net, name, getattr(net, name).astype(np.float64))
        X = np.array([[0.2, 0.7, -0.4], [-0.3, 0.8, 0.5], [0.9, -0.4, 0.1]])
        y = np.array([0, 1, 0])
        net.forward_propagation(X)
        gradients = net.gradients(y)
        epsilon = 1e-6
        for name in names:
            parameter = getattr(net, name)
            numerical = np.empty_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                net.forward_propagation(X)
                plus = net.cross_entropy(net.logits, y)
                parameter[index] = original - epsilon
                net.forward_propagation(X)
                minus = net.cross_entropy(net.logits, y)
                parameter[index] = original
                numerical[index] = (plus - minus) / (2 * epsilon)
            np.testing.assert_allclose(gradients[name], numerical,
                                       atol=1e-7, rtol=1e-5, err_msg=name)

    def test_training_learns_simple_classes(self):
        X = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        net = Net(input_size=2, hidden_size=8, output_size=2, lr=0.2)
        before, _ = net.evaluate(X, y)
        for _ in range(150):
            net.forward_propagation(X)
            net.back_propagation(y)
        after, accuracy = net.evaluate(X, y, batch_size=3)
        self.assertLess(after, before / 5)
        self.assertEqual(accuracy, 1.0)
        np.testing.assert_array_equal(net.predict(X), y)


if __name__ == "__main__":
    unittest.main()
