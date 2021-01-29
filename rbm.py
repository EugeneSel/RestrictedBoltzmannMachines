import numpy as np


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


class RBM:
    def __init__(self, p, q, sigma2=.01):
        self.p = p
        self.q = q

        self.W = sigma2 * np.random.randn(self.p, self.q)
        self.a = np.zeros([1, self.p])
        self.b = np.zeros([1, self.q])

    def train(self, X, lr, epochs, batch_size):
        nb_samples = X.shape[0]
        X_ = X.copy()

        for epoch in range(epochs):
            # Shuffle data:
            np.random.shuffle(X_)
            
            for batch in range(0, X_.shape[0], batch_size):
                # Choose a batch:
                X_batch = X_[batch:min(batch + batch_size, nb_samples)]
                cur_batch_size = X_batch.shape[0]

                # Step 0:
                v0 = X_batch
                p_h0 = sigmoid(v0 @ self.W + self.b)
                h0 = (np.random.random([cur_batch_size, self.q]) < p_h0).astype(int)

                # Step 1:
                p_v1 = sigmoid(h0 @ self.W.T + self.a)
                v1 = (np.random.random([cur_batch_size, self.p]) < p_v1).astype(int)
                p_h1 = sigmoid(v1 @ self.W + self.b)

                # Calculate derivatives:
                da = (v0 - v1).sum(axis=0)
                db = (p_h0 - p_h1).sum(axis=0)
                dW = X_batch.T @ p_h0 - v1.T @ p_h1

                # Update weights:
                self.a += lr * da / cur_batch_size
                self.b += lr * db / cur_batch_size
                self.W += lr * dW / cur_batch_size

    def generate_images(self, iter_gibbs, nb_images, im_size=(20, 16)):
        v = (np.random.random([nb_images, self.p]) < .5).astype(int)

        for i in range(iter_gibbs):
            h = (np.random.random(self.q) < sigmoid(v @ self.W + self.b)).astype(int)
            v = (np.random.random(self.p) < sigmoid(h @ self.W.T + self.a)).astype(int)

        v = np.reshape(v, (nb_images, im_size[0], im_size[1]))
        return v
