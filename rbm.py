import numpy as np
from IPython.display import clear_output


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


PROGRESS_BAR_LENGTH = 20


class RBM:
    def __init__(self, p, q, sigma2=.01):
        self.p = p
        self.q = q

        self.W = sigma2 * np.random.randn(self.p, self.q)
        self.a = np.zeros([1, self.p])
        self.b = np.zeros([1, self.q])

    def train(self, X, lr, epochs, batch_size, verbose=0):
        nb_samples = X.shape[0]
        X_ = X.copy()

        # List of epochs' statistics to output:
        epoch_progress = []
        # epoch_progress = f""

        for epoch in range(1, epochs + 1):
            # Initialize the current epoch statistics:
            epoch_progress.append("")
            # epoch_progress += f"Epoch {epoch}:\n"

            epoch_mse = 0

            # Shuffle data:
            np.random.shuffle(X_)
            
            for batch in range(0, nb_samples, batch_size):
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

                # Compute the MSE:
                mse = ((v1 - X_batch) ** 2).mean()
                epoch_mse += mse

                # Compute the current batch and current epoch statistics:
                batch_number = batch // batch_size + 1
                samples_treated = (batch + batch_size) // nb_samples
                current_progress = int(PROGRESS_BAR_LENGTH * samples_treated)
                progress_bar = "[" + "=" * current_progress + ">" + "-" * (PROGRESS_BAR_LENGTH - current_progress) + "]"
                
                epoch_progress[-1] = f"Epoch {epoch}:\n" + \
                    f"Batch {batch_number}: {progress_bar}. Batch MSE: {mse:.5f}\n"
                # new_line = f"Batch {batch_number}: {progress_bar}. Batch MSE: {mse:.5f}\n"
                # epoch_progress = epoch_progress[:-(len(new_line) + len(str(batch_number)) - len(str(batch_number - 1)))] \
                #                     + new_line

                # Output the current batch and current epoch statistics:
                if verbose:
                    clear_output(wait=True)
                    print("\n\n".join(epoch_progress), flush=True)
                    # print(epoch_progress, flush=True)

                # Calculate derivatives:
                da = (v0 - v1).sum(axis=0)
                db = (p_h0 - p_h1).sum(axis=0)
                dW = X_batch.T @ p_h0 - v1.T @ p_h1

                # Update weights:
                self.a += lr * da / cur_batch_size
                self.b += lr * db / cur_batch_size
                self.W += lr * dW / cur_batch_size
            
            # Compute the current epoch average MSE:
            epoch_progress[-1] += f"Epoch average MSE: {epoch_mse / batch_number:.5f}"
            # epoch_progress += f"Epoch average MSE: {epoch_mse / batch_number:.5f}\n\n"
        
        # The last output:
        clear_output()
        print("\n\n".join(epoch_progress))
        # print(epoch_progress)

    def generate_images(self, iter_gibbs, nb_images, im_size=(20, 16)):
        v = (np.random.random([nb_images, self.p]) < .5).astype(int)

        for i in range(iter_gibbs):
            h = (np.random.random(self.q) < sigmoid(v @ self.W + self.b)).astype(int)
            v = (np.random.random(self.p) < sigmoid(h @ self.W.T + self.a)).astype(int)

        v = np.reshape(v, (nb_images, im_size[0], im_size[1]))
        return v
