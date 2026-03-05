# ---------------------- Imports ---------------------- #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Ingestor import Ingestor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os


# ---------------------- Optimizer ---------------------- #
class AdamOptimizer:
    def __init__(self, lr=0.001, beta_1=0.95, beta_2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, param, grad, key):
        if key not in self.m:
            self.m[key] = np.zeros_like(grad)
            self.v[key] = np.zeros_like(grad)
        self.t += 1
        self.m[key] = self.beta_1 * self.m[key] + (1 - self.beta_1) * grad
        self.v[key] = self.beta_2 * self.v[key] + (1 - self.beta_2) * (grad ** 2)
        m_hat = self.m[key] / (1 - self.beta_1 ** self.t)
        v_hat = self.v[key] / (1 - self.beta_2 ** self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return param


# ---------------------- Reproducibility ---------------------- #
np.random.seed(40)  # 5,40
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ---------------------- Load and Preprocess Data ---------------------- #
amnistadRelease = 'DataSetExport-Discharge Total.Last-24-Hour-Change-in-Storage@08450800-Instantaneous-TCM-20240622194957.csv'
data = Ingestor(amnistadRelease).data

df = pd.DataFrame(data)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df.set_index('Timestamp', inplace=True)
df['Value'] = pd.to_numeric(df['Value'], errors='coerce').ffill()

# Additional time-based features
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['day'] = df.index.day

# Scaling features
scaler_value = MinMaxScaler()
scaler_time = MinMaxScaler()

features = df[['Value', 'hour', 'dayofweek', 'month', 'day']].copy()
scaled_values = scaler_value.fit_transform(features[['Value']])
scaled_time = scaler_time.fit_transform(features.drop(columns='Value'))

combined_scaled = np.hstack([scaled_values, scaled_time])


# ---------------------- Sequence Generation ---------------------- #
def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(seq_length, len(data)):
        sequences.append(data[i - seq_length:i])
        targets.append(data[i, 0])
    return np.array(sequences), np.array(targets).reshape(-1, 1)


seq_length = 10
X, y = create_sequences(combined_scaled, seq_length)

train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]


# ---------------------- NumPy LSTM Model with Full Backpropagation ---------------------- #
class NumPyLSTMFullBP:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.001, loss_fn='rmse', clip_norm=5.0, dropout_rate=0.2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.loss_fn = loss_fn
        self.clip_norm = clip_norm  # maximum allowed norm for gradients
        self.dropout_rate = dropout_rate  # Dropout rate
        self.optimizer = AdamOptimizer(lr=learning_rate)

        # Xavier initialization helper
        def xavier_init(shape):
            return np.random.randn(*shape) * np.sqrt(1. / shape[1])

        # LSTM parameters for each gate: f, i, c, o.
        self.W = {gate: xavier_init((hidden_dim, hidden_dim + input_dim + 1)) for gate in ['f', 'i', 'c', 'o']}
        self.b = {gate: np.zeros((hidden_dim, 1)) for gate in ['f', 'i', 'c', 'o']}

        # Output layer parameters.
        self.W['y'] = xavier_init((output_dim, hidden_dim))
        self.b['y'] = np.zeros((output_dim, 1))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def tanh(x):
        return np.tanh(np.clip(x, -500, 500))

    def dropout(self, x):
        """Apply dropout during training."""
        if self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape)
            return x * mask / (1 - self.dropout_rate)  # Scale by (1 - dropout_rate) to maintain expected value
        return x
    

    def adaptive_thresholding(self, c_t, c_prev, deviation_threshold, optimizer: AdamOptimizer, key: str, alpha=0.95, temperature=1.0):
        """
        Optimized adaptive thresholding with dynamic alpha adjustment based on gradient size.
        """
        # Calculate the gradient (change in cell state)
        grad_c_t = np.abs(c_t - c_prev)

        # Compute smoothed state using the optimizer
        smoothed_state = optimizer.update(c_prev, grad_c_t, key)

        # Precompute standard deviation and deviation
        std_c_t = np.std(c_t) + 1e-6  # Avoid division by zero
        deviation = np.abs(c_t - smoothed_state)

        # Compute adaptive thresholds
        grad_clip = np.clip(grad_c_t / std_c_t, -10, 10)
        adaptive_threshold = deviation_threshold * std_c_t * (1 + np.tanh(grad_clip))
        scaled_threshold = adaptive_threshold * temperature

        # Vectorized replacement logic
        replace_mask = deviation >= scaled_threshold
        replacement = np.random.uniform(-scaled_threshold[replace_mask], scaled_threshold[replace_mask])

        # Dynamically adjust alpha based on gradient size
        adjusted_alpha = alpha * (1 - np.tanh(np.mean(grad_c_t) / std_c_t))  # Use tanh for smoother scaling

        # Update c_t based on adaptive thresholding
        c_t[replace_mask] = adjusted_alpha * c_t[replace_mask] + (1 - adjusted_alpha) * replacement

        return c_t


    def forward(self, x, cache_enabled=True, deviation_threshold=2.0, temperature=1.0):
        T, _ = x.shape
        h_t = np.zeros((self.hidden_dim, 1))  # Initialize hidden state
        c_t = np.zeros((self.hidden_dim, 1))  # Initialize cell state
        cache = []  # to store values for backpropagation if needed

        for t in range(T):
            x_t = x[t].reshape(-1, 1)  # Current input at time t
            combined = np.vstack((h_t, np.ones((1, 1)), x_t))  # Combine previous hidden state with input

            # Compute gate activations
            f_t = self.sigmoid(np.dot(self.W['f'], combined) + self.b['f'])
            i_t = self.sigmoid(np.dot(self.W['i'], combined) + self.b['i'])
            o_t = self.sigmoid(np.dot(self.W['o'], combined) + self.b['o'])
            c_candidate = self.tanh(np.dot(self.W['c'], combined) + self.b['c'])
            c_prev = c_t.copy()

            # Compute cell state update
            c_t = f_t * c_t + i_t * c_candidate

            # Apply adaptive thresholding to c_t, passing temperature to control thresholding
            c_t = self.adaptive_thresholding(c_t, c_prev, deviation_threshold, self.optimizer, f"time_{t}", 0.95, temperature)

            # Compute hidden state output
            h_t = o_t * self.tanh(c_t)

            # Apply dropout on h_t (hidden state) or gates (f, i, o, c_candidate)
            h_t = self.dropout(h_t)  # Dropout on hidden state
            f_t = self.dropout(f_t)  # Optionally apply dropout on gates as well
            i_t = self.dropout(i_t)
            o_t = self.dropout(o_t)
            c_candidate = self.dropout(c_candidate)

            if cache_enabled:
                # Store necessary variables in the cache for backpropagation
                cache.append({
                    'combined': combined,
                    'f': f_t,
                    'i': i_t,
                    'o': o_t,
                    'c_candidate': c_candidate,
                    'c_prev': c_prev,
                    'c': c_t.copy(),
                    'h': h_t.copy()
                })

        # Final output computation
        y_t = np.dot(self.W['y'], h_t) + self.b['y']
        
        if cache_enabled:
            return y_t.flatten(), h_t, cache
        else:
            return y_t.flatten(), h_t


    def compute_loss(self, pred, target):
        if self.loss_fn == 'mae':
            return np.mean(np.abs(pred - target))
        return np.mean((pred - target) ** 2)

    def _clip_gradients(self, grad):
        norm = np.linalg.norm(grad)
        if norm > self.clip_norm:
            grad = grad * (self.clip_norm / norm)
        return grad

    def backward(self, x, y_true, cache, y_pred, lam_bounds=0.0, y_min=0.0, y_max=1.0):
        # Compute derivative of loss with respect to output.
        # # For MSE loss: dL/dy = 2*(y_pred - y_true)
        # d_y = 2 * (y_pred.reshape(-1, 1) - y_true)
        
        # base data gradient (MSE)
        d_y = 2 * (y_pred.reshape(-1, 1) - y_true)

        # --- penalty gradient for bounds ---
        # penalty: mean( max(0,y_min-y)^2 + max(0,y-y_max)^2 )
        y = y_pred.reshape(-1, 1)

        dpen = np.zeros_like(y)

        # if y < y_min: penalty is (y_min - y)^2 -> d/dy = -2*(y_min - y) = 2*(y - y_min)
        mask_low = y < y_min
        dpen[mask_low] = 2.0 * (y[mask_low] - y_min)

        # if y > y_max: penalty is (y - y_max)^2 -> d/dy = 2*(y - y_max)
        mask_high = y > y_max
        dpen[mask_high] = 2.0 * (y[mask_high] - y_max)

        # combine
        d_y = d_y + lam_bounds * dpen
        
        # Gradients for output layer.
        last_cache = cache[-1]
        h_T = last_cache['h']
        dW_y = np.dot(d_y, h_T.T)
        db_y = d_y.copy()
        # Backpropagate into last hidden state.
        d_h = np.dot(self.W['y'].T, d_y)

        # Initialize gradient accumulators for LSTM parameters.
        grad_W = {gate: np.zeros_like(self.W[gate]) for gate in ['f', 'i', 'c', 'o']}
        grad_b = {gate: np.zeros_like(self.b[gate]) for gate in ['f', 'i', 'c', 'o']}

        # Initialize d_c (gradient w.r.t. cell state) as zero.
        d_c = np.zeros((self.hidden_dim, 1))
        T = len(cache)
        # Backpropagation through time (from last time step to first).
        for t in reversed(range(T)):
            cache_t = cache[t]
            combined = cache_t['combined']  # shape: (hidden_dim + 1 + input_dim, 1)
            f_t = cache_t['f']
            i_t = cache_t['i']
            o_t = cache_t['o']
            c_candidate = cache_t['c_candidate']
            c_t = cache_t['c']
            c_prev = cache_t['c_prev']
            h_t = cache_t['h']

            # h_t = o_t * tanh(c_t)
            d_o = d_h * self.tanh(c_t)
            d_o_input = d_o * (o_t * (1 - o_t))

            # Backprop through tanh: derivative tanh'(c_t) = 1 - tanh(c_t)^2.
            d_tanh_c = d_h * o_t * (1 - self.tanh(c_t) ** 2)
            # Total gradient for c_t (accumulate d_c from future time steps).
            d_c_total = d_tanh_c + d_c

            # c_t = f_t * c_prev + i_t * c_candidate.
            d_f = d_c_total * c_prev
            d_f_input = d_f * (f_t * (1 - f_t))
            d_i = d_c_total * c_candidate
            d_i_input = d_i * (i_t * (1 - i_t))
            d_c_candidate = d_c_total * i_t
            d_c_candidate_input = d_c_candidate * (1 - c_candidate ** 2)

            # Accumulate gradients for each gate's weights and biases.
            grad_W['f'] += np.dot(d_f_input, combined.T)
            grad_b['f'] += d_f_input
            grad_W['i'] += np.dot(d_i_input, combined.T)
            grad_b['i'] += d_i_input
            grad_W['o'] += np.dot(d_o_input, combined.T)
            grad_b['o'] += d_o_input
            grad_W['c'] += np.dot(d_c_candidate_input, combined.T)
            grad_b['c'] += d_c_candidate_input

            # Propagate gradient to combined input.
            # d_combined = sum_{gate} (W_gate^T * d_gate_input).
            d_combined = (np.dot(self.W['f'].T, d_f_input) +
                          np.dot(self.W['i'].T, d_i_input) +
                          np.dot(self.W['o'].T, d_o_input) +
                          np.dot(self.W['c'].T, d_c_candidate_input))

            # d_combined is of shape (hidden_dim + 1 + input_dim, 1).
            # We only propagate back into the h part (first hidden_dim rows).
            d_h = d_combined[:self.hidden_dim, :]
            # Also, propagate gradient through c to previous time step.
            d_c = d_c_total * f_t  # derivative of c_t = f_t * c_prev ... so d_c passes to c_prev.

        # Return gradients for all parameters.
        return grad_W, grad_b, dW_y, db_y

    def train(self, X_train, y_train, epochs=50, batch_size=32, initial_temperature=1.0, anneal_rate=0.99, lam_bounds=0.5, y_min=0.0, y_max=1.0): #added lam_bounds for penalty weight, and y_min, y_max for bounds in scaled space.
        print("\nTraining NumPy LSTM model with full backpropagation and mini-batch training...")
        n_samples = len(X_train)
        temperature = initial_temperature  # Initialize temperature
        
        for epoch in range(epochs):
            # Shuffle data at the start of each epoch.
            permutation = np.random.permutation(n_samples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]
            total_loss = 0.0
            batch_count = 0
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Accumulators for gradients over this batch.
                accum_grad_W = {gate: np.zeros_like(self.W[gate]) for gate in ['f', 'i', 'c', 'o']}
                accum_grad_b = {gate: np.zeros_like(self.b[gate]) for gate in ['f', 'i', 'c', 'o']}
                accum_dW_y = np.zeros_like(self.W['y'])
                accum_db_y = np.zeros_like(self.b['y'])
                batch_loss = 0.0

                for j in range(len(X_batch)):
                    x = X_batch[j]
                    y_true = y_batch[j].reshape(-1, 1)
                    # Forward pass with temperature control in adaptive thresholding.
                    y_pred, h, cache = self.forward(x, cache_enabled=True, temperature=temperature)
                    loss = self.compute_loss(y_pred, y_true)
                    
                    # --- Bound penalty on predicted release (scaled space) ---
                    # hinge violations
                    v_low  = np.maximum(0.0, y_min - y_pred)      # if y_pred < y_min
                    v_high = np.maximum(0.0, y_pred - y_max)      # if y_pred > y_max
                    pen_bounds = np.mean(v_low**2 + v_high**2)

                    loss_total = loss + lam_bounds * pen_bounds
                    batch_loss += loss_total
                    

                    grad_W, grad_b, dW_y, db_y = self.backward(x, y_true, cache, y_pred, lam_bounds=lam_bounds, y_min=y_min, y_max=y_max)

                    for gate in ['f', 'i', 'c', 'o']:
                        accum_grad_W[gate] += grad_W[gate]
                        accum_grad_b[gate] += grad_b[gate]
                    accum_dW_y += dW_y
                    accum_db_y += db_y

                batch_size_actual = len(X_batch)
                for gate in ['f', 'i', 'c', 'o']:
                    accum_grad_W[gate] /= batch_size_actual
                    accum_grad_b[gate] /= batch_size_actual
                    accum_grad_W[gate] = self._clip_gradients(accum_grad_W[gate])
                    accum_grad_b[gate] = self._clip_gradients(accum_grad_b[gate])
                accum_dW_y /= batch_size_actual
                accum_db_y /= batch_size_actual
                accum_dW_y = self._clip_gradients(accum_dW_y)
                accum_db_y = self._clip_gradients(accum_db_y)

                for gate in ['f', 'i', 'c', 'o']:
                    self.W[gate] = self.optimizer.update(self.W[gate], accum_grad_W[gate], f'W_{gate}')
                    self.b[gate] = self.optimizer.update(self.b[gate], accum_grad_b[gate], f'b_{gate}')
                self.W['y'] = self.optimizer.update(self.W['y'], accum_dW_y, 'W_y')
                self.b['y'] = self.optimizer.update(self.b['y'], accum_db_y, 'b_y')

                total_loss += batch_loss / batch_size_actual
                batch_count += 1

            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

            # Update temperature for simulated annealing.
            temperature *= anneal_rate  # Simulated annealing: decay temperature

# ---------------------- Train and Evaluate ---------------------- #
# Initialize the model with a dropout rate of 0.2 (20% dropout)
model = NumPyLSTMFullBP(input_dim=5, hidden_dim=128, output_dim=1, learning_rate=0.001, loss_fn='rmse', clip_norm=1.0, dropout_rate=0.001)

# Train the model
model.train(X_train, y_train, epochs=100, batch_size=32)


# Prediction (disable caching)
predictions = [model.forward(x, cache_enabled=False)[0] for x in X_test]
predictions = np.array(predictions).reshape(-1, 1)
predictions = scaler_value.inverse_transform(predictions)
y_test_inv = scaler_value.inverse_transform(y_test)

mse = mean_squared_error(y_test_inv, predictions)
mae = mean_absolute_error(y_test_inv, predictions)
print(f'MSE: {mse:.4f}, MAE: {mae:.4f}')

plt.figure(figsize=(14, 7))
plt.plot(df.index[train_size + seq_length:], y_test_inv, label='Actual', color='black')
plt.plot(df.index[train_size + seq_length:], predictions, label='Predicted', color='blue')
plt.xlabel('Timestamp')
plt.ylabel('Discharge Value')
plt.title('LSTM Prediction vs Actual Values')
plt.legend()
plt.grid(True)
plt.show()


