import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


# Define the Net class (same as in the training script)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16, 20)
        self.fc2 = nn.Linear(16, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x, y):
        h1 = F.relu(self.fc1(x) + self.fc2(y))
        h2 = self.fc3(h1)
        return h2


def gen_correlated_samples(batch_size=128, sub_batch=96, dim=16, flip_prob=0.2):
    # Generate x from Bernoulli(0.5)
    x = dist.Bernoulli(probs=0.5).sample((batch_size, sub_batch, dim))
    # Generate y with correlation based on flip_prob
    flip_mask = dist.Bernoulli(probs=flip_prob).sample(x.shape)
    y = x * (1 - flip_mask) + (1 - x) * flip_mask
    return x, y


def gen_independent_samples(batch_size=128, sub_batch=96, dim=32):
    # Generate independent 32-dim Bernoulli(0.5) vectors
    indep = dist.Bernoulli(probs=0.5).sample((batch_size, sub_batch, dim))
    # Permute dimensions
    perm_dim = torch.randperm(dim)
    indep_perm = indep[..., perm_dim]
    # Split into left and right 16 dims
    left = indep_perm[..., :16]
    right = indep_perm[..., 16:]
    return left, right


def compute_mi(model, left, right):
    # Flatten for shuffling (effective batch: batch_size * sub_batch)
    flat_num = left.shape[0] * left.shape[1]
    left_flat = left.view(flat_num, 16)
    right_flat = right.view(flat_num, 16)

    # Shuffle right along batch dimension (full shuffle)
    perm_shuffle = torch.randperm(flat_num)
    right_shuffle_flat = right_flat[perm_shuffle]

    with torch.no_grad():
        pred_xy = model(left_flat, right_flat)
        pred_x_y = model(left_flat, right_shuffle_flat)

        # Clamp to prevent overflow, as in training
        pred_x_y = torch.clamp(pred_x_y, min=-20, max=20)

        mi_value = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
    return mi_value.item()


if __name__ == '__main__':
    # Load the trained model
    model = Net()
    model.load_state_dict(torch.load('./mine_weights.pth'))
    model.eval()  # Set to evaluation mode

    # Parameters
    batch_size = 128
    sub_batch = 96
    effective_batch = batch_size * sub_batch  # 12288
    num_tests = 5  # Number of tests per scenario for averaging

    # Test scenarios for correlated structure (x and y as (batch_size, sub_batch, 16))
    print("=== Testing Correlated Structure (x and y generated with flip_prob) ===")
    for flip_prob, desc in [(0.0, "Strong correlation (flip_prob=0.0, y == x)"),
                            (0.2, "Training-like correlation (flip_prob=0.2)"),
                            (0.5, "Independent (flip_prob=0.5)")]:
        print(f"\n{desc}: Running {num_tests} tests and averaging")
        mi_values = []
        for test_idx in range(num_tests):
            x, y = gen_correlated_samples(batch_size, sub_batch, flip_prob=flip_prob)
            # Use y as "right", x as "left" for consistency with MI computation
            mi_estimate = compute_mi(model, x, y)
            mi_values.append(mi_estimate)
            print(f"Test {test_idx + 1}: Estimated MI = {mi_estimate:.4f}")

        avg_mi = sum(mi_values) / len(mi_values)
        print(f"Average MI: {avg_mi:.4f}")

    # Test scenarios for independent structure (32-dim vectors, permute dims, split left/right)
    print("\n=== Testing Independent Structure (32-dim Bernoulli, permute dims, split left/right) ===")
    print(f"Running {num_tests} tests (expected MI close to 0) and averaging")
    mi_values = []
    for test_idx in range(num_tests):
        left, right = gen_independent_samples(batch_size, sub_batch)
        mi_estimate = compute_mi(model, left, right)
        mi_values.append(mi_estimate)
        print(f"Test {test_idx + 1}: Estimated MI = {mi_estimate:.4f}")

    avg_mi = sum(mi_values) / len(mi_values)
    print(f"Average MI: {avg_mi:.4f}")
