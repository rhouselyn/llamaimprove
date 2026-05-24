import torch
import os
import torch.serialization


# Define the PatchMLPEncoder class (same as before)
class PatchMLPEncoder(torch.nn.Module):
    def __init__(self, hidden_dim=24, learnable=True):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, hidden_dim, kernel_size=(4, 2), stride=(4, 2), bias=False)
        self.activation = torch.nn.ReLU()
        self.linear = torch.nn.Linear(hidden_dim, 1, bias=False)
        self.norm = torch.nn.LayerNorm(32)
        self.learnable = learnable
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = x.permute(0, 2, 3, 1)
        x = self.linear(x)
        x = x.squeeze(-1)
        x = x.flatten(1)
        x = self.norm(x)
        return x


def print_model_parameters(checkpoint_path):
    """
    Load the checkpoint and print the model parameters.

    Args:
        checkpoint_path (str): Path to the .pt checkpoint file.
    """
    # Check if the file exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' does not exist.")
        return

    # Allowlist argparse.Namespace for safe loading
    import argparse
    torch.serialization.add_safe_globals([argparse.Namespace])

    # Load the checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Extract the model state dictionary
    model_state = checkpoint.get('model')
    if model_state is None:
        print("Error: No 'model' key found in the checkpoint.")
        return

    # Initialize the model
    try:
        # Assuming hidden_dim is 24 as per your default argument
        model = PatchMLPEncoder(hidden_dim=24, learnable=True)
        model.load_state_dict(model_state)
        model.eval()
    except Exception as e:
        print(f"Error loading state_dict into model: {e}")
        return

    print(f"\n=== Parameters in {checkpoint_path} ===")
    print(f"Model: PatchMLPEncoder")
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nParameter details:")
    print("=" * 50)

    # Iterate through the state dictionary
    for name, param in model_state.items():
        print(f"\nParameter: {name}")
        print(f"Shape: {param.shape}")
        print(f"Requires Grad: {param.requires_grad}")
        # Print a sample of the parameter values (first few elements to avoid flooding output)
        if param.numel() > 10:
            print(f"Values (first 10 elements): {param.flatten()[:10].tolist()}")
        else:
            print(f"Values: {param.flatten().tolist()}")
        # Print statistics
        print(f"Mean: {param.mean().item():.6f}")
        print(f"Std: {param.std().item():.6f}")
        print(f"Min: {param.min().item():.6f}")
        print(f"Max: {param.max().item():.6f}")

    print("\n" + "=" * 50)
    print("Parameter printing completed.")


if __name__ == "__main__":
    checkpoint_path = "/mnt/afs/zhengmingkai/raozf/llamagen/results_encoder/encoder_training_95/encoder_best.pt"
    print_model_parameters(checkpoint_path)
