import torch

# Check if Intel XPU is available
if torch.cuda.is_available():
    print("CUDA is available!")
    # Simple tensor operation
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')

    # Convert to CPU for printing
    print("Tensor on Intel XPU:", x.to('cpu'))

else:
    print("CUDA is not available.")
    print(torch.__version__)

