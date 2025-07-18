import torch

# Check if Intel XPU is available
if torch.xpu.is_available():
    print("Intel XPU is available!")
    # Simple tensor operation
    x = torch.tensor([1.0, 2.0, 3.0], device='xpu')

    # Convert to CPU for printing
    print("Tensor on Intel XPU:", x.to('cpu'))

else:
    print("Intel XPU is not available.")
    print(torch.__version__)

