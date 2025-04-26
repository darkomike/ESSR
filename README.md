# Instructions to test the  ESSR PyTorch Implementation

This document provides step-by-step instructions to the model in the PyTorch implementation of the **Enhanced Self-Supervised Super-Resolution (ESSR)** model.


## 1. Environment Setup
Ensure your system is ready to run the ESSR code.

### Prerequisites
- **Operating System**: Linux, macOS, or Windows.
- **Python**: Version 3.12 (recommended).
- **Hardware**: CPU (required), NVIDIA GPU (optional for faster training).

### Install Dependencies
1. Install Python 3.12 if not already installed:
   - On Linux:
     ```bash
     sudo apt update
     sudo apt install python3.12 python3.12-venv
     ```
   - On macOS (using Homebrew):
     ```bash
     brew install python@3.12
     ```
   - On Windows: Download and install from [python.org](https://www.python.org/downloads/).

2. Create a virtual environment:
   ```bash
   python3.12 -m venv essr_env
   source essr_env/bin/activate  # On Windows: essr_env\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install torch torchvision
   ```
   - This installs PyTorch and torchvision, which include all necessary dependencies for the ESSR model.
   - If you have a GPU, ensure CUDA is installed (PyTorch automatically selects the appropriate version).

## 2. Prepare the Code
Save the ESSR PyTorch implementation to a file.

1. Copy the code from the latest implementation (provided in the research paper or accompanying artifacts).
2. Save it as `essr.py` in a working directory (e.g., `~/essr_project`):
   ```bash
   mkdir ~/essr_project
   cd ~/essr_project
   # Paste the code into essr.py using a text editor (e.g., nano, vim, VSCode)
   ```

## 3. Run the Code
Execute the code to verify the fixes.

1. Run the script:
   ```bash
   python essr.py
   ```
   - The code uses a placeholder dataset (random tensors) and trains for 10 epochs.
   - It automatically detects the device (CPU or GPU) and handles VGG16 weights loading.

2. **With Internet Connectivity**:
   - The code will download VGG16 weights (`vgg16-397923af.pth`) to `~/.cache/torch/hub/checkpoints/` if not already cached.
   - Expected output:
     ```
     Using device: cpu
     Loaded VGG16 weights from online source
     Epoch 1, Loss: X.XXXX
     Epoch 2, Loss: X.XXXX
     ...
     ```
     - If a GPU is available:
       ```
       Using device: cuda
       Loaded VGG16 weights from online source
       Epoch 1, Loss: X.XXXX
       ...
       ```

3. **Without Internet Connectivity**:
   - If cached weights exist at `~/.cache/torch/hub/checkpoints/vgg16-397923af.pth`, the code will use them:
     ```
     Using device: cpu
     Loaded VGG16 weights from local cache: ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth
     Epoch 1, Loss: X.XXXX
     ...
     ```
   - If no cached weights are available, the code will run with randomly initialized VGG16 weights and display a warning:
     ```
     Using device: cpu
     UserWarning: No internet connectivity detected and no local weights found at ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth
     To resolve this, either:
     1. Connect to the internet and retry.
     2. Manually download 'vgg16-397923af.pth' from:
        https://download.pytorch.org/models/vgg16-397923af.pth
        and place it in ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth
     3. Check DNS settings (e.g., use Google's DNS: 8.8.8.8):
        sudo echo 'nameserver 8.8.8.8' >> /etc/resolv.conf
     Running with randomly initialized VGG16 weights, which may reduce performance.
     Epoch 1, Loss: X.XXXX
     ...
     ```

## 4. Handle Network Issues
If you encounter network-related errors (e.g., `Temporary failure in name resolution`), follow these steps:

### Manual Weight Download
1. Download the VGG16 weights manually:
   - URL: `https://download.pytorch.org/models/vgg16-397923af.pth`
   - Use a browser or command-line tool:
     ```bash
     wget https://download.pytorch.org/models/vgg16-397923af.pth -P ~/.cache/torch/hub/checkpoints
     ```
     - On Windows, use `curl` or download via a browser and move the file to `C:\Users\<YourUser>\.cache\torch\hub\checkpoints\`.

2. Create the cache directory if it doesn’t exist:
   ```bash
   mkdir -p ~/.cache/torch/hub/checkpoints
   ```

3. Verify the file is in place:
   ```bash
   ls ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth
   ```

### Troubleshoot DNS Issues
If the network error persists:
1. Test connectivity:
   ```bash
   ping 8.8.8.8
   ```
2. Set Google’s DNS:
   ```bash
   sudo echo "nameserver 8.8.8.8" >> /etc/resolv.conf
   ```
3. Retry running the code:
   ```bash
   python essr.py
   ```
4. If issues continue, use a system with internet access to download the weights, then copy them to the target system.

## 5. Test with Real Datasets
The placeholder dataset is for testing. To verify ESSR’s performance, use real super-resolution datasets like SET5 or DIV2K.

1. **Download a Dataset**:
   - **SET5**: Small dataset for quick testing.
     - Download from [official sources](http://vllab.ucmerced.edu/wlai24/LapSRN/) or repositories.
   - **DIV2K**: Standard SR dataset.
     - Download from [DIV2K official site](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
   - Save the dataset to a directory (e.g., `~/essr_project/DIV2K`).

2. **Modify the Code**:
   - Replace the placeholder dataset in `essr.py` with a real dataset loader. Example for DIV2K:
     ```python
     from torchvision import datasets, transforms
     from torch.utils.data import DataLoader

     # Define transforms
     transform = transforms.ToTensor()

     # Load DIV2K (adjust path to your dataset)
     dataset = datasets.ImageFolder('~/essr_project/DIV2K', transform=transform)
     train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

     # Rest of the code remains unchanged
     model = ESSR(upscale_factor=4)
     train_essr(model, train_loader, epochs=10)
     ```
   - Ensure the dataset is formatted as LR-HR image pairs. You may need a custom `Dataset` class for paired loading:
     ```python
     from torch.utils.data import Dataset
     from PIL import Image
     import os

     class SRDataset(Dataset):
         def __init__(self, lr_dir, hr_dir, transform=None):
             self.lr_dir = lr_dir
             self.hr_dir = hr_dir
             self.transform = transform
             self.images = [f for f in os.listdir(lr_dir) if f.endswith('.png')]

         def __len__(self):
             return len(self.images)

         def __getitem__(self, idx):
             lr_path = os.path.join(self.lr_dir, self.images[idx])
             hr_path = os.path.join(self.hr_dir, self.images[idx])
             lr_img = Image.open(lr_path).convert('RGB')
             hr_img = Image.open(hr_path).convert('RGB')
             if self.transform:
                 lr_img = self.transform(lr_img)
                 hr_img = self.transform(hr_img)
             return lr_img, hr_img

     # Example usage
     dataset = SRDataset('~/essr_project/DIV2K/LR', '~/essr_project/DIV2K/HR', transform=transform)
     train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
     ```

3. Run the modified code:
   ```bash
   python essr.py
   ```
   - Verify that the training loop completes without errors and produces reasonable loss values.

## 6. GPU Testing
If you have an NVIDIA GPU with CUDA installed:
1. Ensure PyTorch is installed with CUDA support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
   - Replace `cu118` with your CUDA version (check with `nvidia-smi`).
2. Run the code:
   ```bash
   python essr.py
   ```
3. Verify the output shows:
   ```
   Using device: cuda
   Loaded VGG16 weights from online source
   Epoch 1, Loss: X.XXXX
   ...
   ```
4. Confirm no CUDA-related errors occur, and training is faster than on CPU.

## 7. Troubleshooting Common Issues
- **Memory Errors**:
  - Reduce the batch size (e.g., `batch_size=8`) in the `DataLoader` if you encounter out-of-memory errors.
  - Example:
    ```python
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    ```
- **Dataset Loading Errors**:
  - Ensure the dataset paths are correct and images are in a supported format (e.g., PNG, JPEG).
  - Check for corrupted files using a script to validate images.
- **Persistent Network Issues**:
  - Test internet connectivity:
    ```bash
    ping 8.8.8.8
    ```
  - If unresolved, manually download the VGG16 weights as described in Section 4.
- **Unexpected Errors**:
  - Check the Python version and package versions:
    ```bash
    python --version
    pip list | grep torch
    ```
  - Ensure `torch` and `torchvision` are up-to-date.
  - Share the error traceback for further assistance.

## 8. Verification Checklist
To confirm all fixes are working:
- [ ] **CUDA Fix**: The code runs on CPU without CUDA errors (`Using device: cpu` in output).
- [ ] **VGG16 Weights Fix**:
  - Online: Weights download automatically with internet.
  - Offline: Code runs with cached weights or random initialization, showing a clear warning.
- [ ] **GradScaler Fix**: No `FutureWarning` about `torch.cuda.amp.GradScaler` in the output.
- [ ] **Training Completion**: The training loop completes 10 epochs with the placeholder dataset, printing loss values.
- [ ] **Real Dataset (Optional)**: Training with SET5 or DIV2K runs without errors (if implemented).

## 9. Additional Notes
- **Performance with Random VGG16 Weights**:
  - If VGG16 weights are unavailable, the code uses random weights, which may reduce the perceptual loss’s effectiveness. For optimal results, ensure weights are downloaded or cached.
- **Scaling Up**:
  - For production, use a GPU and real datasets (SET5, DIV2K) to evaluate ESSR’s performance.
  - Increase `epochs` (e.g., 200) for better convergence with real datasets.
- **Cloud Testing**:
  - If local issues persist, test on a cloud platform like Google Colab:
    1. Upload `essr.py` to Colab.
    2. Install dependencies:
       ```bash
       !pip install torch torchvision
       ```
    3. Run the script and verify the output.
- **Contact for Support**:
  - If you encounter issues or need help with dataset integration, share details (error messages, system specs) for tailored assistance.

By following these instructions, you should verify that the ESSR implementation runs without the previous runtime errors, on both CPU and GPU, with or without internet connectivity.

---