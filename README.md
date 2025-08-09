# DIM-for-SVCT-recon



We have currently uploaded the code based on AAPM simulated dataset, in which the sparse sinogram has a resolution of 32 × 256, the full sinogram has a resolution of 512 × 256, and the corresponding CT image has a resolution of 256 × 256.

**run_32_resolution.py** is the most minimal version that implements all functionalities. It contains many comments aimed at helping you understand the process at each level of DIM. It is designed to extract an aliasing-free low-frequency CT image (with a resolution of 16×16, corresponding to a sinogram resolution of 32×16), upsample it to 32×32, and simultaneously inject information from the decomposed sinogram (with a resolution of 32×32).

**run.py** is a more comprehensive version that can train and validate all resolution levels of DIM. Please make sure you understand the flow in **run_32_resolution.py** before running it.
