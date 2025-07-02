# DIM-for-SVCT-recon

The full code will be released before July 07, 2025.

Now, we have currently uploaded the code based on a simulated dataset, where the sparse sinogram has a resolution of 32×256, the full sinogram has a resolution of 512×256, and the corresponding CT image has a resolution of 256×256.

In this setting, the goal of run_32_resolution.py is to extract an aliasing-free low-frequency CT image (with a resolution of 16×16, corresponding to a sinogram resolution of 32×16), upsample it to 32×32, and simultaneously inject information from the decomposed sinogram (with a resolution of 32×32).
