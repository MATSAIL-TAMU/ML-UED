# Generate synthetic UED data
Python scripts to generate synthetic data for Machine Learning for Ultrafast Electron Diffraction

The repo contains two scripts `generate_UED_images_defects_deformation.py` and `generate_UED_images_phonons.py` to generate synthetic data for both CNNSyn and C-VAE models.

Usage of `generate_UED_images_defects_deformation.py` script

`python generate_UED_images_defects_deformation.py --uc unit_cells/CONTCAR_1T --num 8 --save-folder images/DIFF --suffix 1T_DEFORM`

The following options can be used to generate images of different classes
1. `--ucref unit_cells/CONTCAR_2H` creates a 2-channel image with positive (red) and negative (blue) deviations from the reference 2H structure for CNNSyn training. In the absence of this flag, diffraction images with raw intensities are generated for C-VAE training
2. `--add-defects` adds up to 20% vacancies at random Mo and Te lattice sites
3. `--add-distortion` adds random lattice distortion to the crystal structures
4. `--add-degradation` creates a degradated/molten structure with over 90% of the lattice sites missing from the crystal structure



Usage of `generate_UED_images_phonons.py` script

`python generate_UED_images_phonons.py --phonon-path phonons_2H --num 8 --save-folder images/DIFF --suffix 2H_PHONONS`

The following options can be used to generate images of different classes
1. `--ucref unit_cells/CONTCAR_2H` creates a 2-channel image with positive (red) and negative (blue) deviations from the reference 2H structure for CNNSyn training. In the absence of this flag, diffraction images with raw intensities are generated for C-VAE training
2. `--phonon-path phonons_1T` creates 1T' phonon crystals instead of 2H crystals with phonons

