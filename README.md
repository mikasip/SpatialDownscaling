# SpatialDownscaling

## Overview

`SpatialDownscaling` provides deep learning methods for spatial downscaling of gridded data, particularly focused on meteorological and climate variables. The package implements time-aware UNet and super-resolution deep residual networks (SRDRN) for enhancing the spatial resolution of coarse-grid data, along with a statistical baseline method (BCSD).

The package includes the `weather_italy` dataset containing daily gridded weather variables (relative humidity, temperature, and total precipitation) for Italy from November 1 to December 31, 2023, obtained from the ERA5-Land dataset.

The methods are described in detail in:

> Sipilä, M., Maggio, S., De Iaco, S., Nordhausen, K., Palma, M., & Taskinen, S. (2025). Time-aware UNet and super-resolution deep residual networks for spatial downscaling. *arXiv preprint arXiv:2512.13753*. https://arxiv.org/abs/2512.13753

## Installation

You can install the released version of SpatialDownscaling from [CRAN](https://CRAN.R-project.org) with:
```r
install.packages("SpatialDownscaling")
```

Or install the development version from GitHub:
```r
# install.packages("devtools")
devtools::install_github("mikasip/SpatialDownscaling")
```

## Methods

### Time-aware UNet

The UNet architecture is based on Ronneberger et al. (2015) and is extended by incorporating a lightweight temporal module. The temporal module takes the time point of the observation as input, uses radial basis function or sinusoidal positional encoding to encode it to a more representative format, then uses a small feed-forward network followed by a small stack of convolutional layers to encode temporal information into spatial format. The temporal information is concatenated with spatial features obtained by the main UNet at the bottleneck of the network. For more details, see Sipilä et al. (2025).

### Super-Resolution Deep Residual Networks (SRDRN)

The SRDRN architecture is based on Sha et al. (2020) and is extended by incorporating a lightweight temporal module. The temporal module takes the time point of the observation as input, uses radial basis function or sinusoidal positional encoding to encode it to a more representative format, then uses a small feed-forward network followed by a small stack of convolutional layers to encode temporal information into spatial format. The temporal information is concatenated with spatial features obtained by the main SRDRN before the upscaling layers. For more details, see Sipilä et al. (2025).

### Bias Correction and Spatial Disaggregation (BCSD)

BCSD is a statistical baseline method based on Wood et al. (2004) that combines bias correction with spatial disaggregation for downscaling climate model outputs.

## Usage

### Basic Example
```r
library(SpatialDownscaling)

# Load the included weather data for Italy
data(weather_italy)

# Select temperature as variable of interest (index 2)
fine_data <- weather_italy[, , 2, ]

# Create coarse data by selecting every second pixel
coarse_data <- fine_data[seq(1, 60, 2), seq(1, 60, 2), ]

# Split data into training and testing sets
train_times <- 1:50
test_times <- 51:61
train_fine_data <- fine_data[ , , train_times]
train_coarse_data <- coarse_data[ , , train_times]
test_fine_data <- fine_data[ , , test_times]
test_coarse_data <- coarse_data[ , , test_times]


# Train BCSD model
bcsd_model_temp <- bcsd(train_coarse_data,
  train_fine_data)

# Make predictions with BCSD model
bcsd_preds <- predict(bcsd_model_temp,
  test_coarse_data)
# Evaluate BCSD model performance
bcsd_mae <- mean(abs(bcsd_preds - test_fine_data), na.rm = TRUE)
print(paste("BCSD MAE:", round(bcsd_mae, 4)))

# Train time-aware UNet model
# Increase the number of initial_filters, filters and epochs
# for better performance
model_unet <- unet(
   train_coarse_data,
   train_fine_data,
   time_points = train_times,
   filters = c(16, 32),
   initial_filters = c(8),
   epochs = 100,
   batch_size = 16
)

# Make predictions with UNet model
unet_preds <- predict(model_unet,
  test_coarse_data,
  time_points = test_times)
# Evaluate UNet model performance
unet_mae <- mean(abs(unet_preds - test_fine_data), na.rm = TRUE)
print(paste("UNet MAE:", round(unet_mae, 4)))

# Train time-aware SRDRN model
# Increase the number of epochs and num_res_block_filters (to e.g. 64) 
# for better performance
 model_srdrn <- srdrn(
   train_coarse_data,
   train_fine_data,
   num_res_block_filters = 32,
   time_points = train_times,
   epochs = 100,
   batch_size = 32
)
# Make predictions with SRDRN model
srdrn_preds <- predict(model_srdrn,
  test_coarse_data,
  time_points = test_times)

# Evaluate SRDRN model performance
srdrn_mae <- mean(abs(srdrn_preds - test_fine_data), na.rm = TRUE)
print(paste("SRDRN MAE:", round(srdrn_mae, 4)))
```

## Dependencies

The package requires:
- R (≥ 4.4.0)
- tensorflow
- keras3
- Python (for TensorFlow backend)

**Note**: This package requires a working Python installation with TensorFlow. Please ensure Python and TensorFlow are properly installed before using this package.

## Citation

If you use this package in your research, please cite:
```bibtex
@misc{sipila2025timeawareunetsuperresolutiondeep,
  title={Time-aware UNet and super-resolution deep residual networks for spatial downscaling}, 
  author={Sipilä, Mika and Maggio, Sabrina and De Iaco, Sandra and Nordhausen, Klaus and Palma, Monica and Taskinen, Sara},
  year={2025},
  eprint={2512.13753},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2512.13753}
}
```

## References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. *Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015*, 234-241. https://doi.org/10.1007/978-3-319-24574-4_28

- Sha, Y., Gagne II, D. J., West, G., & Stull, R. (2020). Deep-learning-based gridded downscaling of surface meteorological variables in complex terrain. Part II: Daily precipitation. *Journal of Applied Meteorology and Climatology*, 59(12), 2075-2092.

- Wood, A. W., Leung, L. R., Sridhar, V., & Lettenmaier, D. P. (2004). Hydrologic implications of dynamical and statistical approaches to downscaling climate model outputs. *Climatic Change*, 62, 189-216. https://doi.org/10.1023/B:CLIM.0000013685.99609.9e

## License

This package is released under the GPL-3 license.