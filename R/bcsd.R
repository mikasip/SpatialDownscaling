#' Bias Correction Spatial Disaggregation (BCSD) for statistical downscaling
#'
#' @description
#' Implements the BCSD method for statistical downscaling of climate data.
#' The approach consists of two main steps: (1) bias correction using quantile mapping
#' and (2) spatial disaggregation using interpolation.
#'
#' @param coarse_data A 3D array of coarse resolution input data. The two first dimensions should be spatial coordinates 
#' (e.g., latitude and longitude) in grid, and the third should be the training samples (e.g. time).
#' @param fine_data A 3D array of fine resolution output data. The two first dimensions should be spatial coordinates 
#' (e.g., latitude and longitude) in grid, and the third should be the training samples (e.g. time).
#' @param method Character. Interpolation method ('bilinear', 'bicubic', or 'nearest'). Default: 'bilinear'.
#' @param n_quantiles Integer. Number of quantiles for bias correction. Default: 100.
#' @param reference_period Vector. Start and end indices or dates for reference period. Default: NULL (use all data).
#' @param extrapolate Logical. Whether to extrapolate corrections outside calibration range. Default: TRUE.
#' @param normalize Logical. Whether to normalize data before processing. Default: TRUE.
#' 
#' @details 
#' The BCSD method is a statistical downscaling technique that combines bias correction
#' and spatial disaggregation. It uses quantile mapping to correct biases in coarse resolution data
#' and then applies spatial interpolation to disaggregate the data to a finer resolution.
#' 
#' The function allows for different interpolation methods and the option to normalize data before processing.
#' The quantile mapping step involves calculating quantiles from the coarse data and mapping them to the fine data.
#' The interpolation step uses the specified method to create a fine resolution grid from the coarse data.
#' 
#' @return A list containing the trained model components:
#'   \item{quantile_map}{Quantile mapping function for bias correction.}
#'   \item{interpolation_params}{Parameters for spatial interpolation.}
#'   \item{axis_names}{Names of the axes in the fine data.}
#'   \item{scalers}{List of scalers. If `normalize = TRUE`, 
#'    the list contains scalers `coarse` and `fine` 
#'    for the coarse and fine data, respectively.
#'    If `normalize = FALSE`, the list is empty.}
#'   \item{model_params}{List of all model parameters.}
#'
#' @examples
#' # Simple example with random data
#' coarse <- array(rnorm(8 * 8 * 10), dim = c(8, 8, 10))  # e.g. lat x lon x time 
#' fine <- array(rnorm(16 * 16 * 10), dim = c(16, 16, 10))    # e.g. lat x lon x time
#' model <- bcsd(coarse, fine, method = "bilinear", n_quantiles = 100)
#' coarse_new <- array(rnorm(8 * 8 * 3), dim = c(8, 8, 3))  # e.g. lat x lon x time 
#' predictions <- predict(model, coarse_new)
#'
#' @export
bcsd <- function(coarse_data, fine_data, method = "bilinear", n_quantiles = 100,
                reference_period = NULL, extrapolate = TRUE, normalize = TRUE) {
  # Store model parameters
  model_params <- list(
    method = method,
    n_quantiles = n_quantiles,
    reference_period = reference_period,
    extrapolate = extrapolate,
    normalize = normalize
  )
  
  # Initialize containers
  scalers <- list()
  
  if (normalize) {    
    # For coarse data
    coarse_mean <- mean(coarse_data, na.rm = TRUE)
    coarse_sd <- stats::sd(coarse_data, na.rm = TRUE)
    coarse_data_norm <- (coarse_data - coarse_mean) / coarse_sd
    
    # For fine data
    fine_mean <- mean(fine_data, na.rm = TRUE)
    fine_sd <- stats::sd(fine_data, na.rm = TRUE)
    fine_data_norm <- (fine_data - fine_mean) / fine_sd
    
    # Store scalers
    scalers$coarse <- list(mean = coarse_mean, sd = coarse_sd)
    scalers$fine <- list(mean = fine_mean, sd = fine_sd)
    
    # Use normalized data for further processing
    coarse_processed <- coarse_data_norm
    fine_processed <- fine_data_norm
  } else {
    coarse_processed <- coarse_data
    fine_processed <- fine_data
  }
  
  # Apply reference period restriction if provided
  if (!is.null(reference_period)) {
    coarse_ref <- coarse_processed[, , reference_period[1]:reference_period[2], drop = FALSE]
    fine_ref <- fine_processed[, , reference_period[1]:reference_period[2], drop = FALSE]
  } else {
    coarse_ref <- coarse_processed
    fine_ref <- fine_processed
  }
  
  # Calculate quantiles
  quantile_levels <- seq(0, 1, length.out = n_quantiles)
  coarse_quantiles <- stats::quantile(coarse_ref, probs = quantile_levels, na.rm = TRUE)
  fine_quantiles <- stats::quantile(fine_ref, probs = quantile_levels, na.rm = TRUE)
  
  # Create quantile mapping function
  quantile_map <- function(x) {
    # Find where the values fall in the coarse quantiles
    indices <- findInterval(x, coarse_quantiles)
    
    # Handle out-of-bounds values
    indices[indices < 1] <- 1
    indices[indices >= n_quantiles] <- n_quantiles - 1
    
    # Linear interpolation between quantiles
    alpha <- (x - coarse_quantiles[indices]) / 
             (coarse_quantiles[indices + 1] - coarse_quantiles[indices])
    alpha[is.na(alpha)] <- 0.5  # Handle possible division by zero
    
    # Interpolate in fine quantiles
    corrected <- fine_quantiles[indices] + 
                 alpha * (fine_quantiles[indices + 1] - fine_quantiles[indices])
    
    # Extrapolate if requested
    if (extrapolate) {
      # Handle NAs in the logical vectors
      # Below minimum - remove NAs from the logical vector
      below_min <- x < min(coarse_quantiles, na.rm = TRUE)
      below_min[is.na(below_min)] <- FALSE  # Set NA values to FALSE
      
      if (any(below_min)) {  # No need for na.rm here as we removed NAs
        delta <- min(fine_quantiles) - min(coarse_quantiles)
        corrected[below_min] <- x[below_min] + delta
      }
      
      # Above maximum - remove NAs from the logical vector
      above_max <- x > max(coarse_quantiles, na.rm = TRUE)
      above_max[is.na(above_max)] <- FALSE  # Set NA values to FALSE
      
      if (any(above_max)) {  # No need for na.rm here as we removed NAs
        delta <- max(fine_quantiles) - max(coarse_quantiles)
        corrected[above_max] <- x[above_max] + delta
      }
    }
    
    return(corrected)
  }
  
  # Get dimensions
  coarse_dims <- dim(coarse_data)
  fine_dims <- dim(fine_data)
  
  # Create interpolation parameters
  interpolation_params <- list(
    coarse_dims = coarse_dims[-3],  # Spatial dimensions
    fine_dims = fine_dims[-3],      # Spatial dimensions
    method = method
  )
  axis_names <- list(
    longitude = names(fine_data[, 1, 1]),
    latitude = names(fine_data[1, , 1]),
    time = names(fine_data[1, 1, ])
  )
  
  model <- list(
    quantile_map = quantile_map,
    interpolation_params = interpolation_params,
    axis_names = axis_names,
    scalers = scalers,
    model_params = model_params
  )
  class(model) <- "BCSD"
  
  return(model)
}

#' Predict method for BCSD model
#' 
#' @description
#' Generates predictions using the trained BCSD model.
#' 
#' @param object A BCSD model object.
#' @param newdata Matrix, array or raster. The new coarse resolution data to be downscaled.
#' @param ... Additional arguments (not used).
#' 
#' @details
#' The predict method applies the trained BCSD model to new coarse resolution data.
#' It performs bias correction using the quantile mapping function and then applies spatial interpolation
#' to generate fine resolution predictions.
#' 
#' @return A matrix, array or raster of the downscaled predictions at fine resolution.
#' 
#' @examples
#' # Simple example with random data
#' coarse <- array(rnorm(10*20*30), dim = c(10, 20, 30))  # time x lat x lon
#' fine <- array(rnorm(10*40*60), dim = c(10, 40, 60))    # time x lat x lon
#' model <- bcsd(coarse, fine, method = "bilinear", n_quantiles = 100)
#' # New coarse data for prediction
#' new_coarse <- array(rnorm(5*20*30), dim = c(5, 20, 30))  # time x lat x lon
#' predictions <- predict(model, new_coarse)
#' # Check dimensions of predictions
#' dim(predictions)  # Should be (5, 40, 60) for time x lat x lon
#' @seealso \code{\link{bcsd}} for training the model.
#' 
#' @export
predict.BCSD <- function(object, newdata, ...) {
  # Extract components from the model
  quantile_map <- object$quantile_map
  scalers <- object$scalers
  interpolation_params <- object$interpolation_params
  model_params <- object$model_params
  normalize <- model_params$normalize
  method <- interpolation_params$method

  # Apply normalization if used in training
  if (normalize) {
    new_coarse_norm <- (newdata - scalers$coarse$mean) / scalers$coarse$sd
  } else {
    new_coarse_norm <- newdata
  }
  
  # Apply bias correction
  bias_corrected <- apply(new_coarse_norm, 3, quantile_map, simplify = FALSE)
  bias_corrected <- abind::abind(bias_corrected, along = 3)

  # Apply interpolation
  time_steps <- dim(bias_corrected)[3]
  downscaled <- array(NA, dim = c(interpolation_params$fine_dims, time_steps))
  
  for (t in 1:time_steps) {
    # Convert slice to raster for interpolation
    slice_raster <- raster::raster(bias_corrected[, , t])
    
    target_raster <- raster::raster(nrows = interpolation_params$fine_dims[1], 
                              ncols = interpolation_params$fine_dims[2])
    raster::extent(target_raster) <- raster::extent(slice_raster)  # Keep the same geographic extent
    raster::crs(target_raster) <- raster::crs(slice_raster)       # Keep the same projection

    # Then resample
    fine_raster <- raster::resample(slice_raster, target_raster,
                                  method = method)
    downscaled[, , t] <- raster::as.matrix(fine_raster)
  }
  # Denormalize if needed
  if (normalize) {
    downscaled <- downscaled * scalers$fine$sd + scalers$fine$mean
  }
  # Apply correct axis names
  dimnames(downscaled) <- list(
    object$axis_names$longitude,
    object$axis_names$latitude,
    names(newdata[1, 1, ])
  )

  return(downscaled)
}
