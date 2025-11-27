#' UNet model for spatial downscaling using deep learning
#'
#' @description
#' Implements a time-aware UNet convolutional neural network for spatial downscaling of grid data.
#' Time-aware UNet features an encoder-decoder architecture with skip connections and a temporal module.
#' The function allows an option for adding a temporal module for spatio-temporal applications.
#'
#' @param coarse_data 3D or 4D array. The coarse resolution input data in format 
#' `[x, y, variables, time]`, where the variables dimension is optional.
#' @param fine_data 3D or 4D array. The fine resolution target data in format 
#' `[x, y, variables, time]`, where the variables dimension is optional.
#' @param time_points Numeric vector. Optional time points corresponding to each time step in the data.
#' @param val_coarse_data An optional 3D or 4D array of coarse resolution input data in format 
#' `[x, y, variables, time]`, where the variables dimension is optional.
#' @param val_fine_data An optional 3D or 4D array of fine resolution target data in format 
#' `[x, y, variables, time]`, where the variables dimension is optional.
#' @param val_time_points An optional numeric vector of length n representing the time points of the validation samples.
#' @param cycle_onehot Boolean. If TRUE, a onehot encoded vector of temporal cycles is added as input to temporal module.
#' @param cyclical_period Numeric. Optional period for cyclical time encoding (e.g., 365 for yearly seasonality).
#' @param temporal_basis A numeric vector specifying the temporal basis functions to use for time encoding (default is c(9, 17, 37)).
#' @param temporal_layers A numeric vector specifying the number of units in each dense layer for time encoding (default is c(32, 64, 128)).
#' @param cos_sin_transform Logical. Whether to use cosine-sine transformation for time features. Default: FALSE.
#' @param initial_filters Integer vector. Number of filters in the initial convolutional layers. Default: c(16).
#' @param initial_kernel_sizes List of integer vectors. Kernel sizes for the initial convolutional layers. Default: list(c(3, 3)).
#' @param filters Integer vector. Number of filters in each convolutional layer. Default: c(32, 64, 128).
#' @param kernel_sizes List of integer vectors. Kernel sizes for each convolutional layer. Default: list(c(3, 3), c(3, 3), c(3, 3)).
#' @param use_batch_norm Logical. Whether to use batch normalization after convolutional layers. Default: FALSE.
#' @param dropout_rate Numeric. Dropout rate for regularization. Default: 0.2.
#' @param activation Character. Activation function for hidden layers. Default: "relu".
#' @param final_activation Character. Activation function for output layer. Default: "linear".
#' @param optimizer Character or optimizer object. Optimizer for training. Default: "adam".
#' @param learning_rate Numeric. Learning rate for optimizer. Default: 0.001.
#' @param loss Character or loss function. Loss function for training. Default: "mse".
#' @param metrics Optional character vector. Metrics to track during training.
#' @param batch_size Integer. Batch size for training. Default: 32.
#' @param epochs Integer. Number of training epochs. Default: 100.
#' @param validation_split Numeric. Fraction of data to use for validation. Default: 0.2.
#' @param normalize Logical. Whether to normalize data before training. Default: TRUE.
#' @param callbacks List. Keras callbacks for training. Default: NULL.
#' @param seed Integer. Random seed for reproducibility. Default: NULL.
#' @param verbose Integer. Verbosity mode (0, 1, or 2). Default: 1.
#'
#' @return List containing the trained model and associated components:
#'   \item{model}{Trained Keras model}
#'   \item{input_mask}{Mask for input data based on the missing values}
#'   \item{target_mask}{Mask for target data based on the missing values}
#'   \item{min_time_point}{Minimum time point in the training data}
#'   \item{max_time_point}{Maximum time point in the training data}
#'   \item{cyclical_period}{Cyclical period for time encoding}
#'   \item{max_season}{Maximum season for time encoding}
#'   \item{axis_names}{Names of the axes in the input data}
#'   \item{history}{Training history}
#'
#' @details
#' The UNet architecture \insertCite{ronneberger2015u}{SpatialDownscaling} is widely used in image processing 
#' tasks and has recently been adopted for spatial downscaling applications 
#' \insertCite{sha2020deep}{SpatialDownscaling}. The method implemented here consists of:
#' 
#' 1. **Initial Upscaling** – Coarse-resolution inputs are first upsampled using 
#' bilinear interpolation to match the spatial dimensions of the fine-resolution target.
#'
#' 2. **Initial Feature Extraction** – Multiple convolutional layers extract 
#' low-level features before entering the encoder path.
#'
#' 3. **Encoder Path** – A sequence of convolutional blocks with max-pooling 
#' reduces spatial dimensions while increasing feature depth.
#'
#' 4. **Decoder Path** – Spatial resolution is recovered via bilinear upsampling 
#' and convolutional layers. Skip connections from the encoder help preserve 
#' fine-scale information.
#'
#' 5. **Skip Connections** – These link encoder and decoder layers at matching 
#' resolutions, improving gradient flow and retaining fine spatial structure.
#'
#' 6. **Temporal Module (optional)** – Time information can be incorporated 
#' through cosine–sine encoding, one-hot seasonal encoding, or radial-basis 
#' temporal features. These are passed through dense layers and reshaped to 
#' merge with the UNet bottleneck.
#'
#' The function supports missing data via masking, optional normalization, 
#' validation data, and configurable UNet depth and width.
#'
#' @examples
#' \dontrun{
#' library(keras3)
#'
#' # Create tiny dummy data:
#' # Coarse grid: 8x8 → Fine grid: 16x16
#' nx_c <- 8; ny_c <- 8
#' nx_f <- 16; ny_f <- 16
#' T <- 5  # number of time steps
#'
#' # Coarse data:
#' coarse_data <- array(runif(nx_c * ny_c * 1 * T),
#'                      dim = c(nx_c, ny_c, 1, T))
#'
#' # Fine data:
#' fine_data <- array(runif(nx_f * ny_f * 1 * T),
#'                    dim = c(nx_f, ny_f, 1, T))
#'
#' # Optional time points
#' time_points <- 1:T
#'
#' # Fit a tiny UNet (very small filters to keep the example fast)
#' model_obj <- unet_downscale(
#'   coarse_data,
#'   fine_data,
#'   time_points = time_points,
#'   filters = c(8, 16),
#'   initial_filters = c(4),
#'   epochs = 2,
#'   batch_size = 2,
#'   verbose = 0
#' )
#' }
#' 
#' @references 
#' \insertAllCited{}
#' 
#' @export
unet_downscale <- function(coarse_data, fine_data,
                          time_points = NULL,
                          val_coarse_data= NULL, 
                          val_fine_data = NULL, 
                          val_time_points = NULL,
                          cyclical_period = NULL,
                          cycle_onehot = FALSE,
                          cos_sin_transform = FALSE,
                          temporal_basis = c(9, 17, 37),
                          temporal_layers = c(32, 64, 128),
                          initial_filters = c(16),
                          initial_kernel_sizes = list(c(3, 3)),
                          filters = c(32, 64, 128),
                          kernel_sizes = list(c(3, 3), c(3, 3), c(3, 3)),
                          use_batch_norm = FALSE,
                          dropout_rate = 0.2,
                          activation = "relu",
                          final_activation = "linear",
                          optimizer = "adam",
                          learning_rate = 0.001,
                          loss = "mse",
                          metrics = c(),
                          batch_size = 32,
                          epochs = 100,
                          validation_split = 0.2,
                          normalize = TRUE,
                          callbacks = NULL,
                          seed = NULL,
                          verbose = 1) {

  # Import keras and tensorflow with explicit namespaces
  if (verbose > 0) cat("Setting up Keras and TensorFlow...\n")
  
  # Process input and output data to ensure correct format [x, y, variables, time]
  if (verbose > 0) cat("Checking and preprocessing data format...\n")

  axis_names <- dimnames(fine_data)
  # Check and adjust coarse_data format if needed
  coarse_dim <- dim(coarse_data)
  if (length(coarse_dim) == 3) {
    # Format is [x, y, time], add variables dimension
    if (verbose > 0) cat("Adding channel dimension to coarse_data...\n")
    coarse_data <- array(coarse_data, dim = c(coarse_dim[1], coarse_dim[2], 1, coarse_dim[3]))
    if (!is.null(val_coarse_data)) {
      val_coarse_dim <- dim(val_coarse_data)
      val_coarse_data <- array(val_coarse_data, dim = c(val_coarse_dim[1], val_coarse_dim[2], 1, val_coarse_dim[3]))
    }
  } else if (length(coarse_dim) != 4) {
    stop("coarse_data must be 3D [x, y, time] or 4D [x, y, variables, time]")
  }
  
  # Check and adjust fine_data format if needed
  fine_dim <- dim(fine_data)
  if (length(fine_dim) == 3) {
    # Format is [x, y, time], add variables dimension
    if (verbose > 0) cat("Adding channel dimension to fine_data...\n")
    fine_data <- array(fine_data, dim = c(fine_dim[1], fine_dim[2], 1, fine_dim[3]))
    if (!is.null(val_fine_data)) {
      val_fine_dim <- dim(val_fine_data)
      val_fine_data <- array(val_fine_data, dim = c(val_fine_dim[1], val_fine_dim[2], 1, val_fine_dim[3]))
    }
  } else if (length(fine_dim) != 4) {
    stop("fine_data must be 3D [x, y, time] or 4D [x, y, variables, time]")
  }
  
  # Get dimensions after any adjustments
  coarse_dim <- dim(coarse_data)
  fine_dim <- dim(fine_data)
  
  input_shape <- c(coarse_dim[1], coarse_dim[2], coarse_dim[3])
  
  output_shape <- c(fine_dim[1], fine_dim[2], fine_dim[3])

  print(paste0("Input shape ", input_shape))
  print(paste0("Input shape ", output_shape))
  
  # Verify that time dimensions match
  if (coarse_dim[4] != fine_dim[4]) {
    stop("Time dimensions of coarse_data and fine_data must match")
  }
  
  # Calculate upscaling factor
  upscale_factor <- fine_dim[1] / coarse_dim[1]
  
  # Prepare data for Keras - reshape to [time, x, y, variables]
  if (verbose > 0) cat("Reshaping data for Keras format [time, x, y, variables]...\n")
  
  # Permute dimensions from [x, y, variables, time] to [time, x, y, variables]
  coarse_data_keras <- aperm(coarse_data, c(4, 1, 2, 3))
  if (!is.null(val_coarse_data)) {
      val_coarse_data <- aperm(val_coarse_data, c(4, 1, 2, 3))
  }
  bottleneck_width <- round(dim(fine_data)[1]*0.5^(length(filters) - 1))
  bottleneck_height <- round(dim(fine_data)[2]*0.5^(length(filters) - 1))
  fine_data_keras <- aperm(fine_data, c(4, 1, 2, 3))
  if (!is.null(val_fine_data)) {
      val_fine_data <- aperm(val_fine_data, c(4, 1, 2, 3))
  }
  print(dim(coarse_data_keras))
  print(dim(fine_data_keras))
  print(dim(val_coarse_data))
  print(dim(val_fine_data))

  # Initialize storage for scalers
  scalers <- list()
  
  # Normalize data if requested
  if (normalize) {
    if (verbose > 0) cat("Normalizing data...\n")
    
    # For input data
    input_means <- apply(coarse_data_keras, 4, function(x) mean(x, na.rm = TRUE))
    input_sds <- apply(coarse_data_keras, 4, function(x) stats::sd(x, na.rm = TRUE))
    coarse_data_cent <- sweep(coarse_data_keras, 4, input_means, "-")
    coarse_data_norm <- sweep(coarse_data_cent, 4, input_sds, "/")
    if (!is.null(val_coarse_data)) {
      val_coarse_data_cent <- sweep(val_coarse_data, 4, input_means, "-")
      val_coarse_data_norm <- sweep(val_coarse_data_cent, 4, input_sds, "/")
      val_data_coarse <- val_coarse_data_norm
    }
    
    # For target data
    target_means <- apply(fine_data_keras, 4, function(x) mean(x, na.rm = TRUE))
    target_sds <- apply(fine_data_keras, 4, function(x) stats::sd(x, na.rm = TRUE))
    fine_data_cent <- sweep(fine_data_keras, 4, target_means, "-")
    fine_data_norm <- sweep(fine_data_cent, 4, target_sds, "/")
    if (!is.null(val_fine_data)) {
      val_fine_data_cent <- sweep(val_fine_data, 4, target_means, "-")
      val_fine_data_norm <- sweep(val_fine_data_cent, 4, target_sds, "/")
      val_data_fine <- val_fine_data_norm
    }
    
    # Store scalers
    scalers$input <- list(mean = input_means, sd = input_sds)
    scalers$target <- list(mean = target_means, sd = target_sds)
    
    # Use normalized data for training
    x_train <- coarse_data_norm
    y_train <- fine_data_norm
  } else {
    x_train <- coarse_data_keras
    y_train <- fine_data_keras
  }
  # Create masks for missing values
  input_mask <- is.na(x_train)
  x_train[input_mask] <- 0
  target_mask <- is.na(y_train)
  y_train[target_mask] <- 0
  print(dim(target_mask))

  if (!is.null(seed)) {
    set.seed(seed)
    tensorflow::tf$random$set_seed(seed)
  }
  max_season <- NULL
  # --- Temporal branch ---
  if (!is.null(time_points)) {
    if (!is.null(cyclical_period)) {
      season <- (time_points - 1) %/% (cyclical_period)
      season_onehot <- stats::model.matrix(~ as.factor(season) - 1)
      max_season <- max(season)
      if (cycle_onehot) {
        t_input_season <- keras3::layer_input(shape = c(ncol(season_onehot)))
      } else {
        t_input_season <- NULL
      }
      time_points <- time_points %% cyclical_period
      max_time_point <- cyclical_period
      if(!is.null(val_time_points)) {
          val_season <- (val_time_points - 1) %/% (cyclical_period)
          val_season_onehot <- stats::model.matrix(~ as.factor(val_season) - 1)
      }
    } else {
      t_input_season <- NULL
      max_time_point <- max(time_points)
    }
    min_time_point <- min(time_points)
    t_input <- keras3::layer_input(shape = c(1), name = "time_input")
    if (cos_sin_transform) {
        t_input_cos <- t_input %>% keras3::layer_lambda(function(x) tensorflow::tf$keras$ops$cos(2 * pi * x / cyclical_period), output_shape = c(1))
        t_input_sin <- t_input %>% keras3::layer_lambda(function(x) tensorflow::tf$keras$ops$sin(2 * pi * x / cyclical_period), output_shape = c(1))
        if (!is.null(t_input_season)) {
          t_features <- keras3::layer_concatenate(list(t_input_cos, t_input_sin, t_input_season))
        } else {
          t_features <- keras3::layer_concatenate(list(t_input_cos, t_input_sin))
        }
    } else {
        t_features <- t_input %>% layer_temporal_radial_basis(
            temporal_basis = temporal_basis,
            min_time_point = min_time_point,
            max_time_point = max_time_point
        )
        if (!is.null(t_input_season)) {
          t_features <- keras3::layer_concatenate(list(t_features, t_input_season))
        }
    }
    t_branch <- t_features
    for (i in seq_along(temporal_layers)) {
      t_branch <- t_branch %>% 
        keras3::layer_dense(units = temporal_layers[i], activation = activation)
    }
    t_branch <- t_branch %>% 
      keras3::layer_dense(bottleneck_width * bottleneck_height, activation = activation) %>%
      keras3::layer_reshape(c(bottleneck_width, bottleneck_height, 1))
  } else {
    min_time_point <- NULL
    max_time_point <- NULL
  }

  # Define UNet model architecture
  if (verbose > 0) cat("Building UNet model...\n")
    
  input_layer <- keras3::layer_input(shape = input_shape)

  # Initial upscaling
  x <- input_layer %>% keras3::layer_upsampling_2d(size = c(upscale_factor, upscale_factor), interpolation = "bilinear")

  # Initial feature extraction
  for (i in seq_along(initial_filters)) {
    x <- x %>% keras3::layer_conv_2d(
      filters = initial_filters[i], kernel_size = initial_kernel_sizes[[i]], 
      padding = "same", activation = "linear"
    )
    if (use_batch_norm) {
    x <- x %>% keras3::layer_batch_normalization()
    }
    x <- x %>% keras3::layer_activation(activation)
  }

  # Encoder path
  encoder_blocks <- list()
  
  for (i in seq_along(filters)) {
    x <- x %>% keras3::layer_conv_2d(
      filters = filters[i], kernel_size = kernel_sizes[[i]], 
      padding = "same", activation = "linear"
    )
    if (use_batch_norm) {
      x <- x %>% keras3::layer_batch_normalization()
    }
    x <- x %>% keras3::layer_activation(activation)
    x <- x %>% keras3::layer_conv_2d(
      filters = filters[i], kernel_size = kernel_sizes[[i]], 
      padding = "same", activation = "linear"
    )
    if (use_batch_norm) {
      x <- x %>% keras3::layer_batch_normalization()
    }
    x <- x %>% keras3::layer_activation(activation)
    
    # Store for skip connections
    encoder_blocks[[i]] <- x
    
    # Add pooling if not the last layer
    if (i < length(filters)) {
      x <- x %>% keras3::layer_max_pooling_2d(pool_size = c(2, 2), padding = "same")
    }
  }
  x <- x %>% keras3::layer_dropout(rate = dropout_rate) # Optional "spatial dropout"

  if (!is.null(time_points)) {
    x <- keras3::layer_concatenate(list(x, t_branch), axis = -1)
  }
  
  # Decoder path
  for (i in length(filters):2) {
    # Upsampling
    x <- x %>% keras3::layer_upsampling_2d(size = c(2, 2), interpolation = "bilinear") %>%
      keras3::layer_conv_2d(filters = filters[i], kernel_sizes[[i]], padding = "same")
    if (use_batch_norm) {
      x <- x %>% keras3::layer_batch_normalization()
    }
    x <- x %>% keras3::layer_activation(activation)    
    # Skip connection
    if (i < length(filters)) {
      x <- keras3::layer_concatenate(list(x, encoder_blocks[[i - 1]]))
    }

    x <- x %>% keras3::layer_conv_2d(
      filters = filters[i], kernel_size = kernel_sizes[[i]],
      padding = "same", activation = "linear"
    )
    if (use_batch_norm) {
      x <- x %>% keras3::layer_batch_normalization()
    }
    x <- x %>% keras3::layer_activation(activation)
    x <- x %>% keras3::layer_conv_2d(
      filters = filters[i], kernel_size = kernel_sizes[[i]],
      padding = "same", activation = "linear"
    )
    if (use_batch_norm) {
      x <- x %>% keras3::layer_batch_normalization()
    }
    x <- x %>% keras3::layer_activation(activation)
  }
    
  # Output layer
  output_layer <-  x %>% keras3::layer_conv_2d(
    filters = output_shape[3], kernel_size = c(1, 1),
    padding = "same", activation = final_activation
  )

  if (!is.null(time_points)) {
    inputs <- list(input_layer, t_input)
    if (!is.null(t_input_season)) {
      inputs <- append(inputs, t_input_season)
      input_data <- list(x_train, keras3::array_reshape(time_points, c(length(time_points), 1)), season_onehot)
    } else {
      input_data <- list(x_train, keras3::array_reshape(time_points, c(length(time_points), 1)))
    }
  } else {
    inputs <- input_layer
    input_data <- x_train
  }
  if (!is.null(val_coarse_data) & !is.null(val_fine_data)) {
      if (!is.null(time_points)) {
          if (!is.null(t_input_season)) {
              val_input_data <- list(val_data_coarse, keras3::array_reshape(val_time_points %% cyclical_period, c(length(val_time_points), 1)), val_season_onehot)
          } else {
              val_input_data <- list(val_data_coarse, keras3::array_reshape(val_time_points %% cyclical_period, c(length(val_time_points), 1)))
          }
      } else {
          val_input_data <- val_data_coarse
      }
      validation_data <- list(val_input_data, val_data_fine)
      validation_split <- 0
  } else {
      validation_data <- NULL
  }
  
  # Create model
  model <- keras3::keras_model(inputs = inputs, outputs = output_layer)
  print(paste0("Validation input shape: ", dim(val_data_coarse)))
  print(paste0("Validation output shape: ", dim(val_data_fine)))

  # Configure optimizer with learning rate
  if (optimizer == "adam") {
    opt <- keras3::optimizer_adam(learning_rate = learning_rate)
  } else if (optimizer == "rmsprop") {
    opt <- keras3::optimizer_rmsprop(learning_rate = learning_rate)
  } else if (optimizer == "sgd") {
    opt <- keras3::optimizer_sgd(learning_rate = learning_rate)
  } else {
    opt <- optimizer  # Assume it's already an optimizer object
  }
  
  # Compile the model
  model %>% keras3::compile(
    optimizer = opt,
    loss = loss,
    metrics = metrics
  )
  
  # Train the model
  if (verbose > 0) cat("Training UNet model...\n")
  
  history <- model %>% keras3::fit(
    x = input_data,
    y = y_train,
    batch_size = batch_size,
    epochs = epochs,
    sample_weight = !target_mask[, , , 1],
    validation_data = validation_data,
    validation_split = validation_split,
    callbacks = callbacks,
    verbose = verbose
  )
  # Return model and associated components
  obj <- list(
    model = model,
    scalers = scalers,
    input_mask = input_mask,
    target_mask = target_mask,
    min_time_point = min_time_point,
    max_time_point = max_time_point,
    cyclical_period = cyclical_period,
    cycle_onehot = cycle_onehot,
    max_season = max_season,
    axis_names = axis_names,
    history = history
  )
  class(obj) <- "UNet"
  return(obj)
}

#' Predict function for UNet model
#' 
#' @description 
#' Generates predictions using the trained UNet model.
#' 
#' @param object A UNet model object.
#' @param newdata Array or list of arrays. New data to predict on in format `[x, y, variables, time]``.
#' @param time_points An optional numeric vector containing the time points of the new data.
#' @param ... Additional arguments (not used).
#' 
#' @details
#' The predict function applies the trained UNet model to new coarse data.
#' It performs denormalization if the model was trained with normalization.

#' @examples
#' \dontrun{
#' library(keras3)
#'
#' # Create tiny dummy data:
#' # Coarse grid: 8x8 → Fine grid: 16x16
#' nx_c <- 8; ny_c <- 8
#' nx_f <- 16; ny_f <- 16
#' T <- 5  # number of time steps
#'
#' # Coarse data:
#' coarse_data <- array(runif(nx_c * ny_c * 1 * T),
#'                      dim = c(nx_c, ny_c, 1, T))
#'
#' # Fine data:
#' fine_data <- array(runif(nx_f * ny_f * 1 * T),
#'                    dim = c(nx_f, ny_f, 1, T))
#'
#' # Optional time points
#' time_points <- 1:T
#'
#' # Fit a tiny UNet (very small filters to keep the example fast)
#' model_obj <- unet_downscale(
#'   coarse_data,
#'   fine_data,
#'   time_points = time_points,
#'   filters = c(8, 16),
#'   initial_filters = c(4),
#'   epochs = 2,
#'   batch_size = 2,
#'   verbose = 0
#' )
#' 
#' T_new <- 3
#' newdata <- array(runif(nx_c * ny_c * 1 * T_new),
#'                      dim = c(nx_c, ny_c, 1, T_new))
#' predictions <- predict(model_obj, newdata, 1:T_new)
#' }
#' 
#' @export
predict.UNet <- function(object, newdata, time_points = NULL) {
  if (is.null(object$model)) stop("The model has not been trained yet.")
  
  newdata <- aperm(newdata, c(3, 1, 2))
  newdata <- (newdata - object$scalers$input$mean) / object$scalers$input$sd
  newdata <- keras3::array_reshape(newdata, c(dim(newdata), 1))
  newdata[is.na(newdata)] <- 0
  
  # Temporal input if required
  if (!is.null(object$min_time_point)) {
    if (is.null(time_points)) {
      stop("This UNet was trained with temporal input, please provide 'time_points'.")
    }
    if (length(time_points) != dim(newdata)[1]) {
      stop(sprintf("Number of time_points (%d) does not match number of samples (%d).",
                   length(time_points), dim(newdata)[1]))
    }
    if (!is.null(object$cyclical_period)) {
      all_seasons <- 0:object$max_season
      season <- (time_points - 1) %/% (object$cyclical_period)
      max_season_inds <- which(season > object$max_season)
      season[max_season_inds] <- object$max_season
      season_onehot <- model.matrix(~ as.factor(c(all_seasons, season)) - 1)
      season_onehot <- season_onehot[-(1:length(all_seasons)), ]
      time_points <- time_points %% object$cyclical_period
    }
    time_points <- keras3::array_reshape(time_points, c(length(time_points), 1))
    if (!is.null(object$cyclical_period) & !is.null(object$cycle_onehot))  {
      inputs <- list(newdata, time_points, season_onehot)
    } else {
      inputs <- list(newdata, time_points)
    }
  } else {
    inputs <- newdata
  }
  
  predictions <- object$model(inputs)
  predictions <- predictions * object$scalers$target$sd + object$scalers$target$mean

  mask_pred <- abind::abind(replicate(dim(predictions)[1], object$target_mask[1, , ,]), along = 1)
  mask_pred <- aperm(mask_pred, c(3, 1, 2))
  mask_pred <- keras3::array_reshape(mask_pred, c(dim(predictions)))
  
  predictions <- as.array(predictions)
  predictions[mask_pred] <- NA
  predictions <- aperm(predictions, c(2, 3, 1, 4))
  new_dimnames <- append(object$axis_names[1:2], list(1:dim(predictions)[3], 1:dim(predictions)[4]))
  dimnames(predictions) <- new_dimnames
  
  return(predictions)
}
