#' Super Resolution CNN for Spatial Downscaling
#' @importFrom magrittr %>%
#' 
#' @description
#' This function implements a Time-aware Super Resolution Deep Neural Network (SRDRN)
#' for spatial downscaling of grid based data using the TensorFlow and Keras libraries.
#' The function allows an option for adding a temporal module for spatio-temporal applications.
#'
#' @param input_data A 3D array of shape (N_1, N_1, n) representing the input data. 
#' @param target_data A 3D array of shape (N_2, N_2, n) representing the target data.
#' @param time_points An optional numeric vector of length n representing the time points associated with each sample.
#' @param val_input_data An optional 3D array of shape (N_1, N_1, n) representing the input validation data.
#' @param val_target_data An optional 3D array of shape (N_2, N_2, n) representing the target validation data.
#' @param val_time_points An optional numeric vector of length n representing the time points of the validation samples.
#' @param cyclical_period An optional numeric value representing the cyclical period for time encoding (e.g. 365 for yearly seasonality).
#' @param temporal_basis A numeric vector specifying the temporal basis functions to use for time encoding (default is c(9, 17, 37)).
#' @param temporal_layers A numeric vector specifying the number of units in each dense layer for time encoding (default is c(32, 64, 128)).
#' @param activation A character string specifying the activation function to use (default is "relu").
#' @param cos_sin_time A logical value indicating whether to use cosine and sine transformations for time encoding (default is FALSE).
#' @param use_batch_norm A logical value indicating whether to use batch normalization in the residual blocks (default is FALSE).
#' @param output_channels An integer specifying the number of output channels (default is 1).
#' @param num_residual_blocks An integer specifying the number of residual blocks in the model (default is 3).
#' @param validation_split A numeric value between 0 and 1 specifying the fraction of the training data to use for validation (default is 0.2).
#' @param start_from_model An optional pre-trained Keras model to continue training from (default is NULL).
#' @param epochs An integer specifying the number of training epochs (default is 10).
#' @param batch_size An integer specifying the batch size for training (default is 32).
#' @param seed An optional integer value to set the random seed for reproducibility (default is NULL).
#'
#' @return An object of class SRDRN containing:
#' \item{model}{The trained Keras model.}
#' \item{input_mean}{The mean value of the input data used for normalization.}
#' \item{input_sd}{The standard deviation of the input data used for normalization.}
#' \item{target_mean}{The mean value of the target data used for normalization.}
#' \item{target_sd}{The standard deviation of the target data used for normalization.}
#' \item{input_mask}{A logical array indicating the missing values in the input data.}
#' \item{target_mask}{A logical array indicating the missing values in the target data.}
#' \item{min_time_point}{The minimum time point in the input data.}
#' \item{max_time_point}{The maximum time point in the input data.}
#' \item{cyclical_period}{The cyclical period used for temporal encoding.}
#' \item{axis_names}{A list containing the names of the axes (longitude, latitude, time).}
#' \item{history}{The training history of the model.}
#'
#' @details
#' The Super Resolution Deep Residual Network (SRDRN) implements a deep-learning-based
#' spatial downscaling approach inspired by Super-Resolution CNNs (SRCNN)
#' \insertCite{dong2015image}{SpatialDownscaling} and extended for environmental
#' applications following \insertCite{wang2021deep}{SpatialDownscaling}.  
#'
#' The objective of SRDRN is to learn a mapping from coarse-resolution gridded fields
#' to finer-resolution targets by combining convolutional feature extraction,
#' residual learning, and sub-pixel upsampling. The method is designed for both
#' purely spatial and fully spatio-temporal downscaling when time information is
#' provided. The method consists of the following main components:
#'
#' \itemize{
#'   \item *Feature Extraction Block*:  
#'   An initial convolutional layer extracts low-level spatial features from the
#'   coarse-resolution input.
#'
#'   \item *Residual Blocks*:  
#'   A sequence of residual blocks learn higher-order spatial dependencies.
#'   Residual connections stabilize training and allow deeper representations.
#'
#'   \item *Upsampling Module*:  
#'   Sub-pixel convolution (pixel shuffle) layers upscale feature maps to match
#'   the high-resolution target grid.
#' }
#'  
#' If `time_points` are provided, the model includes an auxiliary temporal branch.
#' Time is encoded either via:
#' \itemize{
#'   \item Radial basis temporal encodings (`temporal_basis`), or  
#'   \item Cosine–sine cyclical encodings (`cos_sin_time = TRUE`).  
#' }
#' The encoded temporal features pass through a multilayer perceptron
#' (`temporal_layers`) and are reshaped to spatial form before being concatenated
#' with CNN features. This enables learning time-varying downscaling dynamics
#' (e.g., seasonality, long-term trends).
#'
#' @examples
#' \dontrun{
#' # Generate dummy low-resolution (16×16) and high-resolution (32×32) data
#' n <- 20
#' input  <- array(runif(16 * 16 * n),  dim = c(16, 16, n))
#' target <- array(runif(32 * 32 * n),  dim = c(32, 32, n))
#'
#' # Example 1: Spatial downscaling (no time)
#' model1 <- srdrn(
#'   input_data  = input,
#'   target_data = target,
#'   epochs = 2,
#'   batch_size = 4
#' )
#'
#' # Example 2: Spatio-temporal downscaling using time points
#' time_vec <- 1:n
#' model2 <- srdrn(
#'   input_data  = input,
#'   target_data = target,
#'   time_points = time_vec,
#'   cyclical_period = 365,
#'   temporal_layers = c(32, 64),
#'   epochs = 2,
#'   batch_size = 4
#' )
#' }
#' 
#' @references 
#' \insertAllCited{}
#' 
#' @export
srdrn <- function(input_data, target_data, time_points = NULL, 
                    val_input_data = NULL, val_target_data = NULL, val_time_points = NULL, 
                    cyclical_period = NULL, temporal_basis = c(9, 17, 37), 
                    temporal_layers = c(32, 64, 128), activation = "relu", cos_sin_time = FALSE,
                    use_batch_norm = FALSE, output_channels = 1, num_residual_blocks = 3,
                    validation_split = 0, start_from_model = NULL,
                    epochs = 10, batch_size = 32, seed = NULL) {

    if (dim(input_data)[3] != dim(target_data)[3]) {
        stop("The number of samples in 'input_data' and 'target_data' must be the same.")
    }
    axis_names <- dimnames(target_data)
    input_data <- aperm(input_data, c(3, 1, 2))
    input_width <- dim(input_data)[2]
    input_height <- dim(input_data)[3]
    target_data <- aperm(target_data, c(3, 1, 2))
    if (length(dim(input_data)) == 4) {
        output_channels <- dim(input_data)[4]
    } else {
        output_channels <- 1
    }

    if (!is.null(seed)) {
        set.seed(seed)
        tensorflow::tf$random$set_seed(seed)
        keras3::set_random_seed(as.integer(seed))
    }

    if (!is.null(time_points)) {
        if (!is.null(cyclical_period)) {
            time_points <- time_points %% cyclical_period
            max_time_point <- cyclical_period
            if (!is.null(val_time_points)) {
                val_time_points <- val_time_points %% cyclical_period
            }
        } else {
            max_time_point <- max(time_points)
        }
        min_time_point <- min(time_points)
        t_input <- keras3::layer_input(shape = c(1), name = "time_input")
        if (cos_sin_time) {
            t_input_cos <- t_input %>% keras3::layer_lambda(function(x) tensorflow::tf$keras$ops$cos(2 * pi * x / cyclical_period), output_shape = c(1))
            t_input_sin <- t_input %>% keras3::layer_lambda(function(x) tensorflow::tf$keras$ops$sin(2 * pi * x / cyclical_period), output_shape = c(1))
            t_features <- keras3::layer_concatenate(list(t_input_cos, t_input_sin))
        } else {
            t_features <- t_input %>% layer_temporal_radial_basis(
                temporal_basis = temporal_basis,
                min_time_point = min_time_point,
                max_time_point = max_time_point
            )
        }
        t_branch <- t_features
        for (i in seq_along(temporal_layers)) {
            t_branch <- t_branch %>% keras3::layer_dense(units = temporal_layers[i], activation = activation)
        }
        t_branch <- t_branch %>% keras3::layer_dense(input_width * input_height, activation = activation) %>%
            keras3::layer_reshape(c(input_width, input_height, 1))
    } else {
        min_time_point <- NULL
        max_time_point <- NULL
    }

    upscale_factor <- as.integer(dim(target_data)[2] / dim(input_data)[2])
    
    # Normalize input and target data
    input_mean <- mean(input_data, na.rm = TRUE)
    input_sd <- stats::sd(input_data, na.rm = TRUE)
    input_data <- (input_data - input_mean) / input_sd
    if (!is.null(val_input_data)) {
        val_input_data <- aperm(val_input_data, c(3, 1, 2))
        val_input_data <- (val_input_data - input_mean) / input_sd
    }
    target_mean <- mean(target_data, na.rm = TRUE)
    target_sd <- stats::sd(target_data, na.rm = TRUE)
    target_data <- (target_data - target_mean) / target_sd
    if (!is.null(val_target_data)) {
        val_target_data <- aperm(val_target_data, c(3, 1, 2))
        val_target_data <- (val_target_data - target_mean) / target_sd
    }
    
    # Define residual block function
    residual_block <- function(x, filters, kernel_size = c(3, 3), use_batch_norm = TRUE) {
        skip <- x
        
        # First convolution
        x <- keras3::layer_conv_2d(x, filters = filters, kernel_size = kernel_size, padding = "same")
        
        # Optional batch normalization
        if (use_batch_norm) {
            x <- keras3::layer_batch_normalization(x)
        }
        
        # ReLU activation
        x <- keras3::layer_activation(x, activation = "relu")
        
        # Second convolution
        x <- keras3::layer_conv_2d(x, filters = filters, kernel_size = kernel_size, padding = "same")
        
        # Optional batch normalization
        if (use_batch_norm) {
            x <- keras3::layer_batch_normalization(x)
        }
        
        # Skip connection (element-wise sum)
        x <- keras3::layer_add(list(x, skip))
        
        return(x)
    }

    subpixel_conv2d <- function(scale) {
        keras3::layer_lambda(f = function(x) {
            tensorflow::tf$nn$depth_to_space(x, block_size = scale)
        })
    }

    upsampling_block <- function(x, filters, kernel_size = c(3, 3), scale = 2) {
    # Convolution that outputs filters * (scale^2) feature maps
    x <- keras3::layer_conv_2d(
        x,
        filters = filters * (scale^2),
        kernel_size = kernel_size,
        padding = "same"
    )
    
    # Pixel shuffle (sub-pixel convolution)
    x <- subpixel_conv2d(scale)(x)
    
    # ReLU activation
    x <- keras3::layer_activation(x, activation = "relu")
    
    return(x)
    }

    # Expand dimensions to add channel axis (required for CNNs)
    input_data <- keras3::array_reshape(input_data, c(dim(input_data), 1))
    target_data <- keras3::array_reshape(target_data, c(dim(target_data), 1))
    input_mask <- is.na(input_data)
    input_data[input_mask] <- 0
    target_mask <- is.na(target_data)
    target_data[target_mask] <- 0

    # Define the model
    input_layer <- keras3::layer_input(c(dim(input_data)[2:3], 1))

    # Initial convolution layer
    x <- keras3::layer_conv_2d(input_layer, 
                              filters = 64, 
                              kernel_size = c(3, 3),
                              padding = "same")
    initial_output <- x  # Store for global skip connection

    # Residual blocks
    for (i in 1:num_residual_blocks) {
      x <- residual_block(x, filters = 64, use_batch_norm = use_batch_norm)
    }

    # Convolution layer after residual blocks
    x <- keras3::layer_conv_2d(x, 
                              filters = 64, 
                              kernel_size = c(3, 3),
                              padding = "same")

    # Global skip connection
    x <- keras3::layer_add(list(x, initial_output))

    if (!is.null(time_points)) {
        x <- keras3::layer_concatenate(list(x, t_branch), axis = -1)
    }

    # Upsampling blocks
    upscale_layers <- floor(log2(upscale_factor))
    for (i in 1:upscale_layers) {
      x <- upsampling_block(x, filters = 128)
    }

    # Final convolution to produce output
    x <- keras3::layer_conv_2d(x, 
                              filters = output_channels, 
                              kernel_size = c(3, 3),
                              activation = "linear", 
                              padding = "same")

    if (!is.null(time_points)) {
        inputs <- list(input_layer, t_input)
        input_data <- list(input_data, time_points)
    } else {
        inputs <- list(input_layer)
        input_data <- list(input_data)
    }
    if (!is.null(val_input_data) && !is.null(val_target_data)) {
        if (!is.null(time_points) && !is.null(val_time_points)) {
            val_input_list <- list(val_input_data, val_time_points)
        } else {
            val_input_list <- list(val_input_data)
        }
        validation_data <- list(val_input_list, val_target_data)
    } else {
        validation_data <- NULL
    }
    if (!is.null(start_from_model)) {
        model_sequential <- start_from_model
        keras3::set_weights(model_sequential, keras3::get_weights(model_sequential))
    } else {
        model_sequential <- keras3::keras_model(inputs = inputs, outputs = x)
        model_sequential %>% keras3::compile(
            optimizer = keras3::optimizer_adam(),
            loss = "mse",
            metrics = list("mae", "mse")
        )
    }
    if (!is.null(validation_data)) {
        validation_split <- 0
    }
    
    # Fit the model
    history <- model_sequential %>% keras3::fit(
        input_data,
        target_data,
        sample_weight = !target_mask,
        validation_data = validation_data,
        validation_split = validation_split,
        epochs = epochs,
        batch_size = batch_size,
        shuffle = TRUE
    )
    
    # Return an object of class SRDRN
    srdrn_object <- list(
        model = model_sequential,
        input_mean = input_mean,
        input_sd = input_sd,
        target_mean = target_mean,
        target_sd = target_sd,
        input_mask = input_mask,
        target_mask = target_mask,
        min_time_point = min_time_point,
        max_time_point = max_time_point,
        cyclical_period = cyclical_period,
        axis_names = axis_names,
        history = history
    )
    class(srdrn_object) <- "srdrn"

    return(srdrn_object)
}

#' @title Predict method for SRDRN
#' @description
#' This function makes predictions using a trained SRDRN model.
#' It takes a trained SRDRN object and new data as input,
#' normalizes the new data, and uses the model to make predictions.
#' The predictions are then rescaled back to the original range.
#' 
#' @param object A trained SRDRN object.
#' @param newdata A 3D array of shape (N_1, N_1, n) representing the new data to be predicted.
#' @param ... Additional parameters (not used).
#' 
#' @return A 3D array of shape (N_2, N_2, n) representing the predicted data.
#' 
#' @details
#' The predict method for the SRDRN class takes a trained SRDRN object and new data as input.
#' It normalizes the new data using the same min-max scaling used during training.
#' The new data is reshaped to match the input shape of the model,
#' and the model is used to make predictions.
#' The predictions are then rescaled back to the original range using the min-max scaling parameters
#' obtained during training.
#' The output is a 3D array of the predicted data.
#'
#' @examples
#' \dontrun{
#' # Generate dummy low-resolution (16×16) and high-resolution (32×32) data
#' n <- 20
#' input  <- array(runif(16 * 16 * n),  dim = c(16, 16, n))
#' target <- array(runif(32 * 32 * n),  dim = c(32, 32, n))
#'
#' # Example 2: Spatio-temporal downscaling using time points
#' time_vec <- 1:n
#' model <- srdrn(
#'   input_data  = input,
#'   target_data = target,
#'   time_points = time_vec,
#'   cyclical_period = 365,
#'   temporal_layers = c(32, 64),
#'   epochs = 2,
#'   batch_size = 4
#' )
#' 
#' n_new <- 3
#' newdata <- array(runif(16 * 16 * n_new),
#'                      dim = c(16, 16, n_new))
#' predictions <- predict(model, newdata, 1:n_new)
#' }
#' 
#' @seealso \code{\link{srdrn}} for fitting SRDRN model.
#'
#' @export
predict.srdrn <- function(object, newdata, time_points = NULL, ...) {
    if (is.null(object$model)) {
        stop("The model has not been trained yet.")
    }
    
    # Reorder to (samples, width, height)
    newdata <- aperm(newdata, c(3, 1, 2))
    
    # Normalize
    newdata <- (newdata - object$input_mean) / object$input_sd
    newdata <- keras3::array_reshape(newdata, c(dim(newdata), 1))
    newdata[is.na(newdata)] <- 0
    
    # --- Handle temporal input if model was trained with it ---
    if (!is.null(object$min_time_point)) {
        if (is.null(time_points)) {
            stop("This model was trained with temporal information, please provide 'time_points'.")
        }
        if (!is.null(object$cyclical_period)) {
            time_points <- time_points %% object$cyclical_period
        }
        time_points <- keras3::array_reshape(time_points, c(length(time_points), 1))
        inputs <- list(newdata, time_points)
    } else {
        # model trained without temporal input
        inputs <- newdata
    }
    
    # Predict
    predictions <- object$model(inputs)
    
    # Rescale back
    predictions <- predictions * object$target_sd + object$target_mean
    
    # Apply mask
    mask_pred <- abind::abind(replicate(dim(predictions)[1], object$target_mask[1, , ,]), along = 1)
    mask_pred <- aperm(mask_pred, c(3, 1, 2))
    mask_pred <- keras3::array_reshape(mask_pred, c(dim(predictions)))
    
    predictions <- as.array(predictions)
    predictions[mask_pred] <- NA
    predictions <- aperm(predictions, c(2, 3, 1, 4))
    dimnames(predictions) <- append(object$axis_names[1:2],
                                    list(1:dim(predictions)[3], 1:dim(predictions)[4]))
    
    return(predictions)
}
