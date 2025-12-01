evaluate_downscaling <- function(model, test_input, test_target,
                                 metrics = c("rmse", "mae", "correlation"),
                                 plot = TRUE) {
  # Check for required packages
  if (!requireNamespace("stats", quietly = TRUE)) {
    stop("Package 'stats' is required for evaluation")
  }

  # Make predictions
  predictions <- model$predict(test_input)

  # Compute metrics
  results <- list()

  # Flatten arrays for calculations
  pred_flat <- as.vector(predictions)
  target_flat <- as.vector(test_target)

  # Remove any NAs
  valid_idx <- !is.na(pred_flat) & !is.na(target_flat)
  pred_flat <- pred_flat[valid_idx]
  target_flat <- target_flat[valid_idx]

  # Calculate metrics
  if ("rmse" %in% metrics) {
    results$rmse <- sqrt(mean((pred_flat - target_flat)^2))
  }

  if ("mae" %in% metrics) {
    results$mae <- mean(abs(pred_flat - target_flat))
  }

  if ("correlation" %in% metrics) {
    results$correlation <- stats::cor(pred_flat, target_flat)
  }

  if ("bias" %in% metrics) {
    results$bias <- mean(pred_flat - target_flat)
  }

  # Create plots if requested
  if (plot && requireNamespace("ggplot2", quietly = TRUE)) {
    # Sample index for visualization (first time step)
    sample_idx <- 1

    # Prepare data for plotting
    plot_data <- data.frame(
      Predicted = as.vector(predictions[sample_idx, , ]),
      Observed = as.vector(test_target[sample_idx, , ])
    )

    # Scatter plot
    scatter_plot <- ggplot2::ggplot(plot_data, ggplot2::aes(x = Observed, y = Predicted)) +
      ggplot2::geom_point(alpha = 0.3) +
      ggplot2::geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
      ggplot2::theme_minimal() +
      ggplot2::labs(title = "Predicted vs Observed Values")

    # Add spatial maps if enough dimensions
    if (length(dim(predictions)) >= 3) {
      # Example showing input, prediction, target and difference
      # This would need to be customized based on actual data dimensions
      results$plots <- list(
        scatter = scatter_plot
        # Additional plots would go here
      )
    } else {
      results$plots <- list(scatter = scatter_plot)
    }
  }

  return(results)
}

neighborhood_fixed_n <- function(coords, neighborhood_size) {
  n <- nrow(coords)
  d <- as.matrix(distances::distances(coords))
  index_matrix <- matrix(rep(seq(1:n), n), ncol = n, byrow = TRUE)
  index_sorted <- matrix(index_matrix[order(row(d), d)],
    ncol = n,
    byrow = TRUE
  )
  neighbor_indexes <- index_sorted[, 1:neighborhood_size]
  neighborhood_matrix <- index_matrix
  for (i in 1:n) {
    neighborhood_matrix[i, ][neighbor_indexes[i, ]] <- 1
    neighborhood_matrix[i, ][-neighbor_indexes[i, ]] <- 0
  }
  return(neighborhood_matrix)
}

neighborhood_fixed_radius <- function(coords, radius) {
  n <- dim(coords)[1]
  d <- as.matrix(distances::distances(coords))
  d[which(d <= radius)] <- 1
  d[which(d > radius)] <- 0
  d <- d - diag(n)
  return(d)
}

#' @export
run_in_new_session <- function(
    new_session_function,
    libraries = NULL, dependencies = NULL, ...) {
  if (!is.null(dependencies)) {
    source(dependencies)
  }
  # Define a wrapper function to run in a separate session
  session_function <- function(func, libraries, ...) {
    if (!is.na(libraries)) {
      for (library in libraries) {
        if (!requireNamespace(library, quietly = TRUE)) {
          stop(paste("Required library", library, "is not installed."))
        }
        library(library, character.only = TRUE)
      }
    }

    # Run the model training function and return the trained model
    obj <- func(...)
    return(obj)
  }

  # Call the wrapper function in a separate R session
  result <- callr::r(
    session_function,
    args = list(
      func = new_session_function,
      libraries = libraries, 
      ...
    )
  )

  return(result)
}

# Custom layer for temporal radial basis functions
layer_temporal_radial_basis <- keras3::Layer(
  "TemporalRadialBasisLayer",
  initialize = function(temporal_basis,
                        min_time_point, max_time_point) {
    super$initialize()
    self$temporal_basis <- as.integer(temporal_basis)
    self$min_time_point <- tensorflow::tf$constant(min_time_point, dtype = tensorflow::tf$float32)
    self$max_time_point <- tensorflow::tf$constant(max_time_point, dtype = tensorflow::tf$float32)

    # Pre-compute knots for temporal basis functions
    self$temporal_knots <- list()
    self$temporal_kappas <- c()
    
    for (i in seq_along(temporal_basis)) {
      temp_knots <- seq(min_time_point, max_time_point, length.out = temporal_basis[i] + 2)
      temp_knots <- temp_knots[2:(length(temp_knots) - 1)]  # interior knots

      self$temporal_knots[[i]] <- tensorflow::tf$constant(matrix(temp_knots, ncol = 1), dtype = tensorflow::tf$float32)
      kappa <- abs(temp_knots[1] - temp_knots[2])
      self$temporal_kappas[i] <- kappa^2
    }
    self$temporal_kappas <- tensorflow::tf$constant(self$temporal_kappas, dtype = tensorflow::tf$float32)
  },
  
  call = function(inputs, mask = NULL) {
    # inputs: time_points (batch_size, 1)
    time_points <- inputs
    
    phi_list <- list()
    
    for (i in seq_along(self$temporal_basis)) {
      time_expanded <- tensorflow::tf$expand_dims(time_points, axis = -1L)  # (batch_size, 1, 1)
      temp_knots_expanded <- tensorflow::tf$expand_dims(self$temporal_knots[[i]], axis = 0L)  # (1, num_temp_knots, 1)
      temp_knots_expanded <- tensorflow::tf$transpose(temp_knots_expanded, perm = c(0L, 2L, 1L))  # (1, 1, num_temp_knots)

      temp_distances <- tensorflow::tf$abs(time_expanded - temp_knots_expanded)  # (batch_size, 1, num_temp_knots)
      temp_distances <- tensorflow::tf$squeeze(temp_distances, axis = 1L)  # (batch_size, num_temp_knots)
      
      # Gaussian kernel
      phi_temp <- tensorflow::tf$exp(-0.5 * temp_distances^2 / self$temporal_kappas[i])

      phi_list[[length(phi_list) + 1]] <- phi_temp
    }
    
    phi_all <- tensorflow::tf$concat(phi_list, axis = 1L)
    return(phi_all)  # (batch_size, total_temporal_basis)
  }
)
