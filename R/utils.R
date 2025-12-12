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
