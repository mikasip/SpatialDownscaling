#' Daily Weather Data for Italy
#'
#' A dataset containing daily gridded weather variables for Italy.
#'
#' @format A 4-dimensional array with dimensions:
#' \describe{
#'   \item{longitude}{Longitude grid points (0.2 degree resolution)}
#'   \item{latitude}{Latitude grid points (0.2 degree resolution)}
#'   \item{variable}{Weather variables: relative humidity, temperature, total precipitation}
#'   \item{time}{Daily measurements from 2023-11-01 to 2023-12-31}
#' }
"weather_italy"