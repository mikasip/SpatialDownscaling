#' Daily Weather Data for Italy
#'
#' A dataset containing daily gridded weather variables for Italy that are obtained from the ERA5-Land dataset \insertCite{era5}{SpatialDownscaling}.
#'
#' @format A 4-dimensional array with dimensions:
#' \describe{
#'   \item{longitude}{Longitude grid points (0.2 degree resolution)}
#'   \item{latitude}{Latitude grid points (0.2 degree resolution)}
#'   \item{variable}{Weather variables: relative humidity (%), temperature (°C), total precipitation (meters per day)}
#'   \item{time}{Daily measurements from 2023-11-01 to 2023-12-31}
#' }
"weather_italy"