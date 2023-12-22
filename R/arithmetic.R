
#' Imitate knn
#'
#' @param train_data split the original data randomly into 2 datasets, one is training dataset, train data is its features
#' @param train_labels split the original data randomly into 2 datasets, one is training dataset, train_labels is its target
#' @param test_data split the original data randomly into 2 datasets, another one is testing dataset, test_data is its features
#' @param k number of neighbours in a group
#' @param distance_function such as Euclidean distance,Manhattan distance and Minkowski distance
#'
#' @return prediction of the target variable
#' @export
#'
#' @examples
#' library(tidyr)
#' library(palmerpenguins)
#' library(tidyverse)
#' library(dplyr)
#' data("penguins")
#' penguins <- na.omit(penguins)
#' penguins <- penguins %>% select(-year)
#' features <- penguins[, c("bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g")]
#' target <- penguins$species
#' set.seed(123)
#' indices <- sample(1:nrow(features), size = 0.7 * nrow(features))
#' train_x <- features[indices, ]
#' train_y <- target[indices]
#' test_x <- features[-indices, ]
#' test_y <- target[-indices]
#' k <- 5
#' euclidean_distance = function(a, b){if(length(a) == length(b)){sqrt(sum((a-b)^2))  } else{stop('Vectors must be of the same length')}}
#' predictions <- knn(train_x, train_y, test_x, k, euclidean_distance)
knn <- function(train_data, train_labels, test_data, k, distance_function) {
  # Helper function to find the most common label
  most_common_label <- function(labels) {
    return(names(sort(table(labels), decreasing = TRUE)[1]))
  }

  # Applying the KNN algorithm
  predictions <- sapply(1:nrow(test_data), function(i) {
    # Calculate distances between the test point and all training points
    distances <- sapply(1:nrow(train_data), function(j) {
      distance_function(test_data[i, ], train_data[j, ])
    })

    # Combine distances with labels and sort them
    neighbors <- data.frame(Distance = distances, Label = train_labels) %>%
      arrange(Distance) %>%
      head(k)

    # Return the most common label among the neighbors
    return(most_common_label(neighbors$Label))
  })

  return(predictions)
}




#' use cross-validation to tune hyperparameter(k_folds)
#'
#' @param data dataset use here
#' @param target the dependent variable to predict
#' @param k_folds number of folds in cross-validation
#' @param k_neighbors number of neighbours in a group
#' @param distance_function such as Euclidean distance,Manhattan distance and Minkowski distance
#'
#' @return accuracy
#' @export
#'
#' @examples
#' library(tidyr)
#' library(palmerpenguins)
#' library(dplyr)
#' data("penguins")
#' k_folds <- 5
#' k_neighbors <- 5
#' euclidean_distance = function(a, b){if(length(a) == length(b)){sqrt(sum((a-b)^2))  } else{stop('Vectors must be of the same length')}}
#' knn_cv(penguins, "species", k_folds, k_neighbors, euclidean_distance)
knn_cv <- function(data, target, k_folds, k_neighbors, distance_function) {
  set.seed(123)
  shuffled_data <- data[sample(nrow(data)), ]

  # Split data into k folds
  folds <- split(shuffled_data, cut(seq(1, nrow(shuffled_data)), breaks = k_folds, labels = FALSE))

  # Define a function to calculate accuracy
  calculate_accuracy <- function(predictions, test_labels) {
    sum(predictions == test_labels) / length(test_labels)
  }

  accuracies <- numeric(k_folds)

  # Cross-validation
  for(i in 1:k_folds) {
    # Splitting the data into training and test sets
    test_data <- folds[[i]]
    train_data <- do.call("rbind", folds[-i])

    # Extract labels
    train_labels <- train_data[[target]]
    test_labels <- test_data[[target]]

    # Remove labels from features
    train_data <- train_data[, !names(train_data) %in% target]
    test_data <- test_data[, !names(test_data) %in% target]

    predictions <- knn(train_data, train_labels, test_data, k_neighbors, distance_function)
    accuracies[i] <- calculate_accuracy(predictions, test_labels)
  }

  return(accuracies)
}


