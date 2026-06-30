# Code used to export datasets used in "tests.R" to csv for consistency within wider use

library(AddiVortes)

# Test 1 - Boston data

n <- nrow(Boston)
TrainSet <- sort(sample.int(n, 5 * n / 6))
TestSet <- setdiff(1:n, TrainSet)

train_data <- Boston[TrainSet,]
test_data <- Boston[TestSet,]

write.csv(train_data[,-14], "./datasets/boston/x_train.csv", row.names=FALSE)
write.csv(train_data[,14], "./datasets/boston/y_train.csv", row.names=FALSE)
write.csv(test_data[,-14], "./datasets/boston/x_test.csv", row.names=FALSE)
write.csv(test_data[,14], "./datasets/boston/y_test.csv", row.names=FALSE)

# Test 2 - Synthetic data

set.seed(42) # for reproducibility

# Create a 5-column matrix of predictors
X <- matrix(runif(2500), ncol = 5)
X[, 1] <- -10 - X[, 1] * 10
X[, 2] <- X[, 2] * 100
X[, 3] <- -9 + X[, 3] * 10
X[, 4] <- 8 + X[, 4]
X[, 5] <- X[, 5] * 10

# Create the response 'Y' based on a rule and add noise
Y_underlying <- ifelse(-1 * X[, 2] > 10 * X[, 1] + 100, 10, 0)
Y <- Y_underlying + rnorm(length(Y_underlying))

# --- Generate Test Data ---
set.seed(101) # Use a different seed for the test set
testX <- matrix(runif(1000), ncol = 5)
testX[, 1] <- -10 - testX[, 1] * 10
testX[, 2] <- testX[, 2] * 100
testX[, 3] <- -9 + testX[, 3] * 10
testX[, 4] <- 8 + testX[, 4]
testX[, 5] <- testX[, 5] * 10

# Create the true test response values
testY_underlying <- ifelse(-1 * testX[, 2] > 10 * testX[, 1] + 100, 10, 0)
testY <- testY_underlying + rnorm(length(testY_underlying))

write.csv(X, "./datasets/synthetic/x_train.csv", row.names = FALSE)
write.csv(Y, "./datasets/synthetic/y_train.csv", row.names = FALSE)
write.csv(testX, "./datasets/synthetic/x_test.csv", row.names = FALSE)
write.csv(testY, "./datasets/synthetic/y_test.csv", row.names = FALSE)

# Test 3 - Spherical data

set.seed(42)
n <- 300

# Sample random locations on the globe
lat <- runif(n, -pi / 2, pi / 2) # latitude in radians: [-pi/2, pi/2]
lon <- runif(n, -pi, pi) # longitude in radians: [-pi, pi]

# True function: warmer at the equator, slight east-west gradient
y_true <- 20 * cos(lat) + 5 * sin(lon)

# Add observation noise
y <- y_true + rnorm(n, sd = 2)

# Covariate matrix: latitude first, longitude last (required convention)
x <- cbind(lat, lon)

set.seed(101)
n_test <- 200

lat_test <- runif(n_test, -pi / 2, pi / 2)
lon_test <- runif(n_test, -pi, pi)

y_true_test <- 20 * cos(lat_test) + 5 * sin(lon_test)
y_test <- y_true_test + rnorm(n_test, sd = 2)

x_test <- cbind(lat_test, lon_test)

write.csv(x, "./datasets/spherical/x_train.csv", row.names = FALSE)
write.csv(y, "./datasets/spherical/y_train.csv", row.names = FALSE)
write.csv(x_test, "./datasets/spherical/x_test.csv", row.names = FALSE)
write.csv(y_test, "./datasets/spherical/y_test.csv", row.names = FALSE)


# Test 4 - Categorical data

set.seed(123)
n <- 400

x <- data.frame(
  age = rnorm(n, mean = 40, sd = 10),
  income = runif(n, 20, 120), # income in thousands
  region = sample(c("East", "North", "South", "West"), n, replace = TRUE),
  product = sample(c("Basic", "Premium", "Deluxe"), n, replace = TRUE),
  stringsAsFactors = FALSE
)

# True response: depends on continuous and categorical variables
region_effect <- ifelse(x$region == "North", 5,
                        ifelse(x$region == "South", -5, 0)
)
product_effect <- ifelse(x$product == "Premium", 10,
                         ifelse(x$product == "Deluxe", 20, 0)
)

y <- 0.3 * x$age +
  0.1 * x$income +
  region_effect +
  product_effect +
  rnorm(n, sd = 3)

# Split into training and test sets
set.seed(42)
train_idx <- sample(n, 300)

write.csv(x[train_idx,], "./datasets/categorical/x_train.csv", row.names = FALSE)
write.csv(x[-train_idx,], "./datasets/categorical/x_test.csv", row.names = FALSE)
write.csv(y[train_idx], "./datasets/categorical/y_train.csv", row.names = FALSE)
write.csv(y[-train_idx], "./datasets/categorical/y_test.csv", row.names = FALSE)