# This function uses code from the articles below to give a benchmark test for run times on
# 4 different examples using the AddiVortes algorithm. Each example fits the model, and then
# finds the mean predictions for each data point.

runtime_tests <- function(nDigits = 3){ # nDigits is number of decimal places to round to for run times
  
  # https://johnpaulgosling.github.io/AddiVortes/articles/introduction.html
  test1 <- function(){
  
  require(AddiVortes)
    
  X_Boston <- as.matrix(Boston[, 1:13])
  Y_Boston <- as.numeric(as.matrix(Boston[, 14]))
  n <- length(Y_Boston)
  
  # Set a seed for reproducibility
  set.seed(1025)
  
  # Create a training set containing 5/6 of the data
  TrainSet <- sort(sample.int(n, 5 * n / 6))
  
  # The remaining data will be our test set
  TestSet <- setdiff(1:n, TrainSet)
  
  # Run the AddiVortes algorithm on the training data
  start.time.fit <- Sys.time()
  results <- AddiVortes(
    y = Y_Boston[TrainSet],
    x = X_Boston[TrainSet, ],
    m = 200,
    totalMCMCIter = 2000,
    mcmcBurnIn = 200,
    nu = 6,
    q = 0.85,
    k = 3,
    sd = 0.8,
    Omega = 3,
    LambdaRate = 25,
    InitialSigma = "Linear",
    showProgress = FALSE
  )
  end.time.fit <- Sys.time()
  # Generate predictions on the test set
  start.time.preds <- Sys.time()
  preds <- predict(results,
                   X_Boston[TestSet, ],
                   showProgress = FALSE
  )
  end.time.preds <- Sys.time()
  return (c(
    round(end.time.fit-start.time.fit, nDigits),
    round(end.time.preds-start.time.preds,nDigits)
  ))
  }
  
  # https://johnpaulgosling.github.io/AddiVortes/articles/prediction.html
  test2 <- function(){
    # Load the package
    require(AddiVortes)
    
    # --- Generate Training Data ---
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
    
    # Fit the model
    start.time.fit <- Sys.time()
    AModel <- AddiVortes(Y, X, m = 50, showProgress = FALSE)
    end.time.fit <- Sys.time()
    
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
    
    # --- Make Predictions ---
    # Predict the mean response
    start.time.pred <- Sys.time()
    preds <- predict(AModel, testX,
                     showProgress = FALSE
    )
    end.time.pred <- Sys.time()
   
    return(c(
      round(end.time.fit-start.time.fit, nDigits),
      round(end.time.pred-start.time.pred, nDigits)
    ))
  }
  
  # https://johnpaulgosling.github.io/AddiVortes/articles/spherical.html
  test3 <- function(){
    require(AddiVortes)
    
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
    
    # Convert radians to degrees for a readable plot
    lat_deg <- lat * 180 / pi
    lon_deg <- lon * 180 / pi
    
    start.time.fit <- Sys.time()
    fit_sph <- AddiVortes(
      y = y,
      x = x,
      m = 50,
      totalMCMCIter = 500,
      mcmcBurnIn = 100,
      metric = "S", # use great-circle distance for all columns
      showProgress = FALSE
    )
    end.time.fit <- Sys.time()
    
    set.seed(101)
    n_test <- 200
    
    lat_test <- runif(n_test, -pi / 2, pi / 2)
    lon_test <- runif(n_test, -pi, pi)
    
    y_true_test <- 20 * cos(lat_test) + 5 * sin(lon_test)
    y_test <- y_true_test + rnorm(n_test, sd = 2)
    
    x_test <- cbind(lat_test, lon_test)
    
    start.time.pred <- Sys.time()
    preds_sph <- predict(fit_sph, x_test, showProgress = FALSE)
    end.time.pred <- Sys.time()
    
    return(c(
      round(end.time.fit - start.time.fit, nDigits),
      round(end.time.pred - start.time.pred, nDigits)
    ))
  }
  
  # https://johnpaulgosling.github.io/AddiVortes/articles/categorical.html
  test4 <- function(){
    library(AddiVortes)
    
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
    
    x_train <- x[train_idx, ]
    y_train <- y[train_idx]
    x_test <- x[-train_idx, ]
    y_test <- y[-train_idx]
    
    start.time.fit <- Sys.time()
    fit <- AddiVortes(
      y = y_train,
      x = x_train,
      m = 50,
      totalMCMCIter = 500,
      mcmcBurnIn = 100,
      catScaling = 1, # default: binary columns span [0, 1]
      showProgress = FALSE
    )
    end.time.fit <- Sys.time()
    
    start.time.pred <- Sys.time()
    preds <- predict(fit, x_test, showProgress = FALSE)
    end.time.pred <- Sys.time()
    
    return(c(
      round(end.time.fit-start.time.fit, nDigits),
      round(end.time.pred-start.time.pred, nDigits)
    ))
  }
  
  results <- matrix(c(test1(), test2(), test3(), test4()), byrow=TRUE, nrow=4)
  dimnames(results) <- list(c("Test 1", "Test 2", "Test 3", "Test 4"), c("Fit time", "Prediction time"))
  return(addmargins(results))
}