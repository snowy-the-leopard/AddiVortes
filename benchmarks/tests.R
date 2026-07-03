# This function uses code from the articles below to give a benchmark test for run times on
# 4 different examples using the AddiVortes algorithm. Each example fits the model, and then
# finds the mean predictions for each data point.

tests <- function(nDigits = 3, nTimes = 1){ # nDigits is number of decimal places to round to for run times
  Sys.setenv(
    OMP_NUM_THREADS = "1",
    MKL_NUM_THREADS = "1",
    OPENBLAS_NUM_THREADS = "1"
  )
  library(RhpcBLASctl)
  blas_set_num_threads(1)
  omp_set_num_threads(1)
  library(RhpcBLASctl)
  require(AddiVortes)
  pbapply::pboptions(type = "none") # Disable progress bars for predictions
  # https://johnpaulgosling.github.io/AddiVortes/articles/introduction.html
  test1 <- function(){
  
  x_train <- as.matrix(read.csv("./benchmarks/datasets/boston/x_train.csv"))
  y_train <- as.matrix(read.csv("./benchmarks/datasets/boston/y_train.csv"))
  x_test <- as.matrix(read.csv("./benchmarks/datasets/boston/x_test.csv"))
  y_test <- as.matrix(read.csv("./benchmarks/datasets/boston/y_test.csv"))
  
  # Run the AddiVortes algorithm on the training data
  start.time.fit <- Sys.time()
  results <- AddiVortes(
    y = y_train,
    x = x_train,
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
                   x_test,
                   showProgress = FALSE
  )
  end.time.preds <- Sys.time()

  return (c(
    round(end.time.fit-start.time.fit, nDigits),
    round(end.time.preds-start.time.preds,nDigits),
    round(results$inSampleRmse, nDigits),
    round(sqrt(mean((y_test - preds)^2)), nDigits)
  ))
  }
  
  # https://johnpaulgosling.github.io/AddiVortes/articles/prediction.html
  test2 <- function(){

    x_train <- as.matrix(read.csv("./benchmarks/datasets/synthetic/x_train.csv"))
    y_train <- as.matrix(read.csv("./benchmarks/datasets/synthetic/y_train.csv"))
    x_test <- as.matrix(read.csv("./benchmarks/datasets/synthetic/x_test.csv"))
    y_test <- as.matrix(read.csv("./benchmarks/datasets/synthetic/y_test.csv"))
    
    # Fit the model
    start.time.fit <- Sys.time()
    AModel <- AddiVortes(y_train, x_train, m = 50, showProgress = FALSE)
    end.time.fit <- Sys.time()

    # --- Make Predictions ---
    # Predict the mean response
    start.time.pred <- Sys.time()
    preds <- predict(AModel, x_test,
                     showProgress = FALSE
    )
    end.time.pred <- Sys.time()
   
    return(c(
      round(end.time.fit-start.time.fit, nDigits),
      round(end.time.pred-start.time.pred, nDigits),
      round(AModel$inSampleRmse, nDigits),
      round(sqrt(mean((y_test - preds)^2)), nDigits)
    ))
  }
  
  # https://johnpaulgosling.github.io/AddiVortes/articles/spherical.html
  test3 <- function(){
    
    x_train <- as.matrix(read.csv("./benchmarks/datasets/spherical/x_train.csv"))
    y_train <- as.matrix(read.csv("./benchmarks/datasets/spherical/y_train.csv"))
    x_test <- as.matrix(read.csv("./benchmarks/datasets/spherical/x_test.csv"))
    y_test <- as.matrix(read.csv("./benchmarks/datasets/spherical/y_test.csv"))
    
    start.time.fit <- Sys.time()
    fit_sph <- AddiVortes(
      y = y_train,
      x = x_train,
      m = 50,
      totalMCMCIter = 500,
      mcmcBurnIn = 100,
      metric = "S", # use great-circle distance for all columns
      showProgress = FALSE
    )
    end.time.fit <- Sys.time()
    
    start.time.pred <- Sys.time()
    preds_sph <- predict(fit_sph, x_test, showProgress = FALSE)
    end.time.pred <- Sys.time()
    
    return(c(
      round(end.time.fit - start.time.fit, nDigits),
      round(end.time.pred - start.time.pred, nDigits),
      round(fit_sph$inSampleRmse, nDigits),
      round(sqrt(mean((y_test - preds_sph)^2)), nDigits)
    ))
  }
  
  # https://johnpaulgosling.github.io/AddiVortes/articles/categorical.html
  test4 <- function(){
    
    x_train <- read.csv("./benchmarks/datasets/categorical/x_train.csv")
    y_train <- as.matrix(read.csv("./benchmarks/datasets/categorical/y_train.csv"))
    x_test <- read.csv("./benchmarks/datasets/categorical/x_test.csv")
    y_test <- as.matrix(read.csv("./benchmarks/datasets/categorical/y_test.csv"))
    
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
      round(end.time.pred-start.time.pred, nDigits),
      round(fit$inSampleRmse, nDigits),
      round(sqrt(mean((y_test - preds)^2)), nDigits)
    ))
  }

  times <- list()
  errors <- list()
  
  to_csv <- function(...) {
    paste(paste(..., sep = "", collapse = ","), "\n")
  }
  
  for (i in 1:nTimes){
    results <- list(test1(), test2(), test3(), test4())
    print(paste("Iteration ", i, " completed"))
    times[[i]] <- matrix(c(results[[1]][1:2], results[[2]][1:2], results[[3]][1:2], results[[4]][1:2]), byrow=TRUE, nrow=4)
    errors[[i]] <- matrix(c(results[[1]][3:4], results[[2]][3:4], results[[3]][3:4], results[[4]][3:4]), byrow=TRUE, nrow=4)
    cat(
      paste(unlist(results), collapse = ","),
      "\n",
      file = "benchmarks/r-testlog.csv",
      append = TRUE
    )
  }

  avg_time = round(Reduce("+", times) / nTimes, nDigits)
  avg_errors = round(Reduce("+", errors) / nTimes, nDigits)
  dimnames(avg_time) <- list(c("Test 1", "Test 2", "Test 3", "Test 4"), c("Fit time", "Prediction time"))
  dimnames(avg_errors) <- list(c("Test 1", "Test 2", "Test 3", "Test 4"), c("In-sample RMSE", "Out-of-sample RMSE"))
  return(list(times=addmargins(avg_time), errors=addmargins(avg_errors)))
}

tests(3,1)