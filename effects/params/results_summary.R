library(tidyverse)

script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- script_args[grepl("^--file=", script_args)]
script_dir <- "~/R projects/AddiVortes/effects/params"
setwd(script_dir)

results_paths <- file.path(
  script_dir,
  c("settings_results.csv", "settings_results2.csv")
)
output_dir <- file.path(script_dir, "graphs")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

results <- purrr::map_dfr(
  results_paths,
  ~ read.csv(.x, check.names = FALSE)
)

parameters <- c("m", "nu", "q", "omega", "lambda", "mcmcIter", "mcmcBurnin")
default_parameter_values <- c(
  m = 200,
  nu = 6,
  q = 0.85,
  omega = 3,
  lambda = 25,
  mcmcIter = 1200,
  mcmcBurnin = 200
)
metric_columns <- c(
  "Fit time",
  "Prediction time",
  "In-sample RMSE",
  "Out-of-sample RMSE"
)

required_columns <- c("changed_parameter", parameters, metric_columns)
missing_columns <- setdiff(required_columns, names(results))
if (length(missing_columns) > 0) {
  stop("Missing columns: ", paste(missing_columns, collapse = ", "))
}

metric_labels <- c(
  "Fit time" = "Fit time (s)",
  "Prediction time" = "Prediction time (s)",
  "In-sample RMSE" = "In-sample RMSE",
  "Out-of-sample RMSE" = "Out-of-sample RMSE"
)
x_labels <- c(
  mcmcBurnin = "MCMC burn-in proportion"
)
metric_filenames <- c(
  "Fit time" = "fit.jpg",
  "Prediction time" = "pred.jpg",
  "In-sample RMSE" = "iRMSE.jpg",
  "Out-of-sample RMSE" = "oRMSE.jpg"
)

build_plot_data <- function(results, parameter, metric_name) {
  plot_data <- results %>%
    filter(changed_parameter == parameter) %>%
    select(all_of(unique(c(parameter, "mcmcIter", metric_name)))) %>%
    mutate(across(everything(), as.numeric)) %>%
    rename(x = !!parameter, y = !!metric_name) %>%
    mutate(x = as.numeric(x), y = as.numeric(y))

  if (parameter == "mcmcBurnin") {
    plot_data <- plot_data %>% mutate(x = x / mcmcIter)
  } else if (parameter == "m") {
    max_x <- max(plot_data$x, na.rm = TRUE)
    low_max <- min(400, max_x)
    low_targets <- seq(
      min(plot_data$x, na.rm = TRUE),
      low_max,
      length.out = min(5, nrow(plot_data))
    )
    high_targets <- if (max_x >= 500) seq(500, max_x, by = 100) else numeric(0)
    targets <- unique(c(low_targets, high_targets))

    plot_data <- purrr::map_dfr(
      targets,
      ~ plot_data %>% slice_min(abs(x - .x), n = 1, with_ties = FALSE)
    ) %>%
      distinct(x, .keep_all = TRUE) %>%
      arrange(x)
  }

  plot_data
}

add_best_fit_line <- function(plot, plot_data, fit_model = NULL, color = "tomato") {
  if (is.null(fit_model)) {
    return(plot)
  }

  if (inherits(fit_model, "formula")) {
    fit <- lm(fit_model, data = plot_data)
    x_grid <- tibble(
      x = seq(
        min(plot_data$x, na.rm = TRUE),
        max(plot_data$x, na.rm = TRUE),
        length.out = 200
      )
    )
    fit_line <- x_grid %>%
      mutate(y = predict(fit, newdata = x_grid))

    plot +
      geom_line(
        data = fit_line,
        aes(x = x, y = y),
        color = color,
        linewidth = 1
      )
  } else if (is.function(fit_model)) {
    fit_line <- fit_model(plot_data)
    if (is.null(fit_line)) {
      return(plot)
    }

    if (!is.data.frame(fit_line) || !all(c("x", "y") %in% names(fit_line))) {
      stop("fit_model function must return a data.frame with x and y columns.")
    }

    plot +
      geom_line(
        data = fit_line,
        aes(x = x, y = y),
        color = color,
        linewidth = 1
      )
  } else {
    stop("fit_model must be NULL, a formula, or a function returning a data.frame with x and y columns.")
  }
}

make_metric_plot <- function(results, parameter, metric_name, fit_model = NULL) {
  plot_data <- build_plot_data(results, parameter, metric_name)

  xintercept <- default_parameter_values[[parameter]]
  x_label <- parameter
  if (parameter == "mcmcBurnin") {
    xintercept <- default_parameter_values[[parameter]] /
      default_parameter_values[["mcmcIter"]]
    x_label <- x_labels[[parameter]]
  }

  plot <- ggplot(plot_data, aes(x = x, y = y)) +
    geom_point(size = 1.2) +
    geom_vline(
      xintercept = xintercept,
      linetype = "dotted"
    ) +
    labs(
      x = x_label,
      y = metric_labels[[metric_name]],
      title = paste(metric_labels[[metric_name]], "against", parameter)
    ) +
    theme_minimal()

  add_best_fit_line(plot, plot_data, fit_model = fit_model)
}

save_metric_plot <- function(plot, parameter, metric_name, output_dir) {
  parameter_output_dir <- file.path(output_dir, parameter)
  dir.create(parameter_output_dir, showWarnings = FALSE, recursive = TRUE)

  ggsave(
    file.path(parameter_output_dir, metric_filenames[[metric_name]]),
    plot,
    width = 8,
    height = 5,
    units = "in",
    dpi = 300
  )
}

asymptotic_fit <- function(plot_data) {
  fit <- tryCatch(
    nls(
      y ~ SSasymp(x, Asym, R0, lrc),
      data = plot_data,
      control = nls.control(maxiter = 100)
    ),
    error = function(e) NULL
  )

  if (is.null(fit)) {
    return(NULL)
  }

  x_grid <- data.frame(x = seq(
    min(plot_data$x, na.rm = TRUE),
    max(plot_data$x, na.rm = TRUE),
    length.out = 200
  ))

  data.frame(
    x = x_grid$x,
    y = predict(fit, newdata = x_grid)
  )
}

zero_asymptote_fit <- function(plot_data) {
  positive_data <- plot_data %>% filter(y > 0)
  if (nrow(positive_data) < 2 || n_distinct(positive_data$x) < 2) {
    return(NULL)
  }

  positive_data <- positive_data %>% filter(x > 0)
  if (nrow(positive_data) < 2) {
    return(NULL)
  }

  fit <- tryCatch(
    lm(log(y) ~ log(x), data = positive_data),
    error = function(e) NULL
  )
  if (is.null(fit)) {
    return(NULL)
  }

  coefficients <- coef(fit)
  if (any(!is.finite(coefficients)) || coefficients[[2]] >= 0) {
    return(NULL)
  }

  x_grid <- data.frame(x = seq(
    min(plot_data$x, na.rm = TRUE),
    max(plot_data$x, na.rm = TRUE),
    length.out = 200
  ))

  data.frame(
    x = x_grid$x,
    y = exp(predict(fit, newdata = data.frame(x = x_grid$x)))
  )
}

custom_fit_models <- list(
  `m|Fit time` = y ~ x,
  `m|Prediction time` = y ~ x,
  `m|In-sample RMSE` = zero_asymptote_fit,
  `m|Out-of-sample RMSE` = asymptotic_fit,
  `mcmcIter|Fit time` = y ~ x,
  `mcmcIter|Prediction time` = y ~ x,
  `mcmcIter|In-sample RMSE` = asymptotic_fit,
  `mcmcIter|Out-of-sample RMSE` = asymptotic_fit,
  `lambda|Fit time` = y ~ x,
  `lambda|Prediction time` = y ~ x
)

plot_specs <- expand.grid(
  parameter = parameters,
  metric_name = metric_columns,
  stringsAsFactors = FALSE
)

for (i in seq_len(nrow(plot_specs))) {
  parameter <- plot_specs$parameter[i]
  metric_name <- plot_specs$metric_name[i]
  plot_key <- paste(parameter, metric_name, sep = "|")
  fit_model <- custom_fit_models[[plot_key]]

  plot <- make_metric_plot(
    results = results,
    parameter = parameter,
    metric_name = metric_name,
    fit_model = fit_model
  )

  save_metric_plot(plot, parameter, metric_name, output_dir)
}

cat("Created plots in", output_dir, "\n")
