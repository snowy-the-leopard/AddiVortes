library(tidyverse)

script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- script_args[grepl("^--file=", script_args)]
script_dir <- "~/R projects/AddiVortes/effects/params"
setwd(script_dir)

results_path <- file.path(script_dir, "settings_results.csv")
output_dir <- file.path(script_dir, "graphs")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

results <- read.csv(results_path, check.names = FALSE)

parameters <- c("m", "nu", "q", "omega", "lambda", "mcmcIter", "mcmcBurnin")
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
metric_filenames <- c(
  "Fit time" = "fit.jpg",
  "Prediction time" = "pred.jpg",
  "In-sample RMSE" = "iRMSE.jpg",
  "Out-of-sample RMSE" = "oRMSE.jpg"
)

build_plot_data <- function(results, parameter, metric_name) {
  results %>%
    filter(changed_parameter == parameter) %>%
    select(all_of(c(parameter, metric_name))) %>%
    rename(x = !!parameter, y = !!metric_name) %>%
    mutate(
      x = as.numeric(x),
      y = as.numeric(y)
    )
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

  plot <- ggplot(plot_data, aes(x = x, y = y)) +
    geom_point(size = 1.2) +
    labs(
      x = parameter,
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

custom_fit_models <- list(
  `m|Fit time` = y ~ x,
  `m|Prediction time` = y ~ x,
  `m|In-sample RMSE` = asymptotic_fit,
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
