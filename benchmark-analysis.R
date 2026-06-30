library(tidyverse)

benchmarks_r <- read.csv("r-testlog.csv")
benchmarks_python <- read.csv("py-testlog.csv")

benchmarks_r$Language <- "R"
benchmarks_python$Language <- "Python"

benchmarks <- bind_rows(benchmarks_r, benchmarks_python)

library(tidyverse)

# Function to produce a boxplot for a given column
plot_boxplot <- function(data, column, title, ylab) {
  
  ggplot(data, aes(x = Language, y = .data[[column]], fill = Language)) +
    geom_boxplot(alpha = 0.7, outlier.alpha = 0.6) +
    labs(
      title = title,
      x = "Implementation",
      y = ylab
    ) +
    theme_minimal(base_size = 14) +
    theme(
      legend.position = "none",
      plot.title = element_text(face = "bold")
    )
}
# =======================
# Dataset 1
# =======================

plot_boxplot(benchmarks, "fit1",
             "Dataset 1: Model Fitting Time",
             "Fit Time (s)")

ggsave("fit1.jpg", width=7, height=5)

plot_boxplot(benchmarks, "pred1",
             "Dataset 1: Prediction Time",
             "Prediction Time (s)")

ggsave("pred1.jpg", width=7, height=5)

plot_boxplot(benchmarks, "iRMSE1",
             "Dataset 1: In-Sample RMSE",
             "In-Sample RMSE")

ggsave("irmse1.jpg", width=7, height=5)

plot_boxplot(benchmarks, "oRMSE1",
             "Dataset 1: Out-of-Sample RMSE",
             "Out-of-Sample RMSE")

ggsave("ormse1.jpg", width=7, height=5)

# =======================
# Dataset 2
# =======================

plot_boxplot(benchmarks, "fit2",
             "Dataset 2: Model Fitting Time",
             "Fit Time (s)")

ggsave("fit2.jpg", width=7, height=5)

plot_boxplot(benchmarks, "pred2",
             "Dataset 2: Prediction Time",
             "Prediction Time (s)")

ggsave("pred2.jpg", width=7, height=5)

plot_boxplot(benchmarks, "iRMSE2",
             "Dataset 2: In-Sample RMSE",
             "In-Sample RMSE")

ggsave("irmse2.jpg", width=7, height=5)

plot_boxplot(benchmarks, "oRMSE2",
             "Dataset 2: Out-of-Sample RMSE",
             "Out-of-Sample RMSE")

ggsave("ormse2.jpg", width=7, height=5)

# =======================
# Dataset 3
# =======================

plot_boxplot(benchmarks, "fit3",
             "Dataset 3: Model Fitting Time",
             "Fit Time (s)")

ggsave("fit3.jpg", width=7, height=5)

plot_boxplot(benchmarks, "pred3",
             "Dataset 3: Prediction Time",
             "Prediction Time (s)")

ggsave("pred3.jpg", width=7, height=5)

plot_boxplot(benchmarks, "iRMSE3",
             "Dataset 3: In-Sample RMSE",
             "In-Sample RMSE")

ggsave("irmse3.jpg", width=7, height=5)

plot_boxplot(benchmarks, "oRMSE3",
             "Dataset 3: Out-of-Sample RMSE",
             "Out-of-Sample RMSE")

ggsave("ormse3.jpg", width=7, height=5)

# =======================
# Dataset 4
# =======================

plot_boxplot(benchmarks, "fit4",
             "Dataset 4: Model Fitting Time",
             "Fit Time (s)")

ggsave("fit4.jpg", width=7, height=5)

plot_boxplot(benchmarks, "pred4",
             "Dataset 4: Prediction Time",
             "Prediction Time (s)")

ggsave("pred4.jpg", width=7, height=5)

plot_boxplot(benchmarks, "iRMSE4",
             "Dataset 4: In-Sample RMSE",
             "In-Sample RMSE")

ggsave("irmse4.jpg", width=7, height=5)

plot_boxplot(benchmarks, "oRMSE4",
             "Dataset 4: Out-of-Sample RMSE",
             "Out-of-Sample RMSE")

ggsave("ormse4.jpg", width=7, height=5)

# Summary statistics

# Percentage difference function
pct_diff <- function(r, py) {
  100 * (py - r) / r
}

# ---------------------------
# Table 1: Means (Fit & Pred)
# ---------------------------

fit_pred_means <- data.frame(
  Test = paste("Test", 1:4),
  
  `Fit Mean (R)` = sapply(1:4, function(i) mean(benchmarks[benchmarks$Language=="R", paste0("fit",i)], na.rm=TRUE)),
  `Fit Mean (Py)` = sapply(1:4, function(i) mean(benchmarks[benchmarks$Language=="Python", paste0("fit",i)], na.rm=TRUE)),
  `Fit % Diff` = mapply(pct_diff,
                        sapply(1:4, function(i) mean(benchmarks[benchmarks$Language=="R", paste0("fit",i)], na.rm=TRUE)),
                        sapply(1:4, function(i) mean(benchmarks[benchmarks$Language=="Python", paste0("fit",i)], na.rm=TRUE))),
  
  `Pred Mean (R)` = sapply(1:4, function(i) mean(benchmarks[benchmarks$Language=="R", paste0("pred",i)], na.rm=TRUE)),
  `Pred Mean (Py)` = sapply(1:4, function(i) mean(benchmarks[benchmarks$Language=="Python", paste0("pred",i)], na.rm=TRUE)),
  `Pred % Diff` = mapply(pct_diff,
                         sapply(1:4, function(i) mean(benchmarks[benchmarks$Language=="R", paste0("pred",i)], na.rm=TRUE)),
                         sapply(1:4, function(i) mean(benchmarks[benchmarks$Language=="Python", paste0("pred",i)], na.rm=TRUE)))
)

fit_pred_means[-1] <- round(fit_pred_means[-1], 3)

# ---------------------------
# Table 2: SDs (Fit & Pred)
# ---------------------------

fit_pred_sds <- data.frame(
  Test = paste("Test", 1:4),
  
  `Fit SD (R)` = sapply(1:4, function(i) sd(benchmarks[benchmarks$Language=="R", paste0("fit",i)], na.rm=TRUE)),
  `Fit SD (Py)` = sapply(1:4, function(i) sd(benchmarks[benchmarks$Language=="Python", paste0("fit",i)], na.rm=TRUE)),
  `Fit % Diff` = mapply(pct_diff,
                        sapply(1:4, function(i) sd(benchmarks[benchmarks$Language=="R", paste0("fit",i)], na.rm=TRUE)),
                        sapply(1:4, function(i) sd(benchmarks[benchmarks$Language=="Python", paste0("fit",i)], na.rm=TRUE))),
  
  `Pred SD (R)` = sapply(1:4, function(i) sd(benchmarks[benchmarks$Language=="R", paste0("pred",i)], na.rm=TRUE)),
  `Pred SD (Py)` = sapply(1:4, function(i) sd(benchmarks[benchmarks$Language=="Python", paste0("pred",i)], na.rm=TRUE)),
  `Pred % Diff` = mapply(pct_diff,
                         sapply(1:4, function(i) sd(benchmarks[benchmarks$Language=="R", paste0("pred",i)], na.rm=TRUE)),
                         sapply(1:4, function(i) sd(benchmarks[benchmarks$Language=="Python", paste0("pred",i)], na.rm=TRUE)))
)

fit_pred_sds[-1] <- round(fit_pred_sds[-1], 3)

# -----------------------------
# Table 3: Means (iRMSE & oRMSE)
# -----------------------------

rmse_means <- data.frame(
  Test = paste("Test", 1:4),
  
  `iRMSE Mean (R)` = sapply(1:4, function(i) mean(benchmarks[benchmarks$Language=="R", paste0("iRMSE",i)], na.rm=TRUE)),
  `iRMSE Mean (Py)` = sapply(1:4, function(i) mean(benchmarks[benchmarks$Language=="Python", paste0("iRMSE",i)], na.rm=TRUE)),
  `iRMSE % Diff` = mapply(pct_diff,
                          sapply(1:4, function(i) mean(benchmarks[benchmarks$Language=="R", paste0("iRMSE",i)], na.rm=TRUE)),
                          sapply(1:4, function(i) mean(benchmarks[benchmarks$Language=="Python", paste0("iRMSE",i)], na.rm=TRUE))),
  
  `oRMSE Mean (R)` = sapply(1:4, function(i) mean(benchmarks[benchmarks$Language=="R", paste0("oRMSE",i)], na.rm=TRUE)),
  `oRMSE Mean (Py)` = sapply(1:4, function(i) mean(benchmarks[benchmarks$Language=="Python", paste0("oRMSE",i)], na.rm=TRUE)),
  `oRMSE % Diff` = mapply(pct_diff,
                          sapply(1:4, function(i) mean(benchmarks[benchmarks$Language=="R", paste0("oRMSE",i)], na.rm=TRUE)),
                          sapply(1:4, function(i) mean(benchmarks[benchmarks$Language=="Python", paste0("oRMSE",i)], na.rm=TRUE)))
)

rmse_means[-1] <- round(rmse_means[-1], 3)

# ---------------------------
# Table 4: SDs (iRMSE & oRMSE)
# ---------------------------

rmse_sds <- data.frame(
  Test = paste("Test", 1:4),
  
  `iRMSE SD (R)` = sapply(1:4, function(i) sd(benchmarks[benchmarks$Language=="R", paste0("iRMSE",i)], na.rm=TRUE)),
  `iRMSE SD (Py)` = sapply(1:4, function(i) sd(benchmarks[benchmarks$Language=="Python", paste0("iRMSE",i)], na.rm=TRUE)),
  `iRMSE % Diff` = mapply(pct_diff,
                          sapply(1:4, function(i) sd(benchmarks[benchmarks$Language=="R", paste0("iRMSE",i)], na.rm=TRUE)),
                          sapply(1:4, function(i) sd(benchmarks[benchmarks$Language=="Python", paste0("iRMSE",i)], na.rm=TRUE))),
  
  `oRMSE SD (R)` = sapply(1:4, function(i) sd(benchmarks[benchmarks$Language=="R", paste0("oRMSE",i)], na.rm=TRUE)),
  `oRMSE SD (Py)` = sapply(1:4, function(i) sd(benchmarks[benchmarks$Language=="Python", paste0("oRMSE",i)], na.rm=TRUE)),
  `oRMSE % Diff` = mapply(pct_diff,
                          sapply(1:4, function(i) sd(benchmarks[benchmarks$Language=="R", paste0("oRMSE",i)], na.rm=TRUE)),
                          sapply(1:4, function(i) sd(benchmarks[benchmarks$Language=="Python", paste0("oRMSE",i)], na.rm=TRUE)))
)

rmse_sds[-1] <- round(rmse_sds[-1], 3)

write_table <- function(df, name) {
  library(knitr)
  tex <- kable(df, format = "latex", digits = 3)
  writeLines(tex, paste0(name, ".tex"))
}

colnames(fit_pred_means) <- c(
  "Test",
  "Fit Mean (R)",
  "Fit Mean (Python)",
  "Fit % Diff",
  "Pred Mean (R)",
  "Pred Mean (Python)",
  "Pred % Diff"
)
colnames(fit_pred_sds) <- c(
  "Test",
  "Fit SD (R)",
  "Fit SD (Python)",
  "Fit % Diff",
  "Pred SD (R)",
  "Pred SD (Python)",
  "Pred % Diff"
)
colnames(rmse_means) <- c(
  "Test",
  "iRMSE Mean (R)",
  "iRMSE Mean (Python)",
  "iRMSE % Diff",
  "oRMSE Mean (R)",
  "oRMSE Mean (Python)",
  "oRMSE % Diff"
)
colnames(rmse_sds) <- c(
  "Test",
  "iRMSE SD (R)",
  "iRMSE SD (Python)",
  "iRMSE % Diff",
  "oRMSE SD (R)",
  "oRMSE SD (Python)",
  "oRMSE % Diff"
)

write_table(fit_pred_means, "fit_pred_means")
write_table(fit_pred_sds, "fit_pred_sds")
write_table(rmse_means, "rmse_means")
write_table(rmse_sds, "rmse_sds")