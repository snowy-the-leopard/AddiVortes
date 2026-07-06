library(tidyverse)

benchmarks_r <- read.csv("benchmarks/r-testlog.csv")
benchmarks_python <- read.csv("benchmarks/py-testlog.csv")

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

write_table(fit_pred_means, "benchmarks/fit_pred_means")
write_table(fit_pred_sds, "benchmarks/fit_pred_sds")
write_table(rmse_means, "benchmarks/rmse_means")
write_table(rmse_sds, "benchmarks/rmse_sds")

# Appendix A

fit_pred_ses <- fit_pred_sds[,-c(1,4, 7)]/sqrt(106)
names(fit_pred_ses) <- gsub("SD", "SE", names(fit_pred_ses))
fit_pred_ses <- fit_pred_ses %>% mutate(Fit.SE..Comb. = sqrt(Fit.SE..R.^2 + Fit.SE..Py.^2),
                                Pred.SE..Comb. = sqrt(Pred.SE..R.^2 + Pred.SE..Py.^2))
fit_pred_ses <- data.frame(Test=c("Test 1", "Test 2", "Test 3", "Test 4")) %>% cbind(fit_pred_ses)

rmse_ses <- rmse_sds[,-c(1,4, 7)]/sqrt(106)
names(rmse_ses) <- gsub("SD", "SE", names(rmse_ses))
rmse_ses <- rmse_ses %>% mutate(iRMSE.SE.Comb. = sqrt(iRMSE.SE..R.^2 + iRMSE.SE..Py.^2),
                                oRMSE.SE.Comb. = sqrt(oRMSE.SE..R.^2 + oRMSE.SE..Py.^2))
rmse_ses <- data.frame(Test=c("Test 1", "Test 2", "Test 3", "Test 4")) %>% cbind(rmse_ses)
                

write_table(fit_pred_ses, "benchmarks/fit_pred_ses")
write_table(rmse_ses, "benchmarks/rmse_ses")

fit_pred_sig <- cbind(fit_pred_means %>% 
  mutate(Fit.Diff = abs(Fit.Mean..R.-Fit.Mean..Py.), Pred.Diff = abs(Pred.Mean..R. - Pred.Mean..Py.)) %>% 
  select(Test, Fit.Diff, Pred.Diff), fit_pred_ses[,c(6, 7)]) %>% 
  mutate(Fit.Diff.SEs = Fit.Diff/Fit.SE..Comb., Pred.Diff.SEs = Pred.Diff/Pred.SE..Comb.)

rmse_sig <- cbind(rmse_means %>% 
                        mutate(iRMSE.Diff = abs(iRMSE.Mean..R.-iRMSE.Mean..Py.), oRMSE.Diff = abs(oRMSE.Mean..R. - oRMSE.Mean..Py.)) %>% 
                        select(Test, iRMSE.Diff, oRMSE.Diff), rmse_ses[,c(6, 7)]) %>% 
  mutate(iRMSE.Diff.SEs = iRMSE.Diff/iRMSE.SE.Comb., oRMSE.Diff.SEs = oRMSE.Diff/oRMSE.SE.Comb.)

write_table(fit_pred_sig, "benchmarks/fit_pred_sig")
write_table(rmse_sig, "benchmarks/rmse_sig")
