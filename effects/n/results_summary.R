require(tidyverse)
setwd("~/R projects/AddiVortes/effects/n/")
results <- read.csv("addivortes_results.csv")
results <- results[order(results$n), ]

model_iRMSE <- nls(median_in_sample_rmse ~ SSasymp(n, Asym, R0, lrc), data = results[-(1:9),])
model_oRMSE <- nls(median_out_sample_rmse ~ SSasymp(n, Asym, R0, lrc), data = results)
model_fit <- lm(median_fit_time_s ~ n, results)
model_pred <- lm(median_predict_time_s ~ n, results)

pred_grid <- data.frame(n = seq(min(results$n), max(results$n), length.out = (max(results$n) - min(results$n))/100 + 1))
pred_grid$iRMSE <- c(rep(NA, 9), predict(model_iRMSE, pred_grid[-(1:9),,drop=FALSE]))
pred_grid$oRMSE <- predict(model_oRMSE, pred_grid)
pred_grid$fit <- predict(model_fit, pred_grid)
pred_grid$pred <- predict(model_pred, pred_grid)

adj_r_squared <- function(actual, predicted, p) {
  n <- length(actual)
  ss_res <- sum((actual - predicted)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  r_squared <- 1 - (ss_res / ss_tot)
  1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
}

yhat <- predict(model_iRMSE, results[-(1:9),])
r2 <- adj_r_squared(results$median_in_sample_rmse[-(1:9)], yhat, 1)
r2

yhat <- predict(model_oRMSE)
r2 <- adj_r_squared(results$median_out_sample_rmse, yhat, 1)
r2

r2 <- summary(model_fit)$adj.r.squared
r2

r2 <- summary(model_pred)$adj.r.squared
r2

ggplot(results, aes(x=n, y=median_in_sample_rmse)) +
  geom_point() +
  geom_line(data = pred_grid, aes(y = iRMSE), color = "red", linewidth = 1) +
  labs(
    x = "Size of random subset (n)",
    y = "Median in-sample RMSE",
    title = str_wrap("Median in-sample RMSE over multiple iterations of random subsets of size n", width=40)
  )
ggsave("graphs/iRMSE.jpg")

ggplot(results, aes(x=n, y=median_out_sample_rmse)) +
  geom_point() +
  geom_line(data = pred_grid, aes(y = oRMSE), color = "red", linewidth = 1) +
  labs(
    x = "Size of random subset (n)",
    y = "Median out-of-sample RMSE",
    title = str_wrap("Median out-of-sample RMSE over multiple iterations of random subsets of size n", width=40)
  )
ggsave("graphs/oRMSE.jpg")


ggplot(results, aes(x=n, y=median_fit_time_s)) +
  geom_point() +
  geom_line(data = pred_grid, aes(y=fit), color="red", linewidth=1) +
  labs(
    x = "Size of random subset (n)",
    y = "Median fit time",
    title = str_wrap("Median fit time over multiple iterations of random subsets of size n",width=40)
  )
ggsave("graphs/fit.jpg")

ggplot(results, aes(x=n, y=median_predict_time_s)) +
  geom_point() +
  geom_line(data = pred_grid, aes(y=pred), color="red", linewidth=1) +
  labs(
    x = "Size of random subset (n)",
    y = "Median prediction time (over both test and train set)",
    title = str_wrap("Median prediction time over multiple iterations of random subsets of size n",width=40)
  )

ggsave("graphs/pred.jpg")


ggplot(results, aes(x=n, y=median_out_sample_rmse / median_fit_time_s)) +
  geom_point()

model1 <- lm(median_out_sample_rmse / median_fit_time_s ~ I(1/n**3) + I(1/n**2) + I(1/n), results)

plot(results$n, results$median_out_sample_rmse / results$median_fit_time_s)
plot(results$n, predict(model1))
