import numpy as np
from addivortes import AddiVortesRegressor

rng = np.random.default_rng(123)
X = rng.normal(size=(100, 4))
y = X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.2, size=100)

model = AddiVortesRegressor(
    n_tessellations=25,
    total_mcmc_iter=300,
    burn_in=100,
    random_state=123,
)
model.fit(X, y)

predictions = model.predict(X[:5])
intervals = model.predict(X[:5], kind="quantile", quantiles=(0.025, 0.975))

print(predictions)
print(intervals)

model.plot(X, y, which=(1, 2, 3, 4), show=True)