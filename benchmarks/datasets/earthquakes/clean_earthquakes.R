# =========================================================
# AddiVortes test data prep: USGS earthquake catalog cleanup
# =========================================================

# ---- 0. Load libraries ----
# base R is sufficient for this pipeline; no extra packages required

# ---- 1. Read in the raw CSV ----
df <- read.csv("raw_data.csv", stringsAsFactors = FALSE)

cat("Starting rows:", nrow(df), "\n")
cat("Starting columns:", ncol(df), "\n\n")

# ---- 2. Filter to real earthquakes only ----
# 'type' includes a small number of mining explosions, landslides, etc.
# which are noise for an earthquake-magnitude model.
before_n <- nrow(df)
df <- df[df$type == "earthquake", ]
cat("After filtering to type == 'earthquake':", nrow(df),
    "rows (dropped", before_n - nrow(df), ")\n")

# ---- 3. Drop redundant / non-covariate columns ----
# - 'type' is now constant (all 'earthquake') -> no information left
# - 'locationSource' is ~99.99% identical to 'net' -> redundant
# - 'magSource' is ~99.7% identical to 'net' -> redundant
# - 'time' dropped for this non-time-series test (re-add + engineer
#    year/month/hour if you want temporal covariates later)
# - 'gap' removed from the benchmark covariate set as requested
drop_cols <- c("type", "locationSource", "magSource", "time", "gap")
df <- df[, !(names(df) %in% drop_cols)]
cat("Dropped columns:", paste(drop_cols, collapse = ", "), "\n")
cat("Remaining columns:", paste(names(df), collapse = ", "), "\n\n")

# ---- 4. Drop rows with any remaining missing values ----
# Missingness across the numeric quality/error columns is low (~5.6%)
# and concentrated in a subset of 'ml' and 'mw' events, so row deletion
# is cheap and keeps the dataset fully complete.
before_n <- nrow(df)
df_clean <- df[complete.cases(df), ]
dropped_n <- before_n - nrow(df_clean)
cat("After dropping rows with missing values:", nrow(df_clean),
    "rows (dropped", dropped_n,
    sprintf("[%.1f%%]", 100 * dropped_n / before_n), ")\n\n")

# ---- 5. Restrict to the benchmark subset used in this session ----
# Keep only reviewed body-wave magnitude measurements for the benchmark.
before_n <- nrow(df_clean)
df_clean <- df_clean[df_clean$magType == "mb" & df_clean$status == "reviewed", ]
cat("After filtering to magType == 'mb' and status == 'reviewed':",
    nrow(df_clean), "rows (dropped", before_n - nrow(df_clean), ")\n\n")

# ---- 6. Drop now-constant filtering columns ----
# After subsetting, 'magType' and 'status' are constant and carry no information.
df_clean <- df_clean[, !(names(df_clean) %in% c("magType", "status"))]
cat("After dropping constant columns magType and status:",
    nrow(df_clean), "rows and", ncol(df_clean), "columns\n\n")

# ---- 7. Sanity checks ----
cat("Final shape:", nrow(df_clean), "rows x", ncol(df_clean), "columns\n")
cat("Total missing values remaining:", sum(is.na(df_clean)), "\n\n")

cat("Categorical cardinality:\n")
for (col in c("net")) {
  cat(" ", col, "->", length(unique(df_clean[[col]])), "levels\n")
}

# ---- 8. Write out the cleaned file ----
write.csv(df_clean, "earthquakes.csv", row.names = FALSE)
cat("\nSaved cleaned file to earthquakes.csv\n")
