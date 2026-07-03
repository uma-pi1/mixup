# Helper methods #######################################################################

compute_df <- function(se1, se2, n1, n2) {
  # Compute standard deviations.
  sd1 <- se1 * sqrt(n1)
  sd2 <- se2 * sqrt(n2)

  # Compute WS-equation fraction.
  numerator <- (sd1^2 / n1 + sd2^2 / n2)^2
  denominator <- sd1^4 / (n1^2 * (n1 - 1)) + sd2^4 / (n2^2 * (n2 - 1))

  # Return (approx.) df.
  numerator / denominator
}

# Convenience method for:
# (i):   x 100
# (ii):  Round to two digits
# (iii): Keep trailing zeros
fmt_number <- function(x) {
  format(round(x * 100, 2), nsmall = 2)
}
fmt_number(0.544)

# Global vars.  ########################################################################

# Orders for tables.
dataset_order <- c("MUTAG", "ENZYMES", "IMDB-BINARY", "PROTEINS", "IMDB-MULTI", "NCI1")
method_dict <- c(
  "baseline" = "Baseline",
  "emb_mixup" = "Emb-M.",
  "g_mixup" = "G-Mixup",
  "geomix" = "GeoMix",
  "if_mixup" = "If-Mixup",
  "s_mixup" = "S-Mixup",
  "submix" = "SubMix",
  "ged_mixup" = "GED-M."
)
method_order <- names(method_dict)
model_order <- c("GCN", "GIN")

# Welch t-test #########################################################################

t_test_from_summary <- function(mean1, mean2, var1, var2, n1, n2) {
  # difference in means
  diff <- mean1 - mean2
  
  # standard error under Welch
  se_diff <- sqrt(var1 / n1 + var2 / n2)
  
  # t statistic
  t_stat <- diff / se_diff
  
  # Welch–Satterthwaite degrees of freedom
  df <- (var1 / n1 + var2 / n2)^2 /
    ((var1^2 / (n1^2 * (n1 - 1))) + (var2^2 / (n2^2 * (n2 - 1))))
  
  # two-sided p-value
  p_val <- 2 * stats::pt(-abs(t_stat), df = df)
  
  tibble(
    diff = diff,
    t = t_stat,
    df = df,
    p_value = p_val
  )
}
