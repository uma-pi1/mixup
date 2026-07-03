# =============================================================================
# Conduct t-tests.
# =============================================================================

# ===
# Step 1: Add baseline values per SE type.
# ===

baseline_long <- data_long |>
  filter(method_name == "baseline") |>
  rename(
    baseline_acc = acc,
    baseline_se = se,
    baseline_n = n
  ) |>
  select(se_type, baseline_acc, baseline_se, baseline_n)

# ===
# Step 2: Compute Welch's t-tests.
# ===

t_test_results <- data_long |>
  filter(method_name != "baseline") |>
  left_join(baseline_long, by = "se_type") |>
  mutate(
    se_diff = sqrt(se^2 + baseline_se^2),
    t_stat = (acc - baseline_acc) / se_diff,
    df = compute_df(se, baseline_se, n, baseline_n),
    p = 2 * (1 - pt(abs(t_stat), df))
  ) |>
  select(method_name, se_type, acc, t_stat, df, p)

# ===
# Step 3: Convert back to wide format (for output).
# ===

t_test_wide <- t_test_results |>
  pivot_wider(
    names_from = se_type,
    values_from = c(t_stat, df, p)
  ) |>
  select(method_name, acc, p_se_a1, p_se_a2, p_se_a3)
t_test_wide
