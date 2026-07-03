# Testing ----------------------------------------------------------------------

data_grouped <- data_final_tmp |> 
  mutate(group = case_when(
    method_name %in% c("submix", "if_mixup", "ged_mixup") ~ "better",
    method_name %in% c("s_mixup", "geomix", "g_mixup", "emb_mixup") ~ "worse"
  )) |> 
  filter(!is.na(group))
data_grouped

group_stats <- data_grouped |> 
  group_by(group) |> 
  summarize(
    acc_group = mean(acc),
    var_within_group_a1 = mean(var_a1),
    var_within_group_a2 = mean(var_a2),
    var_within_group_a3 = mean(var_a3),
    var_between_groups = var(acc),
    n = n() * first(n),
    .groups = "drop"
  ) |> 
  mutate(
    var_group_a1 = var_within_group_a1 + var_between_groups,
    var_group_a2 = var_within_group_a2,
    var_group_a3 = var_within_group_a3
  ) |> 
  mutate(
    se_group_a1 = sqrt(var_group_a1 / n),
    se_group_a2 = sqrt(var_group_a2 / n),
    se_group_a3 = sqrt(var_group_a3 / n)
  ) |> 
  select(group, acc_group, se_group_a1, se_group_a2, se_group_a3, n)
group_stats

# Vector of SE column names
se_cols <- c("se_group_a1", "se_group_a2", "se_group_a3")

# Initialize results
results_welch <- data.frame(assumption = character(),
                            diff_mean = numeric(),
                            se_diff = numeric(),
                            t_value = numeric(),
                            df = numeric(),
                            p_value = numeric(),
                            stringsAsFactors = FALSE)

n_better <- group_stats$n[group_stats$group=="better"]
n_worse   <- group_stats$n[group_stats$group=="worse"]

for (se_col in se_cols) {
  
  # difference in means
  diff_mean <- group_stats$acc_group[group_stats$group=="better"] -
    group_stats$acc_group[group_stats$group=="worse"]
  
  # SE of the difference
  se_diff <- sqrt(
    group_stats[[se_col]][group_stats$group=="better"]^2 +
      group_stats[[se_col]][group_stats$group=="worse"]^2
  )
  
  # Welch t-value
  t_val <- diff_mean / se_diff
  
  # Welch degrees of freedom
  df_val <- (group_stats[[se_col]][group_stats$group=="better"]^2 +
               group_stats[[se_col]][group_stats$group=="worse"]^2)^2 /
    ((group_stats[[se_col]][group_stats$group=="better"]^4 / (n_better-1)) +
       (group_stats[[se_col]][group_stats$group=="worse"]^4 / (n_worse-1)))
  
  # two-sided p-value
  p_val <- 2 * (1 - pt(abs(t_val), df = df_val))
  
  # add to results
  results_welch <- rbind(results_welch,
                         data.frame(
                           assumption = se_col,
                           diff_mean = diff_mean,
                           se_diff = se_diff,
                           t_value = t_val,
                           df = df_val,
                           p_value = p_val
                         ))
}

group_stats
results_welch
