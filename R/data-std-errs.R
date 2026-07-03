# =============================================================================
# Compute standard errors.
# =============================================================================

# ===
# Step 1: Group per dataset.
# ===

data_md <- data_mdf |>
  group_by(method_name, model_name, dataset_name) |>
  summarize(
    acc_md = mean(acc_mdf),
    var_md_within_fold = mean(var_mdf),
    var_md_between_folds = var(acc_mdf),
    n_md = n() * first(n_mdf),
    .groups = "drop"
  ) |>
  mutate(
    var_md_a1_a2 = var_md_within_fold + var_md_between_folds,
    var_md_a3 = var_md_within_fold # A3: (I) "between-folds" vanishes
  ) |>
  select(-var_md_within_fold, -var_md_between_folds) |>
  mutate(
    se_md = sqrt(var_md_a1_a2 / n_md)
  )
data_md

# ===
# Step 2: Group per model.
# ===

data_m <- data_md |>
  group_by(method_name, model_name) |>
  summarize(
    acc_m = mean(acc_md),
    var_m_within_ds_a1_a2 = mean(var_md_a1_a2),
    var_m_within_ds_a3 = mean(var_md_a3),
    var_m_between_ds = var(acc_md),
    n_d = n() * first(n_md),
    .groups = "drop"
  ) |>
  mutate(
    var_m_a1 = var_m_within_ds_a1_a2 + var_m_between_ds,
    var_m_a2 = var_m_within_ds_a1_a2, # A2: (I) "between-datasets" vanishes
    var_m_a3 = var_m_within_ds_a3, # A3: (II) "between-datasets" vanishes
  ) |>
  select(-var_m_within_ds_a1_a2, -var_m_within_ds_a3, -var_m_between_ds)

# ===
# Step 3: Group per method.
# ===

data_final_tmp <- data_m |>
  group_by(method_name) |>
  summarize(
    acc = mean(acc_m),
    var_within_model_a1 = mean(var_m_a1),
    var_within_model_a2 = mean(var_m_a2),
    var_within_model_a3 = mean(var_m_a3),
    var_between_models = var(acc_m),
    n = n() * first(n_d),
    .groups = "drop"
  ) |>
  mutate(
    var_a1 = var_within_model_a1 + var_between_models,
    var_a2 = var_within_model_a2, # A2: (II) "between-models" vanishes
    var_a3 = var_within_model_a3, # A3: (III) "between-models" vanishes
  ) |>
  mutate(
    se_a1 = sqrt(var_a1 / n),
    se_a2 = sqrt(var_a2 / n),
    se_a3 = sqrt(var_a3 / n)
  ) |>
  select(method_name, acc, se_a1, se_a2, se_a3, n, var_a1, var_a2, var_a3)

data_final <- data_final_tmp |>
  select(method_name, acc, se_a1, se_a2, se_a3, n)

# ===
# Step 4: Reshape the data to long format for ggplot.
# ===

data_long <- data_final |>
  pivot_longer(
    cols = starts_with("se_a"),
    names_to = "se_type",
    values_to = "se"
  ) |>
  mutate(
    method_name = factor(
      method_name,
      levels = c(setdiff(unique(method_name), "ged_mixup"), "ged_mixup")
    )
  )

# ===
# Step 5: Extract baseline accuracy from original data.
# ===

baseline_acc <- data_final |>
  filter(method_name == "baseline") |>
  pull(acc)
