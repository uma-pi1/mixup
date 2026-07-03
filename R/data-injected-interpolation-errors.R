lambda_corr <- read_csv(here(
  "data",
  "injected-interpolation-errors",
  "eps-025.csv"
)) |>
  select(model_name, method_name, dataset_name, test_acc, fold) |>
  mutate(epsilon = 0.25) |>
  union_all(
    read_csv(here("data", "injected-interpolation-errors", "eps-05.csv")) |>
      select(model_name, method_name, dataset_name, test_acc, fold) |>
      mutate(epsilon = 0.5)
  ) |>
  union_all(
    read_csv(here("data", "injected-interpolation-errors", "eps-0625.csv")) |>
      select(model_name, method_name, dataset_name, test_acc, fold) |>
      mutate(epsilon = 0.625)
  ) |>
  union_all(
    read_csv(here("data", "injected-interpolation-errors", "eps-075.csv")) |>
      select(model_name, method_name, dataset_name, test_acc, fold) |>
      mutate(epsilon = 0.75)
  ) |>
  union_all(
    read_csv(here("data", "injected-interpolation-errors", "eps-0875.csv")) |>
      select(model_name, method_name, dataset_name, test_acc, fold) |>
      mutate(epsilon = 0.875)
  ) |>
  union_all(
    read_csv(here("data", "injected-interpolation-errors", "eps-1.csv")) |>
      select(model_name, method_name, dataset_name, test_acc, fold) |>
      mutate(epsilon = 1.0)
  ) |>
  group_by(model_name, method_name, dataset_name, fold, epsilon) |>
  summarise(
    acc_mdf = mean(test_acc),
    var_mdf = var(test_acc),
    n_mdf = n()
  ) |>
  ungroup() |>
  select(-fold) |>
  union_all(
    data_mdf |>
      mutate(epsilon = 0.0) |>
      filter(
        model_name == "GCN",
        dataset_name == "IMDB-MULTI",
        method_name == "ged_mixup"
      )
  ) |>
  # Group by model, method and dataset.
  group_by(method_name, model_name, dataset_name, epsilon) |>
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
  ) |>
  select(-var_md_a1_a2, -var_md_a3, -n_md)
lambda_corr

# Plot.
lambda_corr |>
  ggplot(aes(
    x = epsilon,
    y = acc_md,
    ymin = acc_md - se_md,
    ymax = acc_md + se_md
  )) +
  theme_minimal() +
  geom_point(aes(color = epsilon)) +
  geom_errorbar(width = 0.01) +
  geom_hline(yintercept = .33, linetype = 2) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 3))

# Markdown.
lambda_corr |>
  mutate(
    val = paste0(fmt_number(acc_md), " ± ", fmt_number(se_md)),
    method_name = method_dict[method_name]
  ) |>
  select(model_name, dataset_name, method_name, epsilon, val) |>
  rename(
    "Model" = model_name,
    "Method" = method_name,
    "Dataset" = dataset_name,
    "Expected mIE" = epsilon,
    "Acc. ± SE (%)" = val
  ) |>
  kable(format = "markdown") |>
  write(here("output-tables", "table-5-injected-interpolation-errors.md"))
