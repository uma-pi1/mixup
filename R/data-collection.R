data_old <- read.csv(
  here("data", "old-datasets", "mixup-results-folds.csv"),
  sep = ";"
) |>
  select(
    model_name,
    method_name,
    dataset_name,
    num_results,
    avg_test_acc_fold,
    std_test_acc_fold
  ) |>
  filter(method_name != "fgw_mixup") |>
  mutate(
    method_name = if_else(method_name == "", "baseline", method_name)
  ) |>
  mutate(
    method_name = if_else(method_name == "s_mixup", "s_mixup_old", method_name)
  ) |>
  mutate(
    method_name = if_else(method_name == "g_mixup", "g_mixup_old", method_name)
  )

data_new <- read.csv(here(
  "data",
  "old-datasets",
  "s_mixup-g_mixup-results-folds-new.csv"
)) |>
  select(
    model_name,
    method_name,
    dataset_name,
    num_results,
    avg_test_acc_fold,
    std_test_acc_fold
  )

data_mdf <- data_old |>
  union_all(data_new) |>
  mutate(
    var_mdf = std_test_acc_fold^2
  ) |>
  select(-std_test_acc_fold) |>
  rename(
    acc_mdf = avg_test_acc_fold,
    n_mdf = num_results
  ) |>
  filter(!(method_name %in% c("g_mixup_old", "s_mixup_old")))
