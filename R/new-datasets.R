new_datasets_mdf <- read_csv(
  # Load baseline results.
    here("data", "new-datasets", "new-gcn-baseline-results.csv")
  ) |> 
  union_all(
    read_csv(here("data", "new-datasets", "other-baseline-results.csv"))
  ) |> 
  mutate(method_name = if_else(is.na(method_name), "baseline", method_name)) |> 
  # Load method results.
  union_all(
    read_csv(here("data", "new-datasets", "new-gcn-method-results.csv"))
  ) |> 
  union_all(
    read_csv(here("data", "new-datasets", "other-method-results.csv"))
  ) |> 
  select(model_name, dataset_name, method_name, fold, test_acc) |> 
  group_by(model_name, dataset_name, method_name, fold) |> 
  summarize(
    acc_mdf = mean(test_acc),
    var_mdf = var(test_acc),
    n_mdf = n()
  ) |> 
  ungroup() |> 
  select(-fold)
new_datasets_mdf

# Union with data from previous datasets.
# Attention: This will cause duplication if it runs multiple times.
data_mdf <- data_mdf |> union_all(new_datasets_mdf)
data_mdf
