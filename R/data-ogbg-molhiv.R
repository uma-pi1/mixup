# ===
# Step 0: Load data.
# ===

ogbg_results <- read.csv(here("data", "ogbg", "ogbg-molhiv-results.csv")) |>
  select(
    model_name,
    method_name,
    dataset_name,
    test_auroc
  )


# ===
# Step 1: Group by model, dataset, & method (only one fold).
# ===

ogbg_md <- ogbg_results |>
  group_by(model_name, dataset_name, method_name) |>
  summarize(
    auroc_md = mean(test_auroc),
    se_md = sd(test_auroc) / sqrt(n()),
    n_md = n(),
    var_md_a1_a2 = var(test_auroc),
    .groups = "drop"
  )

# ===
# Step 2: Empirical results per model and dataset.
# ===

# LaTeX table.
ogbg_md |>
  mutate(val = paste0(fmt_number(auroc_md), " $\\pm$ ", fmt_number(se_md))) |>
  select(model_name, method_name, dataset_name, val) |>
  # Reorder.
  mutate(
    model_name = factor(model_name, levels = model_order),
    dataset_name = factor(dataset_name, levels = dataset_order),
    method_name = factor(method_name, levels = method_order)
  ) |>
  arrange(model_name, dataset_name, method_name) |>
  pivot_wider(names_from = dataset_name, values_from = val) |>
  # Rename methods.
  mutate(method_name = method_dict[method_name]) |>
  # Rename cols.
  rename(Model = model_name, Method = method_name) |>
  kable(booktabs = T, format = "latex", escape = F, linesep = "") |>
  kable_styling() |>
  row_spec(0, bold = T) |>
  write(here("output-tables", "table-11-ogbg-empirical-results.tex"))

# ===
# Step 3: p-Values.
# ===

ogbg_results_a1_a2 <- ogbg_md |>
  group_by(model_name, dataset_name) |>
  # Split off baseline vs others within each group
  group_modify(\(df, key) {
    base <- df |> filter(method_name == "")
    others <- df |> filter(method_name != "")
    # Join baseline stats onto each "other" method row
    others |>
      mutate(
        base_auroc_md = base$auroc_md,
        base_n_md = base$n_md,
        base_var_md = base$var_md_a1_a2
      ) |>
      # compute Welch t-test from summary stats
      mutate(
        test = pmap(
          list(
            mean1 = auroc_md,
            mean2 = base_auroc_md,
            var1 = var_md_a1_a2,
            var2 = base_var_md,
            n1 = n_md,
            n2 = base_n_md
          ),
          t_test_from_summary
        )
      ) |>
      unnest(test)
  }) |>
  ungroup()
ogbg_results_a1_a2

ogbg_results_a1_a2 |>
  select(model_name, dataset_name, method_name, p_value) |>
  mutate(
    method_name = method_dict[method_name],
    p_value = round(p_value, 2)
  ) |>
  pivot_wider(
    names_from = dataset_name,
    values_from = p_value
  ) |>
  rename(
    "Model" = model_name,
    "Method" = method_name
  ) |>
  kable(format = "latex", escape = F, linesep = "", booktabs = T) |>
  write(here("output-tables", "table-11-ogbg-p-values-md-a1-a2.tex"))
