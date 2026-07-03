labelnoise_md <- read_ods(
  here("data", "label-noise", "label-noise-results.ods")
) |>
  select(model_name, method_name, dataset_name, test_acc, fold, labelnoise) |>
  mutate(method_name = if_else(is.na(method_name), "baseline", method_name)) |>
  filter(model_name == "GIN", dataset_name == "IMDB-BINARY") |>
  group_by(model_name, method_name, dataset_name, fold, labelnoise) |>
  summarise(
    acc_mdf = mean(test_acc),
    var_mdf = var(test_acc),
    n_mdf = n()
  ) |>
  ungroup() |>
  select(-fold) |>
  union_all(
    data_mdf |>
      mutate(labelnoise = 0.0) |>
      filter(
        model_name == "GIN",
        dataset_name == "IMDB-BINARY",
        !(method_name %in% c("g_mixup_old", "s_mixup_old"))
      )
  ) |>
  # Group by model, method and dataset.
  group_by(method_name, model_name, dataset_name, labelnoise) |>
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
labelnoise_md

# Tables ###############################################################################

labelnoise_md |>
  mutate(
    # Format value string (use your fmt_number if you prefer)
    val = paste0(fmt_number(acc_md), " $\\pm$ ", fmt_number(se_md)),
    model_name = factor(model_name, levels = model_order),
    dataset_name = factor(dataset_name, levels = dataset_order),
    method_name = factor(method_name, levels = method_order)
  ) |>
  arrange(model_name, dataset_name, method_name, labelnoise) |>
  select(model_name, dataset_name, method_name, labelnoise, val) |>
  mutate(
    labelnoise = as.character(labelnoise),
    method_name = method_dict[method_name]
  ) |>
  # Wide pivot: columns become noise levels
  pivot_wider(names_from = labelnoise, values_from = val, names_sort = TRUE) |>
  # (Optional) rename columns to something more LaTeX-friendly
  rename(
    Model = model_name,
    Dataset = dataset_name,
    Method = method_name
    # Noise columns keep their numeric names: "0", "0.125", "0.25", "0.5"
  ) |>
  kable(format = "latex", booktabs = TRUE, escape = FALSE, linesep = "") |>
  kable_styling(full_width = FALSE, position = "center") |>
  row_spec(0, bold = TRUE) |>
  write(here("output-tables", "table-9-labelnoise.tex"))

# Assumptions (A1) + (A2) ##############################################################

labelnoise_a1_a2 <- labelnoise_md |>
  group_by(model_name, dataset_name, labelnoise) |>
  # Split off baseline vs others within each group
  group_modify(\(df, key) {
    base <- df |> filter(method_name == "baseline")
    others <- df |> filter(method_name != "baseline")
    # Join baseline stats onto each "other" method row
    others |>
      mutate(
        base_acc_md = base$acc_md,
        base_n_md = base$n_md,
        base_var_md = base$var_md_a1_a2
      ) |>
      # compute Welch t-test from summary stats
      mutate(
        test = pmap(
          list(
            mean1 = acc_md,
            mean2 = base_acc_md,
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

labelnoise_a1_a2 |>
  select(model_name, dataset_name, method_name, labelnoise, p_value) |>
  mutate(
    method_name = method_dict[method_name],
    p_value = round(p_value, 2)
  ) |>
  pivot_wider(
    names_from = labelnoise,
    values_from = p_value
  ) |>
  rename(
    "Method" = method_name
  ) |>
  kable(format = "latex", escape = F, linesep = "", booktabs = T) |>
  write(here("output-tables", "table-9-p-values-labelnoise-a1-a2.tex"))

