# Assumptions (A1) + (A2) ##############################################################

results_a1_a2 <- data_md |>
  group_by(model_name, dataset_name) |>
  # Split off baseline vs others within each group
  group_modify(\(df, key) {
    base <- df |> filter(method_name == "baseline")
    others <- df |> filter(method_name != "baseline")
    # Join baseline stats onto each "other" method row
    others |>
      mutate(
        base_acc_md  = base$acc_md,
        base_n_md    = base$n_md,
        base_var_md  = base$var_md_a1_a2
      ) |>
      # compute Welch t-test from summary stats
      mutate(
        test = pmap(
          list(
            mean1 = acc_md,
            mean2 = base_acc_md,
            var1  = var_md_a1_a2,
            var2  = base_var_md,
            n1    = n_md,
            n2    = base_n_md
          ),
          t_test_from_summary
        )
      ) |>
      unnest(test)
  }) |>
  ungroup()

results_a1_a2 |> 
  select(model_name, dataset_name, method_name, p_value) |> 
  mutate(
    method_name = method_dict[method_name],
    p_value = round(p_value, 2)
  ) |> 
  pivot_wider(
    names_from=dataset_name,
    values_from=p_value
  ) |> 
  select(model_name, method_name, all_of(dataset_order)) |> 
  rename(
    "Model" = model_name,
    "Method" = method_name
  ) |> 
  kable(format = "latex", escape = F, linesep = "", booktabs = T) |> 
  write(here("output-tables", "table-10-p-values-md-a1-a2.tex"))
  
