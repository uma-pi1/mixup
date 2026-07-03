# =============================================================================
# Create empirical results table (Tab. 2)

method_model_summ <- data_md |> 
  filter(!grepl("_old", method_name, fixed = T)) |>  # Remove old data.
  select(method_name, dataset_name, model_name, acc_md, se_md) |> 
  group_by(method_name, model_name) |> 
  summarize(
    avg = mean(acc_md),
    se = sd(acc_md) / sqrt(n()),
    .groups = "drop"
  ) |> 
  mutate(Average = paste0(fmt_number(avg), " $\\pm$ ", fmt_number(se))) |> 
  select(method_name, model_name, Average)
method_model_summ 

# LaTeX table.
data_md |> 
  filter(!grepl("_old", method_name, fixed = T)) |>  # Remove old data.
  mutate(val = paste0(fmt_number(acc_md), " $\\pm$ ", fmt_number(se_md))) |> 
  select(model_name, method_name, dataset_name, val) |> 
  # Reorder.
  mutate(
    model_name = factor(model_name, levels = model_order),
    dataset_name = factor(dataset_name, levels = dataset_order),
    method_name = factor(method_name, levels = method_order)
  ) |> 
  arrange(model_name, dataset_name, method_name) |> 
  pivot_wider(names_from = dataset_name, values_from = val) |> 
  # Add row averages.
  inner_join(method_model_summ, by = c("method_name", "model_name")) |> 
  # Rename methods.
  mutate(method_name = method_dict[method_name]) |> 
  # Rename cols.
  rename(Model = model_name, Method = method_name) |> 
  kable(booktabs = T, format = "latex",  escape = F, linesep = "") |> 
  kable_styling() |> 
  row_spec(0, bold = T) |> 
  write(here("output-tables", "table-2-empirical-results.tex"))
  
