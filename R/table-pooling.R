# Prepare data.
pooling_table_data <- data_final |>
  select(-n) |>
  # Add p-values.
  left_join(t_test_wide |> select(-acc), by = "method_name") |>
  filter(!grepl("_old", method_name, fixed = T)) |>
  # Order by method.
  mutate(
    method_name = factor(method_name, levels = method_order)
  ) |>
  arrange(method_name) |>
  # Rename methods.
  mutate(method_name = method_dict[method_name]) |>
  # Format numbers.
  mutate(
    acc = fmt_number(acc),
    se_a1 = fmt_number(se_a1),
    se_a2 = fmt_number(se_a2),
    se_a3 = fmt_number(se_a3),
    p_se_a1 = format(round(p_se_a1, 2), nsmall = 2),
    p_se_a2 = format(round(p_se_a2, 2), nsmall = 2),
    p_se_a3 = format(round(p_se_a3, 2), nsmall = 2),
  ) |>
  # Reorder columns.
  select(
    method_name,
    acc,
    se_a1,
    p_se_a1,
    se_a2,
    p_se_a2,
    se_a3,
    p_se_a3
  )
pooling_table_data


# LaTeX table.
pooling_table_data |>
  # Rename cols.
  rename(Method = method_name, Accuracy = acc) |>
  kable(booktabs = T, format = "latex", escape = F, linesep = "") |>
  kable_styling() |>
  row_spec(0, bold = T) |>
  write(here("output-tables", "table-3-pooling-table.tex"))


