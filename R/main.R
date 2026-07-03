library(tidyverse)
library(here)
library(knitr)
library(kableExtra)
library(readODS)

# ===
# Load functions.
# ===

source(here("functions.R"))

# ===
# Load data.
# ===

source(here("data-collection.R"))

# ===
# Table 9: Label noise experiments.
# ===

source(here("data-label-noise.R"))

# ===
# Load new datasets.
# ===

source(here("new-datasets.R"))

# ===
# Table 11: OGBG-MOLHIV.
# ===

source(here("data-ogbg-molhiv.R"))

# ===
# Compute standard errors.
# ===

source(here("data-std-errs.R"))

# ===
# Table 2: Create empirical results table.
# ===

source(here("table-empirical-results.R"))

# ===
# Test better vs. worse than baseline methods.
# ===

source(here("testing-better-vs-worse-than-baseline.R"))

# ===
# Compute t-tests.
# ===

source(here("data-t-tests.R"))

# ===
# Table 3: Create pooling table.
# ===

source(here("table-pooling.R"))

# ===
# Table 10: p-values.
# ===

source(here("compute-pair-wise-t-tests.R"))

# ===
# Table 5: Injected interpolation errors.
# ===

source(here("data-injected-interpolation-errors.R"))
