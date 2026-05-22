# Compute GED Package

## Overview

The **Compute GED** package provides Python utilities to compute the Graph Edit Distance (GED) between graphs represented as [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) `Data` objects by interfacing with an external C++ GED executable. The package handles the conversion of graphs into a minimal GXL file format and the creation of an XML collection file that the GED executable requires. It further parses the execution output to return the computed GED value, the corresponding node mapping, and the runtime of the computation.

## Features

- **GXL Exporter:** Converts PyG `Data` objects to GXL files with basic nodes and edges.
- **XML Collection Generator:** Automatically creates an XML collection file that lists GXL files.
- **Unified GED Information Function:** A single function that returns a tuple containing the GED, node mapping, and runtime.
- **Caching:** Prevents multiple executions if the same data is processed repeatedly.
- **Flexible Methods:** Supports different GED methods and two modes ("exact" and "approximate") depending on how the graphs are provided and processed.

## Usage

### Exact GED Computation

In exact GED computation, graphs are first converted into a GED library format. The process works as follows:

#### Parameters and Process

- **Mode Parameter:** Set `--mode exact`.

**Process Overview:**

1. A temporary file is created for each graph using a routine that writes the graph in the GED library format (via the graph’s `get_ged_library_format()` method).  
2. The GED executable is invoked with the paths to the two temporary files and additional flags (such as `-q`, `-d`, and `-g`) to perform the computation.  
3. The output from the executable is parsed to extract:
   - The graph edit distance (GED) via a regular expression that searches for a pattern like `GED: <number>`.
   - A node mapping between the two graphs. The mapping is extracted by searching for a line starting with `Mapping: ` and processing the pairs of nodes.
   - The total time taken for the computation (when present) by searching for a line with `Total time: <time> (microseconds)`.
4. If either the GED value or the mapping cannot be found in the output, a corresponding exception is raised.  
5. If the computation exceeds the allowed timeout, the process yields a GED value of `-1` with the runtime set to the timeout value in microseconds.
6. Temporary files are removed after computation, ensuring no residual data remains.


The final output is returned as a `GEDResult` object that includes:
- The IDs of the two graphs.
- The computed GED.
- The total computation time (or the timeout value if the process expires).
- A dictionary representing the node mapping.
- The computed lower bound.

### Approximate GED Computation

In **approximate** mode, graphs are expected to be provided as PyTorch Geometric Data objects (typically obtained via a
method like `get_pyg_data()`). In this mode, the provided Data objects are directly passed to the GED computation utility. 
The processing pipeline converts these Data objects into temporary GXL files by invoking the `write_gxl()` function, creates 
an XML collection file referencing the generated GXL files, and then runs the external GED executable with these files. 
The JSON output from the executable is parsed to extract the approximate GED value, node mapping, and runtime. 

#### Parameters and Process

- **Mode Parameter:** Set `--mode approximate`.
- **Executable Parameter:** Specify the GED executable (for example, `--exec ./bin/edit_path_exec`).
- **Method:** Choose from the available methods (e.g., `IPFP`, `BRANCH`, `RING`, etc.) with `--method <method>`; note that different methods may provide varying approximations of the GED.

**Process Overview:**

1. The graphs are first converted into temporary GXL files via the `write_gxl()` function.
2. An XML collection file is created for the generated GXL files.
3. The GED executable is invoked with the temporary directory and XML collection file.
4. The JSON output is parsed to extract the approximate GED value, node mapping, and runtime.
5. The result tuple is returned.

#### Example Usage

```python
from torch_geometric.data import Data
from graph_mixup.compute_ged import
    __main__ as ged_main  # __main__ provides compute_ged_info
from graph_mixup.compute_ged.parser import parse_args

# Create two sample PyTorch Geometric Data objects.
# In approximate mode, these Data objects are typically generated via a get_pyg_data() method.
data0 = Data(x=[[1.0], [2.5], [3.2]], edge_index=[[0, 1], [1, 2]])
data1 = Data(x=[[1.0], [2.5], [3.2]], edge_index=[[0, 1], [1, 0]])

# Ensure that the parameters are set for approximate mode:
#   --mode approximate, --exec ./bin/edit_path_exec, --method IPFP (or any alternative method)
args = parse_args()
print("Running in Approximate Mode with parameters:")
print(f"Executable: {args.exec}")
print(f"Method: {args.approx_method}")
print(f"Mode: {args.mode}")

# Compute GED info
result = ged_main.compute_ged_info(data0, data1)
if result is not None:
    ged, mapping, runtime = result
    print("Approximate GED Computation Results:")
    print(f"Graph Edit Distance: {ged}")
    print(f"Node Mapping: {mapping}")
    print(f"Runtime (µs): {runtime}")
else:
    print("Approximate GED computation failed.")
```

**Expected Results:**

- **Graph Edit Distance (GED):** An approximate integer value determined by the chosen method.
- **Node Mapping:** A dictionary mapping nodes between graphs (keys converted to integers).
- **Runtime:** Execution time measured in microseconds.

**Choosing Different Methods:**

- The `--method` parameter allows selecting among various GED approximation techniques such as:
  - **IPFP:** Integer Project Fixed Point method for GED estimation. This method repeatedly refines the node mapping by projecting the current solution onto a space where the edit cost is minimized. It typically offers a good balance between runtime and accuracy.
  - **BRANCH / BRANCH_FAST:** These are branch-and-bound techniques that systematically explore all possible mappings while using heuristics to prune unlikely candidates. `BRANCH_FAST` incorporates additional shortcuts to reduce the search space, sacrificing some accuracy for speed.
  - **BRANCH_TIGHT / BRANCH_UNIFORM / BRANCH_COMPACT**: Variations of branch-and-bound strategies. They differ in how they prioritize or compress the search space. For example, `BRANCH_TIGHT` emphasizes tighter bounds, while `BRANCH_COMPACT` might focus on a more condensed representation of candidate solutions.
  - **PARTITION**: This technique divides the graphs into smaller partitions and computes GED on these parts, using the aggregated results to approximate the full graph edit distance. Partitioning can enhance performance on large graphs.
  - **HYBRID**: As the name suggests, this method combines multiple heuristics to achieve a more robust GED approximation. It leverages strengths from different approaches to improve overall accuracy without a significant increase in runtime.
  - **RING**: A specialized method that restructures graphs into a ring or circular representation. By reducing the graphs to cyclic structures, it exploits inherent patterns that can simplify the mapping problem when graphs exhibit cyclic characteristics.
  - **ANCHOR_AWARE_GED**: This variant selects certain "anchor" nodes based on graph properties. These anchors guide the matching process, potentially yielding a more stable and semantically strong approximation of the GED.
  - **WALKS**: Instead of considering the entire graph structure, this method estimates the GED by analyzing random walks or sampled paths through the graphs. It then infers the similarity based on these substructure explorations.
  - **BIPARTITE**: By formulating the mapping problem as a bipartite graph matching task, this approach applies well-known matching algorithms to obtain a GED estimate. It is particularly effective when node similarities can be clearly defined.
  - **SUBGRAPH**: This method breaks down graphs into smaller subgraphs and computes distances between these components individually. The aggregated differences provide an overall approximation of the edit distance.
  - **NODE**: Focused primarily on node attributes, this technique estimates the GED by considering differences in node labels and their corresponding features, often via heuristics that assess the cost of relabel operations.
  - **REFINE**: Starting with a coarse initial mapping, the refine method iteratively improves the mapping by adjusting pairings, thereby incrementally reducing the error in the GED estimation.
  - **BP_BEAM**: This approach utilizes a beam search strategy over possible candidate mappings. At each iteration, it retains a limited number of the most promising partial solutions to manage complexity while still exploring a diverse set of possibilities.
  - **SIMULATED_ANNEALING**: Borrowing concepts from statistical mechanics, this method uses randomized search that gradually cools—accepting worse solutions early on to escape local minima and converging later for a finer approximation.
  - **HED**: A heuristic edit distance strategy that uses problem-specific insights to quickly estimate the minimum number of operations required, especially when exact computation is too resource-intensive.
  - **STAR**: This method leverages star patterns centered around specific nodes. By focusing on central nodes and their immediate neighbors, it simplifies the matching process based on local connectivity and reduces the overall edit distance estimate.

Select the method based on your application's needs. For example, if you desire a balance between speed and accuracy, you might choose `IPFP`. If you require a fast approximation with a slightly relaxed bound, `BRANCH_FAST` might be appropriate.

## Configuration and Parameters

The package uses a parser defined in `parser.py` to configure computations. Key parameters include:

- **dataset_name:** Name of the dataset (used for database management in larger projects).
- **n_cpus:** Number of CPUs to use for parallel processing.
- **timeout:** Timeout (in seconds) for each GED calculation.
- **lb_threshold:** Lower bound threshold; if the computed lower bound exceeds this value, GED computation is skipped.
- **batch_size:** Batch size for processing graph pairs.
- **mixup_only:** Flag to denote whether only mixup graph pairs are processed.
- **method:** GED method (e.g., `IPFP`, `BRANCH`, `RING`, etc.).
- **exec:** Path to the GED executable.
- **mode:** Computation mode; either `exact` or `approximate`.

Parameters can be set via command-line arguments or adjusted in the parser defaults.
