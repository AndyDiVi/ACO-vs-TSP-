# Laboratory 3 – Traveling Salesman Problem using Ant Colony Optimization
Authors: Di Via, Graziani & Hiller-Szala

<img width="850" height="654" alt="image" src="https://github.com/user-attachments/assets/13cfed7d-cae0-4d0e-a075-2a5ce12f298e" />



## Overview

This laboratory implements Ant Colony Optimization (ACO) in MATLAB to search for a short closed tour through 27 European capitals.

The Traveling Salesman Problem (TSP) requires a route that:

- Visits every city exactly once.
- Returns to the starting city.
- Minimizes the total tour length.

The algorithm uses a population of artificial ants that construct candidate tours and communicate indirectly through a shared pheromone matrix.

The implementation supports:

- Four ant initialization strategies.
- A pseudo-random-proportional state transition rule.
- Four local pheromone update options.
- Two global pheromone reinforcement strategies.
- Stagnation-based stopping.
- Tour animation and convergence plots.
- Experiments comparing different algorithm configurations.

## Dataset

The main experiment uses the coordinate file:

```text
capitali_EU.txt
```

The dataset contains the geographical coordinates and names of 27 European capitals.

The coordinate reader can also process other `.txt` files with the same structure.

### Input format

Each row must contain seven whitespace-separated fields:

| Column | Description |
|---|---|
| 1 | Latitude degrees |
| 2 | Latitude minutes |
| 3 | Latitude hemisphere: `N` or `S` |
| 4 | Longitude degrees |
| 5 | Longitude minutes |
| 6 | Longitude hemisphere: `E` or `W` |
| 7 | City name |

City names should be written as single tokens, without embedded spaces.

### Coordinate conversion

Degrees and minutes are converted into decimal degrees:

```math
\theta_{\mathrm{decimal}} = s\left(d+\frac{m}{60}\right)
```

Here:

- $d$ is the number of degrees.
- $m$ is the number of minutes.
- $s=1$ for north latitudes and east longitudes.
- $s=-1$ for south latitudes and west longitudes.

The resulting coordinate matrix stores longitude in the first column and latitude in the second:

```math
C \in \mathbb{R}^{n\times2}
```

```matlab
[city_names, cities, n] = read_cities_coordinates(data_filename);
```

The implemented conversion produces decimal-degree coordinates; it does not apply a full Mercator projection.

## Mathematical Background

### Tour representation

A tour is represented as an ordered sequence of city indices, with the starting city repeated at the end:

```math
\pi = (\pi_1,\pi_2,\ldots,\pi_n,\pi_1)
```

Its length is:

```math
L(\pi) = \sum_{i=1}^{n}d(\pi_i,\pi_{i+1})
```

where $\pi_{n+1}=\pi_1$.

### Distance matrix

The implementation computes pairwise Euclidean distances between the coordinate rows:

```matlab
distance_matrix = squareform(pdist(cities));
```

For cities $r$ and $s$:

```math
d(r,s) = \sqrt{(x_r-x_s)^2+(y_r-y_s)^2}
```

The distance matrix is symmetric:

```math
d(r,s) = d(s,r)
```

Because the inputs are longitude and latitude in decimal degrees, the resulting lengths are coordinate-space distances, not distances in kilometers.

### Heuristic information

The heuristic desirability of an edge is defined as the inverse distance:

```math
\eta(r,s) = \frac{1}{d(r,s)}
```

Shorter edges therefore receive higher heuristic values.

The implementation sets infinite inverse-distance entries to zero.

### Pheromone initialization

All off-diagonal pheromone entries are initialized with the same value:

```math
\tau(r,s) = \tau_0 \qquad r\neq s
```

```math
\tau(r,r) = 0
```

The default initialization is:

```matlab
tau = tau0 .* (1 - eye(n));
```

## Methodology

### 1. Ant initialization strategies

The implementation provides four starting-city strategies.

| Strategy | Description | Ants in the main experiment |
|---|---|---:|
| 1 | Distinct random starting cities, fixed across iterations | 10 |
| 2 | Distinct random starting cities reassigned at each iteration | 10 |
| 3 | All ants start from the same user-selected city | 10 |
| 4 | One ant starts from each city | 27 |

For Strategy 4:

```math
m=n
```

Every constructed tour returns to its own starting city.

Strategy 2 resamples the starting assignments at each iteration; an ant may still receive the same starting city in consecutive iterations.

### 2. Tour construction

Each ant maintains a record of the cities it has not yet visited.

At each step, the ant selects one unvisited city using a combination of pheromone intensity and inverse-distance information.

The desirability score is:

```math
a(r,s) = \tau(r,s)\eta(r,s)^{\beta}
```

The parameter $\beta$ controls the influence of the distance heuristic.

### 3. State transition rule

Let $J_k(r)$ denote the unvisited cities available to ant $k$ when it is located at city $r$.

A random value $q$ is drawn uniformly from $[0,1]$.

#### Exploitation

If $q\leq q_0$, the ant chooses the city with the largest desirability score:

```math
s = \arg\max_{u\in J_k(r)}\left\{\tau(r,u)\eta(r,u)^{\beta}\right\}
```

#### Exploration

Otherwise, the next city is sampled using normalized desirability scores:

```math
p_k(r,s) = \frac{\tau(r,s)\eta(r,s)^{\beta}}{\sum_{u\in J_k(r)}\tau(r,u)\eta(r,u)^{\beta}}
```

This distribution is defined over the currently unvisited cities.

The implementation performs this sampling through a cumulative probability distribution.

### 4. Local pheromone updates

Local updates are applied during tour construction.

The general update is:

```math
\tau(r,s) \leftarrow (1-\rho)\tau(r,s)+\rho\Delta\tau_{\mathrm{local}}(r,s)
```

The implementation offers four options.

#### Local Rule 1: Ant-Q

The update uses a discounted estimate of future pheromone:

```math
\Delta\tau_{\mathrm{local}}(r,s) = \gamma\max_{z\in J_k(s)}\tau(s,z)
```

The parameter $\gamma$ controls the contribution of this future estimate.

The interface labels this option `Q-Ant`.

#### Local Rule 2: ACS

The update moves the edge pheromone toward the initial level:

```math
\Delta\tau_{\mathrm{local}}(r,s) = \tau_0
```

#### Local Rule 3: AC

No pheromone is added locally:

```math
\Delta\tau_{\mathrm{local}}(r,s) = 0
```

The update therefore reduces to local evaporation.

#### Local Rule 4: None

No local pheromone update is applied.

### 5. Global pheromone updates

After all ants have completed their tours, the algorithm reinforces a selected tour.

Two selection rules are available:

| Global rule | Reinforced tour |
|---|---|
| 1: Global Best | Shortest tour found since the beginning of the run |
| 2: Iteration Best | Shortest tour found in the current iteration |

Reinforcement depends on the inverse length of the selected tour, while evaporation reduces existing pheromone levels.

Regardless of the reinforcement rule, the algorithm separately tracks the best tour found throughout the complete run.

### 6. Stopping criteria

The algorithm stops when:

- The best tour length has not improved for the configured number of consecutive iterations.
- The maximum number of iterations is reached.

The default limits are:

```matlab
stop = 10;
max_iter = 2500;
```

Reaching the stagnation threshold indicates that the current search has stopped improving; it does not prove that the best tour is globally optimal.

## Algorithm Parameters

| Parameter | Default value | Description |
|---|---:|---|
| `beta` | 2 | Distance heuristic exponent |
| `q0` | 0.9 | Exploitation threshold |
| `alpha` | 0.1 | Global update parameter |
| `rho` | 0.1 | Local update parameter |
| `tau0` | 0.00022 | Initial pheromone |
| `m` | 10 | Number of ants for Strategies 1–3 |
| `m` for Strategy 4 | Number of cities | One ant per city |
| `gamma` | 0.2 | Ant-Q discount factor in the main code |
| `stop` | 10 | Consecutive iterations without improvement |
| `max_iter` | 2500 | Maximum iterations |

The main code sets `gamma = 0.2` for Ant-Q, while the comparison experiments explore additional values.

The initial pheromone is assigned directly in the script:

```matlab
tau0 = 2.2e-4;
```

## Implementation

The laboratory contains a main script and two principal custom functions.

### Coordinate reader

```matlab
[city_names, cities, n] = read_cities_coordinates(filename);
```

This function:

- Reads a seven-column coordinate file.
- Validates the input structure.
- Converts degrees and minutes into signed decimal degrees.
- Returns city names, coordinates, and the number of cities.

### ACO function

```matlab
[bestTour, best_distance, total_Lbest, ...
    min_distance, total_Lmin, iter] = ACO( ...
    cities, m, alpha, rho, beta, q0, tau0, ...
    stop, max_iter, strategy, city, ...
    local_update_rule, gamma, global_update_rule);
```

#### Inputs

| Input | Description |
|---|---|
| `cities` | Matrix of city coordinates |
| `m` | Number of ants |
| `alpha` | Global update parameter |
| `rho` | Local update parameter |
| `beta` | Distance heuristic exponent |
| `q0` | Exploitation threshold |
| `tau0` | Initial pheromone |
| `stop` | Stagnation limit |
| `max_iter` | Maximum iterations |
| `strategy` | Starting-city strategy: 1–4 |
| `city` | Starting city index for Strategy 3 |
| `local_update_rule` | Local update option: 1–4 |
| `gamma` | Discount factor for Ant-Q |
| `global_update_rule` | Global reinforcement option: 1–2 |

All 14 arguments are passed to the function. The values of `city` and `gamma` are ignored when their corresponding options are not selected.

#### Outputs

| Output | Description |
|---|---|
| `bestTour` | Best closed tour found |
| `best_distance` | Best-so-far length history |
| `total_Lbest` | Length of the best tour found |
| `min_distance` | Iteration-best length history |
| `total_Lmin` | Best length in the final iteration |
| `iter` | Number of executed iterations |

The returned `bestTour` contains $n+1$ indices because the final entry repeats the starting city.

## Experimental Results

The report compares initialization strategies, local update rules, and global reinforcement rules over three runs of the European capitals dataset.

The lowest reported tour length is:

```math
L_{\mathrm{best}} = 153.3879
```

Another frequently reported result is:

```math
L = 153.7302
```

These values use the Euclidean coordinate-space distance defined by the implementation.

### Example configurations

| Run | Strategy | Local rule | Gamma | Global rule | Best length | Executed iterations |
|---|---:|---|---:|---|---:|---:|
| 1 | 1 | Ant-Q | 0.20 | Iteration Best | 153.3879 | 36 |
| 1 | 2 | AC | Not used | Iteration Best | 153.7302 | 40 |
| 1 | 4 | ACS | Not used | Iteration Best | 153.7302 | 40 |
| 1 | 4 | AC | Not used | Iteration Best | 153.3879 | 27 |
| 2 | 1 | Ant-Q | 0.10 | Global Best | 153.3879 | 18 |

The iteration counts include execution until the stopping condition, not necessarily just the iterations required to first discover the reported tour.

### Reported observations

The report highlights:

- Strategies 2 and 4 frequently produced high-quality tours.
- Ant-Q appeared frequently among the selected best-performing results.
- Both Global Best and Iteration Best reinforcement produced the lowest observed length.
- Strategy 3 did not appear in the report's selected best-result subset.

These observations describe the reported experiments; they do not establish a universally best ACO configuration.

Strategy 4 also uses more ants than the other strategies, so comparisons based only on iteration count do not represent equal computational effort.

## Visualizations

The main script produces:

- A labeled plot of the input cities.
- A step-by-step animation of the best closed tour.
- A comparison of best-so-far and iteration-best tour lengths.

The two length histories are:

```math
L_{\min}^{(g)} = \min_{1\leq k\leq m}L_k^{(g)}
```

```math
L_{\mathrm{best}}^{(g)} = \min_{1\leq h\leq g}L_{\min}^{(h)}
```

The best-so-far curve cannot increase, while the iteration-best curve may fluctuate.

The script also includes sections for opening previously saved comparison figures:

```text
Lplot_local1-2.fig
Lplot_local3-4.fig
```

These files are needed only when running the corresponding `openfig` sections.

## Requirements

- MATLAB.
- Statistics and Machine Learning Toolbox for `pdist` and `squareform`.
- A coordinate `.txt` file with the required structure.
- The `read_cities_coordinates` and `ACO` functions.
- A graphical environment for the interactive `listdlg` selections.

Saved `.fig` files are optional unless the comparison-figure sections are executed.

## How to run

1. Clone or download this repository.
2. Place the coordinate dataset in the MATLAB working directory.
3. Ensure that the custom functions are included in the script or available on the MATLAB path.
4. Open and execute the main script.
5. Enter the dataset filename when prompted.
6. Select an ant initialization strategy.
7. If Strategy 3 is selected, choose the starting city.
8. Select a local pheromone update rule.
9. Select a global reinforcement rule.

The script runs the search, displays the best tour, animates its edges, and plots the tour-length histories.

If the saved comparison figures are unavailable, skip or comment out the corresponding `openfig` calls.

## Reproducibility and Customization

### Fixed random seed

Enable a seed before running the stochastic search:

```matlab
rng(42);
```

The comparison experiment code uses seeds 42, 100, and 150 for its three runs.

Results also depend on the chosen configuration and the sequence of random operations.

### Alternative datasets

Other coordinate datasets can be used as long as they follow the same seven-column format.

When changing datasets:

- Check the coordinate interpretation.
- Review the initial pheromone value.
- Ensure the number of ants is valid for the selected strategy.
- Set `m = n` for Strategy 4.

### Parameter experiments

The script contains commented experiment sections for comparing:

- Starting-city strategies.
- Local update rules.
- Global reinforcement rules.
- Ant-Q discount factors.
- Multiple random seeds.

These sections can be enabled separately from the interactive workflow.

## Limitations

- The algorithm is stochastic and does not guarantee a globally optimal solution.
- The best reported length is an experimental result, not a certified optimum.
- Distances are computed directly from longitude and latitude in decimal degrees, rather than from road routes or great-circle distances.
- Stagnation-based stopping may terminate a run before a better tour is discovered.
- The reported configuration comparisons concern one main dataset and a limited number of runs.
- Different ant counts imply different computational costs per iteration.
- Starting at a different city does not by itself define a different closed cycle: cyclic shifts and reversed tours may represent the same symmetric TSP solution.

## Conclusions

This laboratory demonstrates a configurable Ant Colony Optimization implementation for the Traveling Salesman Problem.

Artificial ants construct tours using pheromone information and inverse-distance heuristics, while local and global updates influence subsequent exploration.

The implementation compares four initialization strategies, four local update options, and two global reinforcement rules.

For the 27-capital dataset, the lowest reported tour length is 153.3879 in the coordinate-space distance used by the code.

The experiment illustrates how
