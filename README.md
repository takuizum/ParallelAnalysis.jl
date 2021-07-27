# ParallelAnalysis


## Basic usage

Parallel analysis [(Houts, 1965)](https://link.springer.com/article/10.1007/BF02289447) diagnoses the number of dimension to approximate data.

`parallel` simulate eigen values of the random matrix.

```
fit = parallel(data, 1000, f = fa); # Return `Parallel` struct
```

Type of `data` is Matrix or DataFrame and its elements, by the default argument,  are assumed to be ordered categorical. If `data` has continuous variables, `...f = x -> fa(x, cor_method = :Pearson)` has to be used.

```
fit = parallel(data, 1000, f = x -> fa(x, cor_method = :Pearson));
```

## Visualization

Plot recipe for `Parallel` is implemented.

```
using Plots
plot(fit)
```