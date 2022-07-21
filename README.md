# Agds hyperparameter optimization


Implementation of hyperparameter optimization algorithm based on
Associative Self-adapting Structures, described in: <br>
"S. Czaplak and A. HORZYK, Automatic Optimization of Hyperparameters Using Associative Self-adapting Structures, Proc. of IJCNN 2022 (WCCI 2022), Padwa, Italy, IEEE Xplore, 2022,"

To use it you can simply call:
```python
best_score, best_hyperparameters = run_AGDS_algorithm_minimise(func, max_iter,
                                       n_random_params_per_epoch=15000,
                                       n_initial_population=2, n_population=1,
                                       n_units_for_exploration=1, explore_always=True,
                                       random_param_function=f_random_param)

```
Where:
* func - funciton that you want to minimize
* max_iter - maximum number of function evaluations 
* n_random_params_per_epoch - number of randomly sampled hyperparameters set to test in each epoch
* n_initial_population - size of initial, randomly sampled population
* n_population - how many 'best' examples in each epoch should be tested
* n_units_for_exploration - how many hyperparameters set should be tested for exploration in each epoch
* explore_always - If True, then on each epoch _n_units_for_exploration_ hyperparameters for exploration will be tested,
If false, then exploration is done only if there was no improvement (comparing to previous epoch)
* random_param_function - function to sample one hyperparameter set
