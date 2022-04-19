grid = benchmark_grid(tasks = c(madeline_tsk, madelon_tsk), resampling = rsmp("cv", folds =10), learners = c(glrn_madelon,glrn_madeline )
                      result = benchmark(grid)
                      result$aggregate()
