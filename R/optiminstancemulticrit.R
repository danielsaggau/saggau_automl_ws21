#
# @ input Param
# @ input Param
# @ input Param
# @ input Param
# @ input Param
# @ return (`list()`) with performance estimate and number of features

source("search_space.R")

# resampling for graph learner
learner$param_set$values$information_gain.filter.frac = 0.25
cv3 = rsmp("cv", folds = 10)
resample(task, learner, cv3)





objective = Objective$new(
  id = "filter_xgboost",
  properties = character(),
  #domain,
  codomain = ,
  constants = ps(),
  check_values = TRUE
)



OptimInstanceMultiCrit$new(
  objective = objective,
  search_space = search_space,
  terminator = trm(runtime = , secs= ),
  keep_evals = "all",
  check_values = TRUE )
