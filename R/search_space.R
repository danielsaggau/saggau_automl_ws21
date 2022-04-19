# @hyperparam eta
# @hyperparam lambda
# @phyperaram nrounds
# @hyperparam informationgain.filter frac
# @return search space

search_space = ps(
  classif.xgboost.lambda = p_dbl(lower = 1e-03 , upper = 100),
  classif.xgboost.eta = p_dbl(lower = 1e-04, upper =1),
  classif.xgboost.nrounds = p_int(lower = 1, upper = 30, tags ="budget"), # defining budget parameter
  information_gain.filter.frac = p_dbl(lower =0.25, upper = 1)
  )
