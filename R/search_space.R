# @param eta
# @param lambda
# @param lower (`) set based on experience
# @param upper (``) set based on experience
# @return search space

#search_space = ps(
#  eta = p_dbl(lower = 1e-04, upper =1),
#  lambda = p_dbl(lower = 1e-03 , upper = 1000)
#)

search_space = ps(
  variance.filter.frac = p_dbl(lower = 0.25, upper = 1),
  classif.xgboost.lambda = p_dbl(lower = 1e-03 , upper = 100),
  classif.xgboost.eta = p_dbl(lower = 1e-04, upper =1),
  classif.xgboost.nrounds = p_int(lower = 1, upper = 30, tags ="budget") # defining budget parameter
)
