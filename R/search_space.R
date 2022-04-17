# @param eta
# @param lambda
# @param lower (`) set based on experience
# @param upper (``) set based on experience
# @return search space

search_space = ps(
  eta = p_dbl(lower = 1e-04, upper =1),
  lambda = p_dbl(lower = 1e-03 , upper = 1000)
)
