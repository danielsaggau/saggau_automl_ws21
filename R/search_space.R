# @param lower (`) set based on experience
# @param upper (``) set based on experience
# @return search space


search_space = ps(
  cp = p_dbl(lower = lower, upper = upper),
  minsplit = p_int(lower = 1, upper = 10)
)
search_space

