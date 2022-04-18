# helpers
# @input param: Archive results from bbotk
# get hypervol

get_hypervol_1 = function(Archive){
Hypervolume = Archive$nds_selection(batch = NULL, n_select = 1, ref_point = NULL)
  }

get_hypervol_5 = function(Archive){
  Hypervolume = Archive$nds_selection(batch = NULL, n_select = 5, ref_point = NULL)
}
get_hypervol_10 = function(Archive){
  Hypervolume = Archive$nds_selection(batch = NULL, n_select = 10, ref_point = NULL)
}

# get pareto set

get_pareto = function(Archive){
  pareto =  Archive$best(batch = NULL, n_select = 1)
}

get_pareto_5 = function(Archive){
  pareto =  Archive$best(batch = NULL, n_select = 5)
}

get_pareto_10 = function(Archive){
pareto =  Archive$best(batch = NULL, n_select = 10)
}
