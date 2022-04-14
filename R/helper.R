# helpers

# get hypervol

get_hypervol = function(Archive){
Hypervolume = Archive$nds_selection(batch = NULL, n_select = 1, ref_point = NULL)
  }

# get pareto set

get_pareto = function(Archive){
pareto =  Archive$best(batch = NULL, n_select = 1)
}
