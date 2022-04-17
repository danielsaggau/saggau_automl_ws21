# auto ml tool
#@required: library(mlr3tuning)

#@ input param: terminator
#@ input param: task
#@ input param: learner
#@ input param: filter
#@ output:

search_space = ps()
search_space = ps(
  eta = p_dbl(lower = 1e-04, upper =1),
  lambda = p_dbl(lower = 1e-03 , upper = 1000)
)
#automl = function(task, learner = NULL, filter = NULL ,measure = NULL, terminator = NULL){
benchmarkgrid( # unsure whether combinable

#task = tsks(task))


#instance$result_learner_param_vals
#learner$param_set$values = instance$result_learner_param_vals
#learner$param_set
library("mlr3filters")
library(mlr3fselect)
measures = msrs(c("classif.ce", "time_train"))
evals20 = trm("evals", n_evals = 20)

instance2 = FSelectInstanceMultiCrit$new( task = madeline_tsk,
learner = lrn("classif.xgboost"),
#search_space = search_space,
resampling = rsmp("cv", folds = 10),
measures = measures,
terminator = evals20)
instance2

lgr::get_logger("bbotk")$set_threshold("warn")
fselector$optimize(instance2)

instance2$result

