# auto ml tool
library(mlr3tuning)

automl = function(task, learner = NULL, measure = NULL, runtime = NULL, terminator = NULL )){
measure = msrs(c("classif.ce"))
#learners = makeLearner("classif.xgboost", eval_metric ="")
learner = lrn("classif.xgboost"),
terminator =  trm("none"),
filter flt(),
tuner = tnr("grid_search"),
resampling = rsmp("cv", folds = 10)
}


