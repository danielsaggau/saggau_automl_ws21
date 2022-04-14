# auto ml tool


automl = function(task, learner = NULL, measure = NULL, runtime = NULL, terminator = NULL )){

measure = msrs(c("classif.ce"))
#learners = makeLearner("classif.xgboost", eval_metric ="")
learner = list(lrn("classif.xgboost"
)

terminator =
filter flt()
tuner =
}













