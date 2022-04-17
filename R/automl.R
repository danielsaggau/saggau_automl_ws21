# auto ml tool
#@required: library(mlr3tuning)

#@ input param: terminator
#@ input param: task
#@ input param: learner
#@ input param: filter
#@ output:

automl = function(task, learner = NULL, filter = NULL ,measure = NULL, terminator = NULL){
benchmarkgrid( # unsure whether combinable

task = tsks(task))

learner = lrn("classif.xgboost", # based on default values
              eta = to_tune(lower = 1e-04, upper =1 , logscale = TRUE),
              nrounds = to_tune(lower = 1e+00, upper = 5000, logscale= FALSE),
              max_depth = to_tune(lower = 1e+00, upper 20= , logscale =FALSE),
              colsample_bytree = to_tune(lower = 1e-01, upper=1, logscale= FALSE),
              colsample_bylevel = to_tune(lower = 1e-01, upper= 1, logscale = FALSE),
              lambda = to_tune(lower = 1e-03 , upper = 1000, logscale = TRUE),
              alpha = to_tune(lower = 1e-03, upper = 1000, logscale = TRUE),
              subsample= to_tune(lower = 1e-01, upper = 1, logscale = FALSE)
)

measures = msrs(measures)

terminator =  terminator

filter = flt(filter)

tuner = tnr("grid_search", batch_size =10)

resampling = rsmps("cv", folds = 10)

instance = tune(
  method = tuner,
  task = task,
  leaarner = learner,
  resampling = resampling,
  measure = measures,
  term_evals = 50
)


instance$results

learner$param_set$values = instance$result_learner_param_vals

#learner$train(task) make sense here ?





}

automl(
task = c(madeline,madelon),
learner = "xgboost",
measures = c("classif_auc","time_train"),
terminator = evals20,
filter = "auc"
)

