# hpo

learner = lrn("classif.xgboost", # based on default values
              eta = to_tune(lower = 1e-04, upper =1 , logscale = TRUE),
              #              nrounds = to_tune(lower = 1e+00, upper = 5000, logscale= FALSE),
              #              max_depth = to_tune(lower = 1e+00, upper =20 , logscale =FALSE),
              #              colsample_bytree = to_tune(lower = 1e-01, upper = 1, logscale= FALSE),
              #              colsample_bylevel = to_tune(lower = 1e-01, upper= 1, logscale = FALSE),
              lambda = to_tune(lower = 1e-03 , upper = 1000, logscale = TRUE)
              #              alpha = to_tune(lower = 1e-03, upper = 1000, logscale = TRUE)
              #              subsample= to_tune(lower = 1e-01, upper = 1, logscale = FALSE)
)

measures = msrs(measures)
terminator =  terminator

filter = flt(filter)
tuner = tnr("grid_search", batch_size =10)
resampling = rsmps("cv", folds = 10)

instance = tune(
  method =
    task = task,
  leaarner = learner,
  resampling = resampling,
  measure = measures,
  term_evals = 50
)

instance$results
learner$param_set$values = instance$result_learner_param_vals
learner$param_set

##############################################################################

evals20 = trm("evals", n_evals = 20)
instance = tune(
  task = madeline_tsk,
  method = "grid_search" ,
  learner = learner,
  resampling = rsmp("cv", folds = 10),
  measure = msrs(c("time_train", "classif.fp")),
  terminator = "",
  term_evals = 2,
  batch_size = 2
)
