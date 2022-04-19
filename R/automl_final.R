# automl function
# requires mlr3pipelines, mlr3, bbotk, mlr3tuning
# @param task(``)
# @param
# @param
# Method Tuner: Hyperband
# Note: budget parameter specified in search space

automl = function(task){
  xgboost = mlr_pipeops$get("learner", lrn("classif.xgboost"))
  subsample = po("subsample", frac = 0.7) # use example value of mlr3 book
  filter = po("filter",
              filter = mlr3filters::flt("information_gain"),
              param_vals = list(filter.frac = 0.1))
  graph =
    subsample %>>%
    filter %>>%
    po("learner",
       learner = lrn("classif.xgboost"))

#  graph$plot(html = FALSE) # optional plot the graph

  glrn = as_learner(graph)
  glrn$param_set$values$information_gain.filter.frac = 0.25 # set minimum as starting value
  cv10 = rsmp("cv", folds = 10) # set cross validation 10 fold

#define objective function
  objective = ObjectiveTuning$new(
    task = task,
    learner = glrn,
    measures = msr("classif.ce"),
    resampling = cv10,
    check_values = TRUE
  )

  instance = OptimInstanceMultiCrit$new(
    objective = objective,
    search_space = search_space,
    keep_evals = "all",
    terminator = trm("combo",
                     list(trm("clock_time", stop_time = Sys.time() + 60),
                          trm("evals", n_evals = 10)), any = FALSE)
  )

  tuner = opt("hyperband")
  tuner$optimize(instance)
  return(instance$archive)
}
