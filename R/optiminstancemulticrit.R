# @ require bbotk , mlr3 , mlr3pipelines
# @ input Param
# @ input Param
# @ input Param
# @ input Param
# @ input Param
# @ return (`list()`) with performance estimate and number of features

# call our search space
source("search_space.R")

# call the correct learner

future::plan("multisession")

set.seed(123)
xgboost = mlr_pipeops$get("learner", lrn("classif.xgboost", eval_metric ="logloss"))
subsample = po("subsample", frac = 0.7) # use example value of mlr3 book
filter = po("filter",
            filter = mlr3filters::flt("information_gain"),
            param_vals = list(filter.frac = 0.1))

graph =
  subsample %>>%
  filter %>>%
  po("learner",
     learner = lrn("classif.xgboost"))

graph$plot(html = FALSE)

glrn = as_learner(graph)
glrn$param_set$values$information_gain.filter.frac = 0.25
cv10 = rsmp("cv", folds = 10)

# resampling for graph learner
cv10 = rsmp("cv", folds = 10)

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
           trm("evals", n_evals = 10)), any = FALSE )

  )

tuner = opt("hyperband")
x = tuner$optimize(instance)
x$subsample.frac
