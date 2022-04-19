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

automl = function(task){
xgboost = mlr_pipeops$get("learner", lrn("classif.xgboost", eval_metric ="logloss"))
subsample = po("subsample", frac = 0.7) # use example value of mlr3 book
filter = po("filter",
            filter = mlr3filters::flt("information_gain"),
            param_vals = list(filter.frac = 0.1, stratify = TRUE))

graph =
  subsample %>>%
  filter %>>%
  po("learner",
     learner = lrn("classif.xgboost"))

graph$plot(html = FALSE)

glrn = as_learner(graph)
glrn$param_set$values$information_gain.filter.frac = 0.25 # set minimum as starting value
cv10 = rsmp("cv", folds = 10)

# resampling for graph learner
cv10 = rsmp("cv", folds = 10)


mlr3measures::

objective = ObjectiveTuning$new(
  task = task,
  learner = glrn,
  measures = msrs(c("classif.ce")),
  resampling = cv10,
  check_values = TRUE
)
instance = OptimInstanceMultiCrit$new(
  objective = objective,
  search_space = search_space,
  keep_evals = "all",
  terminator = trm("combo",
      list(trm("clock_time", stop_time = Sys.time() + 60),
           trm("evals", n_evals = 100)), any = FALSE )
  )

tuner = opt("hyperband")
tuner$optimize(instance)
}

madelon = automl(madelon_tsk)

glrn$param_set$values$subsample.frac= 1
glrn$param_set$values$classif.xgboost.eta = 0.8620727
instance$result_x_domain

instance$archive$best(n_select=10)
instance$archive$nds_selection(n_select =10)

