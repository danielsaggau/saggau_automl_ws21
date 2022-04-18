# pipeline
# @requirement library(bbtok) require("mlr3pipelines"), library(mlr3tuning),library("magrittr") library("paradox")
# @input param
# @
# @

future::plan("multisession")

set.seed(123)
xgboost = mlr_pipeops$get("learner", lrn("classif.xgboost", eval_metric ="logloss"))
filter = po("filter",
             filter = mlr3filters::flt("variance"),
             param_vals = list(filter.frac = 0.1))

graph =
  filter %>>%
  po("learner",
     learner = lrn("classif.xgboost"))

graph$plot(html = FALSE)

glrn = as_learner(graph)
glrn$param_set$values$variance.filter.frac = 0.25
cv10 = rsmp("cv", folds = 10)
resample(task, glrn, cv10)

source("search_space.R")

#uningInstanceSingleCrit$new

instance = tune(
  method ="mbo",
  task = task,
  learner = glrn,
  resampling = rsmp("cv", folds =10),
  measure = msrs("classif.ce"),
  search_space = search_space,
  #terminator = trm("run_time" ,secs = 60)
)

instance$result_learner_param_vals
as.data.table(instance$result_learner_param_vals)
glrn$param_set$values

graph$train(task) # only on training set
graph$predict(task) # only on test set todo: ensure split works
