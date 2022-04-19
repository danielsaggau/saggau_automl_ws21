# pipeline
# @requirement library(bbtok) require("mlr3pipelines"), library(mlr3tuning),library("magrittr") library("paradox")
# @input param
# @
# @
devtools::install_github("https://github.com/mlr-org/mlr3mbo/")
library(mlr3mbo)
library(data.table)
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

#tuningInstanceSingleCrit$new

instance = TuningInstanceMultiCrit$new(
  task = task,
  learner = glrn,
  resampling = rsmp("cv", folds =10),
  measure = msrs("classif.ce"),
  search_space = search_space,
  terminator = trm("run_time" ,secs = 60)
)
instance
tuner = tnr("grid_search", resolution = 5)
tuner$optimize(instance)


#
library(mlr3hyperband)
instance = tune(
  method ="hyperband",
  task = task,
  learner = glrn ,
  resampling = rsmp("cv", folds =10),
  measure = msrs("classif.ce"),
  search_space = search_space,
  #terminator = trm("run_time" ,secs = 60)
)
library(mlr3tuning)
?TuningInstanceMultiCrit
instance = TuningInstanceMultiCrit$new(
  task = task,
  learner = glrn ,
  resampling = rsmp("cv", folds =10),
  measure = msrs(c("classif.ce","subsample.frac")),
  search_space = search_space,
  terminator = trm("run_time" ,secs = 180)
)

tuner =
tuner$optimize(instance)

archive = instance$archive
archive$nds_selection(n_select=5)
install.packages("emoa")
library(emoa)
?nds_selection()
plot(instance$result_x_search_space$classif.xgboost.nrounds, instance$result_y$classif.ce)


instance$result_learner_param_vals
instance$result$

instance$nds_selection(n_select=2)

as.data.table(instance$result_learner_param_vals)
glrn$param_set$values

graph$train(task) # only on training set
graph$predict(task) # only on test set todo: ensure split works
