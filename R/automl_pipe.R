# pipeline
# @requirement require("mlr3pipelines"), library("magrittr")
as.data.table(mlr_pipeops)
xgboost = mlr_pipeops$get("learner", lrn("classif.xgboost"))
filter = po("filter",
             filter = mlr3filters::flt("variance"),
             param_vals = list(filter.frac = 0.1))
tuner = tnr("grid_search")

gr = filter %>>% xgboost
gr$plot(html = FALSE)

graph =
  filter %>>%
  po("learner",
     learner = lrn("classif.xgboost"))
task = tsk_madeline
graph$train(task)
graph$predict(task)

glrn = as_learner(graph)
cv10 = rsmp("cv", folds = 10)
resample(task, glrn, cv10)

glrn$param_set$values$variance.filter.frac = 0.25
cv10 = rsmp("cv", folds = 10)
resample(task, glrn, cv10)


search_space =


library("paradox")
instance = TuningInstanceMultiCrit$new(
  task = task,
  learner = glrn,
  resampling = rsmp("holdout"),
  measure = msrs(c("classif.ce", "time_train")),
  search_space = search_space,
  terminator = trm("evals", n_evals = 20)
)

tuner = tnr("random_search")
tuner$optimize(instance)
