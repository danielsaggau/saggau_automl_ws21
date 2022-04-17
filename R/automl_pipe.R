# pipeline
# @requirement library(bbtok) require("mlr3pipelines"), library(mlr3tuning),library("magrittr") library("paradox")
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
glrn$param_set$values$variance.filter.frac = 0.25
cv10 = rsmp("cv", folds = 10)
resample(task, glrn, cv10)

search_space = ps(
    variance.filter.frac = p_dbl(lower = 0.25, upper = 1),
    classif.xgboost.lambda = p_dbl(lower = 1e-03 , upper = 100),
    classif.xgboost.eta = p_dbl(lower = 1e-04, upper =1),
    classif.xgboost.nrounds = p_int(lower = 1, upper = 300)
  )

instance = TuningInstanceSingleCrit$new(
  task = task,
  learner = glrn,
  resampling = rsmp("cv", folds =10),
  measure = msrs("classif.ce"),
  search_space = search_space,
  terminator = trm("run_time" ,secs = 60)
)
future::plan("multisession")
instance

tuner = tnr("grid_search", resolution =2)
tuner$optimize(instance)

instance$result_learner_param_vals
as.data.table(instance$result_learner_param_vals)
glrn$param_set$values
op_graph = po("learner", lrn("classif.xgboost"))
op_graph$param_set
glrn = as_learner(gr %>>% op_rpart)
glrn$param_set
gr = op_graph %>>% po("scale")
