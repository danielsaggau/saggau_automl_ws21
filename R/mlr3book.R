library("mlr3pipelines")
library(mlr3)
as.data.table(mlr_pipeops)
pca = mlr_pipeops$get("pca")
learner = mlr_pipeops$get("learner", lrn("classif.xgboost"))
po("filter", mlr3filters::flt("variance"), filter.frac = 0.5)
library("magrittr")

gr = po("scale") %>>% po(learner)
gr$plot(html = FALSE)

mutate = po("mutate")

filter = po("filter",
            filter = mlr3filters::flt("variance"),
            param_vals = list(filter.frac = 0.5))
graph = mutate %>>% filter

graph = Graph$new()$
  add_pipeop(mutate)$
  add_pipeop(filter)$
  add_edge("mutate", "variance") # add connection mutate -> filter

graph$plot()

#graph$add_pipeop(po("pca"))
#graph$add_pipeop(po("pca", id = "pca2"))
graph$add_pipeop(po(learner))
graph$plot()


mutate = po("mutate")
filter = po("filter",
            filter = mlr3filters::flt("variance"),
            param_vals = list(filter.frac = 0.5))

graph = mutate %>>%
  filter %>>%
  po("learner",
     learner = lrn("classif.rpart"))

task = tsk("iris")
graph$train(task)

graph$predict(task)

glrn = as_learner(graph)


cv10 = rsmp("cv", folds = 10)
resample(task, glrn, cv10)

glrn$param_set$values$variance.filter.frac = 0.25
cv3 = rsmp("cv", folds = 3)
resample(task, glrn, cv3)

library("paradox")
ps = ps(
  classif.rpart.cp = p_dbl(lower = 0, upper = 0.05),
  variance.filter.frac = p_dbl(lower = 0.25, upper = 1)
)

library("mlr3tuning")
instance = TuningInstanceSingleCrit$new(
  task = task,
  learner = glrn,
  resampling = rsmp("holdout"),
  measure = msr("classif.ce"),
  search_space = ps,
  terminator = trm("evals", n_evals = 20)
)

tuner = tnr("random_search")
tuner$optimize(instance)

instance$result_learner_param_vals
instance$result_y


graph = po("branch", c("nop", "pca", "scale")) %>>%
  gunion(list(
    po("nop", id = "null1"),
    po("pca"),
    po("scale")
  ))

graph$plot(html = FALSE)


(graph %>>% po("unbranch", c("nop", "pca", "scale")))$plot(html = FALSE)

# List of pipeops
opts = list(po("nop", "no_op"), po("pca"), po("scale"))
# List of po ids
opt_ids = mlr3misc::map_chr(opts, `[[`, "id")
po("branch", options = opt_ids) %>>%
  gunion(opts) %>>%
  po("unbranch", options = opt_ids)

task = tsk("iris")
train.idx = sample(seq_len(task$nrow), 120)
test.idx = setdiff(seq_len(task$nrow), train.idx)


single_pred = po("subsample", frac = 0.7) %>>%
  po("learner", lrn("classif.rpart"))

pred_set = ppl("greplicate", single_pred, 10L)

bagging = pred_set %>>%
  po("classifavg", innum = 10)

bagging$plot(html = FALSE)

baglrn = as_learner(bagging)
baglrn$train(task, train.idx)
baglrn$predict(task, test.idx)

lrn = lrn("classif.rpart")
lrn_0 = po("learner_cv", lrn$clone())
lrn_0$id = "rpart_cv"


level_0 = gunion(list(lrn_0, po("nop")))
combined = level_0 %>>% po("featureunion", 2)


stack = combined %>>% po("learner", lrn$clone())
stack$plot(html = FALSE)


stacklrn = as_learner(stack)
stacklrn$train(task, train.idx)
stacklrn$predict(task, test.idx)

library("magrittr")
library("mlr3learners") # for classif.glmnet
library(glmnet)


rprt = lrn("classif.rpart", predict_type = "prob")
glmn = lrn("classif.glmnet", predict_type = "prob")

#  Create Learner CV Operators
lrn_0 = po("learner_cv", rprt, id = "rpart_cv_1")
lrn_0$param_set$values$maxdepth = 5L
lrn_1 = po("pca", id = "pca1") %>>% po("learner_cv", rprt, id = "rpart_cv_2")
lrn_1$param_set$values$rpart_cv_2.maxdepth = 1L
lrn_2 = po("pca", id = "pca2") %>>% po("learner_cv", glmn)

# Union them with a PipeOpNULL to keep original features
level_0 = gunion(list(lrn_0, lrn_1, lrn_2, po("nop", id = "NOP1")))

# Cbind the output 3 times, train 2 learners but also keep level
# 0 predictions
level_1 = level_0 %>>%
  po("featureunion", 4) %>>%
  po("copy", 3) %>>%
  gunion(list(
    po("learner_cv", rprt, id = "rpart_cv_l1"),
    po("learner_cv", glmn, id = "glmnt_cv_l1"),
    po("nop", id = "NOP_l1")
  ))

# Cbind predictions, train a final learner
level_2 = level_1 %>>%
  po("featureunion", 3, id = "u2") %>>%
  po("learner", rprt, id = "rpart_l2")

# Plot the resulting graph
level_2$plot(html = FALSE)

task = tsk("iris")
lrn = as_learner(level_2)


lrn$
  train(task, train.idx)$
  predict(task, test.idx)$
  score()


po("filter", mlr3filters::flt("information_gain"))
task = as_task_classif(iris, target = "Species")
lrn = lrn("classif.rpart")
rsmp = rsmp("holdout")
resample(task, lrn, rsmp)
po = po("pca")
po$train(list(task))[[1]]$data()

single_line_task = task$clone()$filter(1)
po$predict(list(single_line_task))[[1]]$data()

po$state

op_boost = po("learner",lrn("classif.xgboost"))
#op_pca$param_set
#op_pca$param_set$values

glrn = as_learner(gr %>>% op_boost)
glrn$param_set
