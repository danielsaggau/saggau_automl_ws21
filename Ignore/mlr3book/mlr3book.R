
library(mlr3)

data("mtcars", package = "datasets")
data = mtcars[, 1:3]
str(data)

task_mtcars = as_task_regr(data, target = "mpg", id = "cars")
print(task_mtcars)

library("mlr3viz")
autoplot(task_mtcars, type = "pairs")

mlr_tasks

as.data.table(mlr_tasks)

task_penguins = tsk("penguins")
print(task_penguins)


library("mlr3verse")
as.data.table(mlr_tasks)[, 1:4]

help("mlr_tasks_german_credit")

task_penguins$help()

task_mtcars
task_mtcars$data()
task_mtcars$feature_names
summary(as.data.table(task_mtcars))

# during construction
data("Sonar", package = "mlbench")
task = as_task_classif(Sonar, target = "Class", positive = "R")

# switch positive class to level 'M'
task$positive = "M"

print(task_mtcars$col_roles)

# with `keep.rownames`, data.table stores the row names in an extra column "rn"
data = as.data.table(datasets::mtcars[, 1:3], keep.rownames = TRUE)
task_mtcars = as_task_regr(data, target = "mpg", id = "cars")

# there is a new feature called "rn"
task_mtcars$feature_names

names(task_mtcars$col_roles)

# assign column "rn" the role "name", remove from other roles
task_mtcars$set_col_roles("rn", roles = "name")

# note that "rn" not listed as feature anymore
task_mtcars$feature_names

task_penguins = tsk("penguins")
task_penguins$select(c("body_mass", "flipper_length")) # keep only these features
task_penguins$filter(1:3) # keep only these rows
task_penguins$head()

task_penguins$cbind(data.frame(letters = letters[1:3])) # add column letters
task_penguins$head()

library("mlr3viz")

# get the pima indians task
task = tsk("pima")

# subset task to only use the 3 first features
task$select(head(task$feature_names, 3))

# default plot: class frequencies
autoplot(task)

# pairs plot (requires package GGally)
autoplot(task, type = "pairs")

# duo plot (requires package GGally)
autoplot(task, type = "duo")

library("mlr3viz")

# get the complete mtcars task
task = tsk("mtcars")

# subset task to only use the 3 first features
task$select(head(task$feature_names, 3))

# default plot: boxplot of target variable
autoplot(task)

autoplot(task, type = "pairs")


## learner
remotes::install_github("mlr-org/mlr3extralearners")
library(mlr3extralearners)
head(mlr3extralearners::list_mlr3learners()) # show first six learners

mlr_learners

learner = lrn("classif.rpart")
print(learner)

learner$param_set

learner$param_set$values = list(cp = 0.01, xval = 0)
learner

pv = learner$param_set$values
pv$cp = 0.02
pv
learner$param_set$values = pv

learner = lrn("classif.rpart", id = "rp", cp = 0.001)
learner$id

learner$param_set$values


# thresholding

data("Sonar", package = "mlbench")
task = as_task_classif(Sonar, target = "Class", positive = "M")
learner = lrn("classif.rpart", predict_type = "prob")
pred = learner$train(task)$predict(task)

measures = msrs(c("classif.tpr", "classif.tnr")) # use msrs() to get a list of multiple measures
pred$confusion

pred$score(measures)

pred$set_threshold(0.2)
pred$confusion

pred$score(measures)


task = tsk("penguins")
learner = lrn("classif.rpart")

# train test split

train_set = sample(task$nrow, 0.8 * task$nrow)
test_set = setdiff(seq_len(task$nrow), train_set)
learner$model
learner$train(task, row_ids = train_set)
learner$model

prediction = learner$predict(task, row_ids = test_set)
print(prediction)
head(as.data.table(prediction)) # show first six predictions
prediction$confusion

learner$predict_type = "prob"

# re-fit the model
learner$train(task, row_ids = train_set)

# rebuild prediction object
prediction = learner$predict(task, row_ids = test_set)

# data.table conversion
head(as.data.table(prediction)) # show first six

head(prediction$response)
# directly access the matrix of probabilities:
head(prediction$prob)


task = tsk("penguins")
learner = lrn("classif.rpart", predict_type = "prob")
learner$train(task)
prediction = learner$predict(task)
autoplot(prediction)

mlr_measures
measure = msr("classif.acc")
print(measure)

msr("classif.fdr")
msrs(c("classif.fp", "classif.fn"))
prediction$score(measure)

###############################################################################

# performance evaluation and comparison


data("Sonar", package = "mlbench")
task = as_task_classif(Sonar, target = "Class", positive = "M")
learner = lrn("classif.rpart", predict_type = "prob")
pred = learner$train(task)$predict(task)
C = pred$confusion
print(C)


library("mlr3viz")

# TPR vs FPR / Sensitivity vs (1 - Specificity)
autoplot(pred, type = "roc")

install.packages("precrec")
library(precrec)

# Precision vs Recall
autoplot(pred, type = "prc")

library("mlr3verse")

task = tsk("penguins")
learner = lrn("classif.rpart")

as.data.table(mlr_resamplings)

resampling = rsmp("holdout")
print(resampling)

resampling$param_set$values = list(ratio = 0.8)
rsmp("holdout", ratio = 0.8)

resampling$instantiate(task)
str(resampling$train_set(1))


task = tsk("penguins")
learner = lrn("classif.rpart", maxdepth = 3, predict_type = "prob")
resampling = rsmp("cv", folds = 3)

rr = resample(task, learner, resampling, store_models = TRUE)
print(rr)
rr$aggregate(msr("classif.ce"))
rr$score(msr("classif.ce"))

rr$warnings
rr$errors
rr$resampling
rr$resampling$iters

lrn = rr$learners[[1]]
lrn$model

rr$prediction() # all predictions merged into a single Prediction object
rr$predictions()[[1]] # predictions of first resampling iteration

rr$filter(c(1, 3))
print(rr)



# costum splits

resampling = rsmp("custom")
resampling$instantiate(task,
                       train = list(c(1:10, 51:60, 101:110)),
                       test = list(c(11:20, 61:70, 111:120))
)
resampling$iters

resampling$train_set(1)
resampling$test_set(1)

task = tsk("pima")
task$select(c("glucose", "mass"))
learner = lrn("classif.xgboost", predict_type = "prob")
rr = resample(task, learner, rsmp("cv"), store_models = TRUE)

# boxplot of AUC values across the 10 folds
autoplot(rr, measure = msr("classif.auc"))

# ROC curve, averaged over 10 folds
autoplot(rr, type = "roc")

# learner predictions for the first fold
rr$filter(1)
autoplot(rr, type = "prediction")

library("mlr3verse")
install.packages("ranger")
library(ranger)

design = benchmark_grid(
  tasks = tsks(c("spam", "german_credit", "sonar")),
  learners = lrns(c("classif.ranger", "classif.rpart", "classif.featureless"),
                  predict_type = "prob", predict_sets = c("train", "test")),
  resamplings = rsmps("cv", folds = 3)
)
print(design)

bmr = benchmark(design)

measures = list(
  msr("classif.auc", predict_sets = "train", id = "auc_train"),
  msr("classif.auc", id = "auc_test")
)

tab = bmr$aggregate(measures)
print(tab)

library("data.table")
# group by levels of task_id, return columns:
# - learner_id
# - rank of col '-auc_train' (per level of learner_id)
# - rank of col '-auc_test' (per level of learner_id)
ranks = tab[, .(learner_id, rank_train = rank(-auc_train), rank_test = rank(-auc_test)), by = task_id]
print(ranks)


# group by levels of learner_id, return columns:
# - mean rank of col 'rank_train' (per level of learner_id)
# - mean rank of col 'rank_test' (per level of learner_id)
ranks = ranks[, .(mrank_train = mean(rank_train), mrank_test = mean(rank_test)), by = learner_id]

# print the final table, ordered by mean rank of AUC test
ranks[order(mrank_test)]


autoplot(bmr) + ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))

bmr_small = bmr$clone()$filter(task_id = "german_credit")
autoplot(bmr_small, type = "roc")

tab = bmr$aggregate(measures)
rr = tab[task_id == "german_credit" & learner_id == "classif.ranger"]$resample_result[[1]]
print(rr)

measure = msr("classif.auc")
rr$aggregate(measure)

# get the iteration with worst AUC
perf = rr$score(measure)
i = which.min(perf$classif.auc)

# get the corresponding learner and training set
print(rr$learners[[i]])

head(rr$resampling$train_set(i))

task = tsk("iris")
resampling = rsmp("holdout")$instantiate(task)

rr1 = resample(task, lrn("classif.rpart"), resampling)
rr2 = resample(task, lrn("classif.featureless"), resampling)

# Cast both ResampleResults to BenchmarkResults
bmr1 = as_benchmark_result(rr1)
bmr2 = as_benchmark_result(rr2)

# Merge 2nd BMR into the first BMR
bmr1$combine(bmr2)

bmr1

################################################################################

library("mlr3verse")
task = tsk("pima")
print(task)

learner = lrn("classif.rpart")
learner$param_set

search_space = ps(
  cp = p_dbl(lower = 0.001, upper = 0.1),
  minsplit = p_int(lower = 1, upper = 10)
)
search_space

hout = rsmp("holdout")
measure = msr("classif.ce")

library("mlr3tuning")

evals20 = trm("evals", n_evals = 20)

instance = TuningInstanceSingleCrit$new(
  task = task,
  learner = learner,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

tuner = tnr("grid_search", resolution = 5)
tuner$optimize(instance)

instance$result_learner_param_vals
instance$result_y

as.data.table(instance$archive)
instance$archive$benchmark_result


instance$archive$benchmark_result$score(msr("classif.acc"))


learner$param_set$values = instance$result_learner_param_vals
learner$train(task)


measures = msrs(c("classif.ce", "time_train"))

library("mlr3tuning")

evals20 = trm("evals", n_evals = 20)

instance = TuningInstanceMultiCrit$new(
  task = task,
  learner = learner,
  resampling = hout,
  measures = measures,
  search_space = search_space,
  terminator = evals20
)
instance

tuner$optimize(instance)

instance$result_learner_param_vals
instance$result_y



learner = lrn("classif.rpart")
search_space = ps(
  cp = p_dbl(lower = 0.001, upper = 0.1),
  minsplit = p_int(lower = 1, upper = 10)
)
terminator = trm("evals", n_evals = 10)
tuner = tnr("random_search")

at = AutoTuner$new(
  learner = learner,
  resampling = rsmp("holdout"),
  measure = msr("classif.ce"),
  search_space = search_space,
  terminator = terminator,
  tuner = tuner
)
at

at$train(task)





# creating parameter set

library("mlr3verse")

search_space = ps()
print(search_space)

search_space = ps(
  cost = p_dbl(lower = 0.1, upper = 10),
  kernel = p_fct(levels = c("polynomial", "radial"))
)
print(search_space)



search_space = ps(cost = p_dbl(0.1, 10), kernel = p_fct(c("polynomial", "radial")))

search_space = ps(
  cost = p_dbl(-1, 1, trafo = function(x) 10^x),
  kernel = p_fct(c("polynomial", "radial")),
  .extra_trafo = function(x, param_set) {
    if (x$kernel == "polynomial") {
      x$cost = x$cost * 2
    }
    x
  }
)
rbindlist(generate_design_grid(search_space, 3)$transpose())

search_space = ps(
  class.weights = p_dbl(0.1, 0.9, trafo = function(x) c(spam = x, nonspam = 1 - x))
)
generate_design_grid(search_space, 3)$transpose()


search_space = ps(
  cost = p_fct(c(0.1, 3, 10)),
  kernel = p_fct(c("polynomial", "radial"))
)
rbindlist(generate_design_grid(search_space, 3)$transpose())

search_space = ps(
  cost = p_fct(c("0.1", "3", "10"),
               trafo = function(x) list(`0.1` = 0.1, `3` = 3, `10` = 10)[[x]]),
  kernel = p_fct(c("polynomial", "radial"))
)
rbindlist(generate_design_grid(search_space, 3)$transpose())


search_space = ps(
  cost = p_fct(c(0.1, 3, 10)),
  kernel = p_fct(c("polynomial", "radial"))
)
typeof(search_space$params$cost$levels)

search_space = ps(
  class.weights = p_fct(
    list(
      candidate_a = c(spam = 0.5, nonspam = 0.5),
      candidate_b = c(spam = 0.3, nonspam = 0.7)
    )
  )
)
generate_design_grid(search_space)$transpose()


search_space = ps(
  cost = p_dbl(-1, 1, trafo = function(x) 10^x),
  kernel = p_fct(c("polynomial", "radial")),
  degree = p_int(1, 3, depends = kernel == "polynomial")
)
rbindlist(generate_design_grid(search_space, 3)$transpose(), fill = TRUE)

# creating tuning paramset from other paramsets

learner = lrn("classif.svm")
learner$param_set$values$kernel = "polynomial" # for example
learner$param_set$values$degree = to_tune(lower = 1, upper = 3)

print(learner$param_set$search_space())

rbindlist(generate_design_grid(learner$param_set$search_space(), 3)$transpose())

learner$param_set$values$shrinking = to_tune()

print(learner$param_set$search_space())

rbindlist(generate_design_grid(learner$param_set$search_space(), 3)$transpose())


learner$param_set$values$type = "C-classification" # needs to be set because of a bug in paradox
learner$param_set$values$cost = to_tune(c(val1 = 0.3, val2 = 0.7))
learner$param_set$values$shrinking = to_tune(p_lgl(depends = cost == "val2"))

print(learner$param_set$search_space())









################################################################################
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


################################################################################

library("mlr3verse")

learner = lrn("classif.ranger")
learner$param_set$ids(tags = "threads")





