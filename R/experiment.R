#
source("data_loader.R")
source("automl_pipe.R")
library("magrittr")
install.packages("FSelectorRcpp")
library("FSelectorRcpp")
library(mlr3)
library(mlr3pipelines)
eval = function(task, learner){
  filter = po("filter", mlr3filters::flt("information_gain"), filter.frac = 0.5)
  graph = filter %>>% po("learner", learner = lrn(learner))
  graph_l =  as_learner(graph)
  cv10 = rsmp("cv", folds = 10)
  task = task
  r_base = resample(task, graph_l, cv10, store_models = TRUE)
  r_base$aggregate(msrs("classif.ce"))
  as_benchmark_result(r_base)
}

r_base$aggregate()
base_madeline = eval(task = madeline_tsk, learner = "classif.featureless")
base_madelon = eval(task = madelon_tsk, learner = "classif.featureless")
ranger_madeline = eval(task = madeline_tsk, learner = "classif.ranger")
ranger_madelon = eval(task = madelon_tsk, learner = "classif.ranger")


base_madeline$combine(base_madelon)
base_madeline$combine(ranger_madelon)
base_madeline$combine(ranger_madeline)
base_madeline$aggregate()

set.seed(123)

scores = base_madeline$aggregate(measures = msrs("subsample.frac"))

scores[, c("iteration", "selected_features")]

subsample.frac

## baseline
graph = filter %>>% po("learner", learner = lrn("classif.featureless"))
graph_l =  as_learner(graph)

#graph_l$train(task, train.idx)$
#  predict(task, test.idx)$
#  score()

#task = tsks(c("madeline_tsk","madelon_tsk"))

cv10 = rsmp("cv", folds = 10)
r_base = resample(task, graph_l, cv10,store_models = TRUE)
r_base$aggregate(msrs("classif.ce"))

#scores = r_base$score(msr("selected_features"))
#scores[, c("iteration", "selected_features")]

r1 = as_benchmark_result(r_base)

## random forest
graph =
  po("learner",
     learner = lrn("classif.ranger"))

graph_l =  as_learner(graph)
graph_l$train(task, train.idx)$
  predict(task, test.idx)$
  score()

cv10 = rsmp("cv", folds = 10)
rr = resample(task, graph_l, cv10)
rr$aggregate(msr("classif.ce"))


## autoplot

## ggplot to visualize results

ggplot(data = data, aes(x = , y= ))
+ geom_points()

ggplot(data = data, aes(x = , y= ))
+ geom_points()

ggplot(data = data, aes(x = , y= ))
+ geom_points()
+ labs(y = "number of features", x = "misclassification rate")
