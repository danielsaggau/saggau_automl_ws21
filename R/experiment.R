#
source("dataloader.R")
source("automl_pipe.R")
library("magrittr")
install.packages("FSelectorRcpp")
library("FSelectorRcpp")






#po("filter", mlr3filters::flt("variance"), filter.frac = 0.5)
#filter = po("filter", mlr3filters::flt("selected_features",learner = lrn("classif.featureless")))
filter = po("filter", mlr3filters::flt("information_gain"), filter.frac = 0.5)

set.seed(123)
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

r_base$score
MeasureSelectedFeatures$new()
l
scores = r_base$score(msr("selected_features"))
scores[, c("iteration", "selected_features")]
r_base$errors

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

r2 = as_benchmark_result(rr)



r1$combine(r2)
r1$aggregate()


## ggplot to visualize results

ggplot(data = data, aes(x = , y= ))
+ geom_points()

ggplot(data = data, aes(x = , y= ))
+ geom_points()

ggplot(data = data, aes(x = , y= ))
+ geom_points()
+ labs(y = "number of features", x = "misclassification rate")


