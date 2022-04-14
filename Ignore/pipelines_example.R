# creating a pipeline documentation

library("mlr3pipelines")
as.data.table(mlr_pipeops)
pca = mlr_pipeops$get("pca")
pca = po("pca")
learner = mlr_pipeops$get("learner", lrn("classif.rpart"))


