#
source("dataloader.R")
source("objective.R")
source("pareto")

## baseline

graph =
  po("learner",
     learner = lrn("classif.avg"))

graph$predict(test.idx)

## random forest
graph =
  po("learner",
     learner = lrn("classif.randomForest"))


graph$predict(test.idx)


## ggplot to visualize results

ggplot(data = data, aes(x = , y= ))
+ geom_points()

ggplot(data = data, aes(x = , y= ))
+ geom_points()

ggplot(data = data, aes(x = , y= ))
+ geom_points()



