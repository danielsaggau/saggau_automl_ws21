search_space = ps(
  eta = lower = , upper =
  nrounds =  lower =  , upper =
  max_depth = lower =  , upper =
  colsample_bytree = lower =  , upper =
  lambda = lower =  , upper =
  alpha = lower =  , upper =
  subsample  =  lower =  , upper =
)

library("data.table")

# print keys and learners
as.data.table(mlr_tuning_spaces)
tuning_space = lts("classif.xgboost.default")
tuning_space$values



# get learner and set learner$param_set

as.data.table(graph_learner$param_set)
