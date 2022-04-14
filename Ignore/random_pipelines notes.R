g = po("imputesample", id = "impute") %>>%
  po("encode", method = "one-hot") %>>%
  po("scale") %>>%
  po("learner", lrn("classif.xgboost"))
g$plot()

glrn = as_learner(g)
resample(tsk, glrn, rsmp("holdout"))

train.idx = sample(seq_len(task$nrow), 120)
test.idx = setdiff(seq_len(task$nrow), train.idx)

pipelrn = as_learner(pipeline)

pipelrn$train(task, train.idx)$
  predict(task, train.idx)$
  score()

po("filter", mlr3filters::flt("information_gain"))
lrn("classif.xgboost")$param_set

tuning_space_rpart = lts("classif.xgboost.default")
tuning_space_rpart$values

# filter

po("filter", mlr3filters::flt("information_gain"))
