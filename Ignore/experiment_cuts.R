
learner = mlr_pipeops$get("learner", lrn("classif.ranger"))

graph$add_pipeop(po(learner))
graph = Graph$new()$
  graph$add_pipeop(po("learner",id = "randomforest", learner = lrn("classif.ranger")))

graph$plot()

#graph
rf = lrn("classif.ranger", predict_type ="prob")
base = lrn("classif.featureless", predict_type = "prob")
lrn_rf = po("learner_cv", rf, id = "rf_1")
lrn_base = po("learner_cv", base, id = "base_1") # learner cv ok=? todo: check
level_0 = gunion(list(lrn_rf, lrn_base))
level_0$plot()

