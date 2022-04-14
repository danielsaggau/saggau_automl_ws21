library(mlr3verse)
library(skimr)
library(DiceKriging)
set.seed(7832)

lgr::get_logger("mlr3")$set_threshold("warn")
lgr::get_logger("bbotk")$set_threshold("warn"

# retrieve the task from mlr3
task = tsk("iris")

# generate a quick textual overview using the skimr package
skimr::skim(task$data())

learner = lrn("classif.svm", type = "C-classification", kernel = "radial")

as.data.table(learner$param_set)

learner$param_set$values$cost = to_tune(0.1, 10)
learner$param_set$values$gamma = to_tune(0, 5)

resampling = rsmp("cv", folds = 3)
measure = msr("classif.ce")
terminator = trm("none")

instance = TuningInstanceSingleCrit$new(
  task = task,
  learner = learner,
  resampling = resampling,
  measure = measure,
  terminator = terminator
)

print(instance)
tuner = tnr("grid_search", resolution = 5)
print(tuner)


generate_design_grid(learner$param_set$search_space(), resolution = 5)
tuner$optimize(instance)

autoplot(instance, type = "surface", cols_x = c("cost", "gamma"),
         learner = lrn("regr.km"))


learner = lrn("classif.svm")
learner$param_set$values = instance$result_learner_param_vals
learner$train(task)

learner = lrn("classif.svm", type = "C-classification")
learner$param_set$values$cost = to_tune(p_dbl(1e-5, 1e5, logscale = TRUE))
learner$param_set$values$gamma = to_tune(p_dbl(1e-5, 1e5, logscale = TRUE))
learner$param_set$values$kernel = to_tune(c("polynomial", "radial"))
learner$param_set$values$degree = to_tune(1, 4)

resampling_inner = rsmp("cv", folds = 3)
terminator = trm("none")
tuner = tnr("grid_search", resolution = 3)

at = AutoTuner$new(
  learner = learner,
  resampling = resampling_inner,
  measure = measure,
  terminator = terminator,
  tuner = tuner,
  store_models = TRUE)

resampling_outer = rsmp("cv", folds = 3)
rr = resample(task = task, learner = at, resampling = resampling_outer, store_models = TRUE)

extract_inner_tuning_results(rr)

############################################################################################################################################

library(mlr3pipelines)
library(xgboost)

# retrieve the task from mlr3
task = tsk("pima")

# create data frame with categorized pressure feature
data = task$data(cols = "pressure")
breaks = quantile(data$pressure, probs = c(0, 0.33, 0.66, 1), na.rm = TRUE)
data$pressure = cut(data$pressure, breaks, labels = c("low", "mid", "high"))

# overwrite the feature in the task
task$cbind(data)

# generate a quick textual overview
skimr::skim(task$data())

learner = lrn("classif.xgboost", nrounds = 100, id = "xgboost", verbose = 0)

round(task$missings() / task$nrow, 2)
