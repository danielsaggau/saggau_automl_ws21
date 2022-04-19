library("magrittr")
library(mlr3)
library(mlr3tuning)
library(mlr3pipelines)
library(ranger)
library(mlr3viz)
library(patchwork)
library(ggplot2)
library(bbotk)
library(mlr3)
require("paradox")

# Call other files

source("./R/data_loader.R")
source("./R/search_space.R")
source("./R/automl_final.R")
source("./R/helper.R")
set.seed(123)

# Automl Function, taking task as argument
madelon <- automl(madelon_tsk)
learn_madelon <- assign(madelon)
# select pareto set and hypervolume via bbotk
madelon <- madelon$archive
madelon_best <- madelon$best(n_select = 10) # selet 10 points for pareto set; note that actually if the multicrit worked, there should be more than 1 solution but it seems that it did not work
madelon_select <- madelon$nds_selection(n_select = 10) # select 10 points via evolutionary algorithm for multi-crit optimization and determining hypervolume

# plot pareto front ; call helper function plot_pareto(``)
plot_pareto(madelon_best) + labs(title = "Pareto Set for Madelon", subtitle="Pareto Set Determined via the `bbotk` package", y = "Relative number of feature measured via `frac.` and information gain filter", x = "Classification Error")

# plot hypervolume; call helper function plot_ds(``)
plot_nds(madelon_select)  + labs(title = "NDS Algorithm for Madelon", subtitle ="10 best Configurations using the NDS Algorithm provided by `bbotk`", y = "Relative number of feature measured via frac.and information gain filter", x = "Classification Error")

madeline <- automl(madeline_tsk)
learn_madeline <- assign(madeline)
madeline = madeline$archive
madeline_best <- madeline$best(n_select = 10)
madeline_select <- madeline$nds_selection(n_select = 10)

#  plot pareto front; call helper function plot_pareto(``)
plot_pareto(madeline_best) + labs(title = "Pareto Set For Madeline", subtitle="Pareto Set Determined via the `bbotk` package", y = "Relative number of feature measured via `frac.` and information gain filter", x = "Classification Error")
# plot hypervolume; call helper function plot_ds(``)
plot_nds(madeline_select) + labs(title = "NDS Algorithm For Madeline", subtitle ="10 best Configurations using the NDS Algorithm provided by `bbotk`", y = "Relative number of feature measured via frac.and information gain filter", x = "Classification Error")

# function: resample
# input param: resampling strategy rsmp(``)
# input param
cv10 <- rsmp("cv", folds = 10)
boost_madelon <- resample(cv10, learner = learn_madelon, task = madelon_tsk, store_models = TRUE)
boost_madeline <- resample(cv10, learner = learn_madeline, task = madeline_tsk, store_models = TRUE)
boost_madelon_score <- boost_madelon$score()
boost_madeline_score <- boost_madeline$score()


base_madeline <- eval(task = madeline_tsk, learner = "classif.featureless")
base_madelon <- eval(task = madelon_tsk, learner = "classif.featureless")
ranger_madeline <- eval(task = madeline_tsk, learner = "classif.ranger")
ranger_madelon <- eval(task = madelon_tsk, learner = "classif.ranger")

ggplot() +
  geom_point(data = base_madeline, aes(base_madeline$iteration, base_madeline$classif.ce, colour = "base"), size = 4) +
  geom_point(data = ranger_madeline, aes(ranger_madeline$iteration, ranger_madeline$classif.ce, colour = "ranger"), size = 4) +
  geom_point(data = boost_madeline_score, aes(boost_madeline_score$iteration, boost_madeline_score$classif.ce, color = "boost"), size = 4) +
  labs(x = "Iteration", y = "Classification Error", title = "Madeline Data Set")

ggplot() +
  geom_point(data = base_madelon, aes(base_madelon$iteration, base_madelon$classif.ce, colour = "base"), size = 4) +
  geom_point(data = ranger_madelon, aes(ranger_madelon$iteration, ranger_madelon$classif.ce, colour = "ranger"), size = 4) +
  geom_point(data = boost_madelon_score, aes(boost_madelon_score$iteration, boost_madelon_score$classif.ce, color = "boost"), size = 4) +
  labs(x = "Iteration", y = "Classification Error", title = "Madelon Data Set")

# Benchmark

base_madeline_rr <- eval_resample(task = madeline_tsk, learner = "classif.featureless")
base_madelon_rr <- eval_resample(task = madelon_tsk, learner = "classif.featureless")
ranger_madeline_rr <- eval_resample(task = madeline_tsk, learner = "classif.ranger")
ranger_madelon_rr <- eval_resample(task = madelon_tsk, learner = "classif.ranger")

base_madeline_bm <- as_benchmark_result(base_madeline_rr)
ranger_madeline_bm <- as_benchmark_result(ranger_madeline_rr)
base_madelon_bm <- as_benchmark_result(base_madelon_rr)
ranger_madelon_bm <- as_benchmark_result(ranger_madelon_rr)
boost_madeline_score_bm <- as_benchmark_result(boost_madeline)
boost_madelon_score_bm <- as_benchmark_result(boost_madelon)

base_madeline_bm$combine(base_madelon_bm)
base_madeline_bm$combine(ranger_madelon_bm)
base_madeline_bm$combine(ranger_madeline_bm)
base_madeline_bm$combine(boost_madeline_score_bm)
base_madeline_bm$combine(boost_madelon_score_bm)
autoplot(base_madeline_bm)
