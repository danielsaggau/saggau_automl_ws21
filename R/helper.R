# helpers
#Function: plot_pareto
# @ Description: Function to plot the pareto set
# @ param : best which is equal Archive$best(``) result from `bbotk from Instance$Archive
# @ output ggplot graph

plot_pareto = function(best){
  ggplot(data = best,
         aes(y = best$information_gain.filter.frac,
             x= best$classif.ce)) +
    geom_point()
}


# Function plot_nds
# @ Description  plot the results of the evolutionary algorithm
# @ param : select which is the  Instance$Archive$nds_selection() output
# @ output ggplot graph
plot_nds = function(select){
  ggplot(data = select,
         aes(frac,
             x= select$classif.ce,
             y = select$information_gain.filter)) +
    geom_point()

}

# asign values of optimization to learner
# @ Description
# @ param: instance object from `bbotk
# @ param: here we use the output by our automl function
# @ output: learner with assigned weights

assign = function(instance){
  xgboost <- mlr_pipeops$get("learner", lrn("classif.xgboost"))
  subsample <- po("subsample", frac = 0.7) # use example value of mlr3 book
  filter <- po("filter",
               filter = mlr3filters::flt("information_gain"),
               param_vals = list(filter.frac = 0.25)
  )
  graph <-
    subsample %>>%
    filter %>>%
    po("learner",
       learner = lrn("classif.xgboost")
    )
  glrn <- as_learner(graph)
  glrn$param_set$values$classif.xgboost.lambda <- instance$result_x_search_space$classif.xgboost.lambda
  glrn$param_set$values$subsample.frac <- instance$result_x_search_space$subsample.frac
  glrn$param_set$values$information_gain.filter.frac <- instance$result_x_search_space$information_gain.filter.frac
  glrn$param_set$values$classif.xgboost.eta <- instance$result_x_search_space$classif.xgboost.eta
  glrn
}

# Function: Eval as score output
# @ param task: respective task in R6 Class
# @ param learner: respective learner in R6 Class
# @ output resampling score results

eval <- function(task, learner) {
  filter <- po("filter", mlr3filters::flt("information_gain"), filter.frac = 0.5)
  graph <- filter %>>% po("learner", learner = lrn(learner))
  graph_l <- as_learner(graph)
  cv10 <- rsmp("cv")
  task <- task
  r <- resample(task, graph_l, cv10, store_models = TRUE)
  r = r$score()
}

# @ param task
# @ param tuner
# output resample output format
# eval as resampling result
eval_resample <- function(task, learner) {
  filter <- po("filter", mlr3filters::flt("information_gain"), filter.frac = 0.5)
  graph <- filter %>>% po("learner", learner = lrn(learner))
  graph_l <- as_learner(graph)
  cv10 <- rsmp("cv")
  task <- task
  r <- resample(task, graph_l, cv10, store_models = TRUE)
}
