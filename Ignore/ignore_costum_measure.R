#@requirement library(mlr3)
#@requirement library(mlr3pipelines)
#@requirement library(R6)

MeasureFrac = R6::R6Class("MeasureFrac",
                   inherit = mlr3::MeasureClassif,
                   public = list(
                     initalize = function(){
                       super$initaialize(
                         id = "info.frac",
                         packages = "mlr3pipelines",
                         properties = "requires_task",
                         predict_type ="response",
                         range = c(0, Inf),
                         minimize = TRUE
                       )
                     }
                   ),

                   private = list(
                     .score = function(task, ...){
                       filter = po("filter",
                                   filter = mlr3filters::flt("information_gain"),
                                   param_vals = list(filter.frac = .50))
                       filtered_features = filter$train(list(task))[[1]]
                       #sum(length(filtered_features$feature_names)) / sum(length(task$feature_names))
                       filtered_features
                     }
                   ))
mlr3::mlr_measures$add("info.frac", MeasureFrac)


# this code is from mlr3measures

add_measure = function(obj, title, type, lower, upper, minimize, obs_loss = NA_character_, aggregated = TRUE) {
  id = deparse(substitute(obj))

  ptype = intersect(names(formals(obj)), c("response", "prob", "se"))
  if (length(ptype) == 0L) {
    ptype = NA_character_
  }
  assign(id, list(
    id = id,
    title = assert_string(title),
    type = assert_choice(type, c("binary", "classif", "regr", "similarity")),
    lower = assert_number(lower),
    upper = assert_number(upper),
    predict_type = ptype,
    minimize = assert_flag(minimize, na.ok = TRUE),
    obs_loss = assert_string(obs_loss, na.ok = TRUE),
    aggregated = assert_flag(aggregated),
    sample_weights = "sample_weights" %in% names(formals(obj))
  ), envir = measures)

frac = function(task, ...){
filter = po("filter",
            filter = mlr3filters::flt("information_gain"),
            param_vals = list(filter.frac = .40))
filtered_features = filter$train(list(task))[[1]]
return(filtered_features$feature_names)
}
#sum(length(filtered_features$feature_names)))
#return(sum(length(filtered_features$feature_names)) / sum(length(task$feature_names)))


frac(madelon_tsk)

