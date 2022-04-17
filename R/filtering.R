# @input param: task
# @input param: learner
# @input param: filter
# @input param:
# @output: results (`data.table`) with importance scores for the different features

filtering_generic = function(task, learner, filter){
  task = task
  learner = lrn(learner)
  filter = flt(filter)
  result = as.data.table(filter$calculate(task))
  result
}

#filtering(learner = "classif.xgboost", task = madeline_tsk, filter = "auc")

filtering_adv = function(task, learner, measures, resampling){
  instance = FSelectInstanceMultiCrit$new(
    task = task,
    learner = lrn(learner),
    resampling = rsmp(resampling),
    measures = msrs(measures),
    terminator = trm("evals", n_evals = 1)
  )
lgr::get_logger("bbotk")$set_threshold("warn")
fselector$optimize(instance)
}

filtering_adv(task = madeline_tsk,
              learner = "classif.xgboost" ,
              measures = c("classif.ce", "time_train"),
              resampling = "cv"
)

# compare results of subset with ull feature set # todo
grid = benchmark_grid()

