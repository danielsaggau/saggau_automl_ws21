# dataloader
# reference: https://mlr3oml.mlr-org.com/reference/OMLData.html
library(mlr3oml)
oml_data = OMLData$new(41144)
madeline = oml_data$data

madeline_tsk = as_task_classif(
madeline, target = "class")# predict_type = "prob")

split = function(task){
  test.idx = sample(seq_len(task$nrow), 30)
  train.idx = setdiff(seq_len(task$nrow), test.idx)
  task$row_roles$use = train.idx
}

oml_data_madelon = OMLData$new(1485) # fix overwritting
madelon = oml_data_madelon$data

madelon_tsk = as_task_classif(madelon, target = "Class")
madeline_tsk = as_task_classif(madeline, target = "class")
