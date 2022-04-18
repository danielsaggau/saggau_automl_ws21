# dataloader
# reference: https://mlr3oml.mlr-org.com/reference/OMLData.html
library(mlr3oml)
oml_data = OMLData$new(41144)
madeline = oml_data$data

madeline_tsk = as_task_classif(
madeline, target = "class",# predict_type = "prob")

split = function(task){
  test.idx = sample(seq_len(task$nrow), 30)
  train.idx = setdiff(seq_len(task$nrow), test.idx)
  task$row_roles$use = train.idx
}

split(madelon_tsk)

oml_data = OMLData$new(1485) # fix overwritting
madelon = oml_data$data

madelon_tsk = as_task_classif(madelon, target = "Class")
madeline_tsk = as_task_classif(madeline, target = "class")

task = madeline_tsk
test.idx = sample(seq_len(task$nrow), 30)
train.idx = setdiff(seq_len(task$nrow), test.idx)
# Set task to only use train indexes
task$row_roles$use = train.idx


