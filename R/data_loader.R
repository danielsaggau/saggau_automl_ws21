# dataloader
# reference: https://mlr3oml.mlr-org.com/reference/OMLData.html
library(mlr3oml)
oml_data = OMLData$new(41144)
madeline = oml_data$data

madeline_tsk = as_task_classif(
madeline, target = "class",# predict_type = "prob")


test.idx = sample(seq_len(task$nrow), 30)
train.idx = setdiff(seq_len(task$nrow), test.idx)
# Set task to only use train indexes
task$row_roles$use = train.idx

oml_data = OMLData$new(1485)
madelon = oml_data$data

madelon_tsk = as_task_classif(
  madelon, target = "Class")


#######
madeline_tsk = as_task_classif(
  madeline, target = "class")

#madeline_tsk = TaskClassif$new(id ="madeline",
#                              backend = as_data_backend(madeline),
#                              target = "class")


task = madeline_tsk
test.idx = sample(seq_len(task$nrow), 30)
train.idx = setdiff(seq_len(task$nrow), test.idx)
# Set task to only use train indexes
task$row_roles$use = train.idx


#madelon_tsk = TaskClassif$new(id ="madelon",
#                              backend = as_data_backend(madelon),
#                              target = "Class")


####---------------------------------####---------------------------------






