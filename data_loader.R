# dataloader

odata = OMLData$new(id = 41144)

print(odata)
print(odata$target_names)
print(odata$feature_names)

mlr3oml::read_arff("https:/old.openml.org/data/v1/download/19335517/madeline.arff")
library(mlr3)
tsk("oml", data_id = 41144)
oml_data = OMLData$new(41144)
oml_data$name
data = oml_data$data
