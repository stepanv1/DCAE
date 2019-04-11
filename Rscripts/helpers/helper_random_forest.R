#to get label assignments for the data points identified as outliers
#using random forest
library(Rborist)
helper_assign_outliers<-function(bulk_data, out_data, bulk_labels){
  #system.time(rb <- Rborist(bulk_data, as.factor(bulk_labels), classWeight = rep(1, length(unique(bulk_labels))), nTree=500))
  classWeight = rep(1, length(unique(bulk_labels)))
  system.time(rb <- Rborist(bulk_data, as.factor(bulk_labels), classWeight = rep(1, length(unique(bulk_labels))), nTree=500))
  system.time(pred <- predict(rb, out_data))
  yPred <- pred$yPred
  return(yPred)
}  