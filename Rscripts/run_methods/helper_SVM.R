#to get label assignments for the data points identified as outliers
#using linear SVM
library(LiblineaR)
helper_assign_outliersSVM<-function(bulk_data, out_data, bulk_labels){
  dataset.train=as.data.frame(cbind(bulk_data, bulk_labels))
  system.time(rb <- Rborist(dataset.train[,-ncol(dataset.train)], as.factor(dataset.train[,ncol(dataset.train)]), classWeight = rep(1, length(unique(bulk_labels))), nTree=500))
  system.time(pred <- predict(rb, out_data))
  yPred <- pred$yPred
  return(yPred)
}  