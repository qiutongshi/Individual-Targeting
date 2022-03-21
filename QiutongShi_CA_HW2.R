install.packages("/Users/charlotteshi/Desktop",repos=NULL,type="source")
install.packages("caret", dependencies = TRUE) 
install.packages("gower")

library(caret)
library(pROC) # for ROC curves

ds <- read.csv("D2.3 Blue apron.csv")  # load data  

#set menu and frequency as factors
ds$menu = as.factor(ds$menu)  
ds$frequency = as.factor(ds$frequency)  

#prepare train and test dataset for regression
set.seed(2)
idx = sample(2,nrow(ds),replace=TRUE,prob=c(.7,.3)) 

#Task 1: churn as outcome
#specification 1
glm.fit1 = glm(churn~tenure+rating+partysize+urban+menu+frequency, family=binomial,data=ds[idx==1,])
summary(glm.fit1)
#specification 2
glm.fit2 = glm(churn~tenure+rating+partysize+urban+menu+frequency+rating*partysize+rating*urban+
                 partysize*urban+urban*tenure, family=binomial,data=ds[idx==1,])
summary(glm.fit2)


#classification glm1
glm1.probs = predict(glm.fit1,newdata=ds,type="response") 
glm1.preds = ifelse(glm1.probs>.5,"Response","No response") 

churn_obs = ifelse(ds$churn==1,"Response","No response")
#confusion matrices for glm1 
 
cm1_is = table(predicted=glm1.preds[idx==1], actual=churn_obs[idx==1])
cm1_os = table(predicted=glm1.preds[idx==2], actual=churn_obs[idx==2]) 
confusionMatrix(cm1_is, positive = "Response")
confusionMatrix(cm1_os, positive = "Response")

#ROC and AUC for glm1
roc(ds$churn[idx==2]~glm1.probs[idx==2], plot = TRUE, print.auc = TRUE)  


#classification glm2
glm2.probs = predict(glm.fit2,newdata=ds,type="response") 
glm2.preds = ifelse(glm2.probs>.5,"Response","No response") 

#confusion matrices for glm2 
cm2_is = table(predicted=glm2.preds[idx==1], actual=churn_obs[idx==1])
cm2_os = table(predicted=glm2.preds[idx==2], actual=churn_obs[idx==2]) 
confusionMatrix(cm2_is, positive = "Response")
confusionMatrix(cm2_os, positive = "Response")

#ROC and AUC for glm2
roc(ds$churn[idx==2]~glm2.probs[idx==2], plot = TRUE, print.auc = TRUE)  

#selecting model 2: glm.fit2 as the final model
glm.log_final = glm(churn~tenure+rating+partysize+urban+menu+frequency+rating*partysize+rating*urban+
                      partysize*urban+urban*tenure, family=binomial,data=ds)
churn_prediction = predict(glm.log_final,newdata=ds,type="response") 
hist(churn_prediction)



#Task 2: monthlyaddons as outcome
#specification 3
glm.fit3 = glm(monthlyaddons~tenure+rating+partysize+urban+menu+frequency, family="gaussian",data=ds[idx==1,])
summary(glm.fit3)

#specification 4
glm.fit4 = glm(monthlyaddons~tenure+rating+partysize+urban+menu+frequency+rating*partysize+rating*urban+
                 partysize*urban+urban*tenure, family="gaussian",data=ds[idx==1,])
summary(glm.fit4)

#specification 3: MRSE
lm3_pred = predict(glm.fit3,newdata=ds)
table(is.na(lm3_pred))
hist(lm3_pred)
postResample(pred = lm3_pred[idx==1 & is.na(ds$churn)==FALSE], obs = ds$churn[idx==1 & is.na(ds$churn)==FALSE])
postResample(pred = lm3_pred[idx==2 & is.na(ds$churn)==FALSE], obs = ds$churn[idx==2 & is.na(ds$churn)==FALSE])

#specification 4: MRSE
lm4_pred = predict(glm.fit4,newdata=ds)
table(is.na(lm4_pred))
hist(lm4_pred)
postResample(pred = lm4_pred[idx==1 & is.na(ds$churn)==FALSE], obs = ds$churn[idx==1 & is.na(ds$churn)==FALSE])
postResample(pred = lm4_pred[idx==2 & is.na(ds$churn)==FALSE], obs = ds$churn[idx==2 & is.na(ds$churn)==FALSE])

#select model 3 as the final model for linear regression
glm.linear_final=glm(monthlyaddons~tenure+rating+partysize+urban+menu+frequency, family="gaussian",data=ds)
addons_prediction=predict(glm.linear_final,newdata=ds,type="response")
#since monthlyaddons cannot be negative, replacing all negative monthlyaddons with "0"
addons_prediction[addons_prediction<0]<-0
if (all(addons_prediction >= 0)){
  print("All values are non-negatives!")
}
hist(addons_prediction)
table(is.na(addons_prediction))#check missing value

#export file
export_df = cbind(ds,churn_prediction,addons_prediction)
write.csv(export_df,"blueapron_predicted.csv")
