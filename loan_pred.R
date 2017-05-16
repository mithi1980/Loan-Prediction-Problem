### start loading train and test data
loandata<-read.csv('/home/mithilesh/Documents/MK/AV_Loan Prediction/train_u6lujuX_CVtuZ9i.csv',header=TRUE,sep = ',')
loandatatest<-read.csv('/home/mithilesh/Documents/MK/AV_Loan Prediction/test_Y3wMUE5_7gLdaTN.csv',header=TRUE,sep = ',')
### end loading train and test data
library(class)
library(rpart)
library(corrplot)
summary(loandatatest[,-c(1)])

loandata$LoanAmount[is.na(loandata$LoanAmount)]<-0
loandata$LoanAmount[loandata$LoanAmount==0]<-mean(loandata$LoanAmount) ## imputed mean
loandata$Loan_Amount_Term[is.na(loandata$Loan_Amount_Term)]<-360 ## imputed the maximum repeating categorical
loandata$Credit_History[is.na(loandata$Credit_History)]<-1 ## imputed the maximum repeating categorical

# start feature engineering
 #loandata$emi<-loandata$LoanAmount/loandata$Loan_Amount_Term
 #loandatatest$emi<-loandatatest$LoanAmount/loandatatest$Loan_Amount_Term
 loandata$loantoincome<-loandata$LoanAmount/((loandata$ApplicantIncome+loandata$CoapplicantIncome)**2)
 loandatatest$loantoincome<-loandatatest$LoanAmount/((loandatatest$ApplicantIncome+loandatatest$CoapplicantIncome)**2)
 loandata$dummy<-ifelse(loandata$Dependents==c('0','1') 
                        & loandata$Education=='Graduate' & loandata$Self_Employed=='No' 
                        & loandata$ApplicantIncome+loandata$CoapplicantIncome>6000
                        ,1,0)
 loandatatest$dummy<-ifelse(loandatatest$Dependents==c('0','1') 
                        & loandatatest$Education=='Graduate' & loandatatest$Self_Employed=='No' 
                        & loandatatest$ApplicantIncome+loandatatest$CoapplicantIncome>6000
                        ,1,0)
 
 
 #end feature engineering
table(loandatatest$dummy)
# start to check correlation among independent variables 
cor1<-cor(loandata[sapply(loandata, is.numeric)])
corrplot(cor1)
# end to check correlation among independent variables 

loan_model<-rpart(Loan_Status~.,data=loandata[,-c(1)],method = "class",control = rpart.control(cp=0.001))
# start model fitness
pred<-predict(loan_model,loandata[,-c(1)],type = "class")
table(loandata$Loan_Status,pred)
#86.3% accuracy
# end model fitness


# start predicting for test data
pred<-predict(loan_model,loandatatest[,-c(1)],type = "class")
loandatatest$Loan_Status_Predicted<-pred
table(pred)
write.csv(loandatatest[,c(1,14)],file = "/home/mithilesh/Documents/MK/AV_Loan Prediction/sample_submission.csv")
# end predicting for test data

### XGBOOST
library(xgboost)
bst <- xgboost(data = data.matrix(loandata[,-c(13)])
               ,label = data.matrix(ifelse(loandata$Loan_Status=='Y',1,0))
               , max.depth = 3, eta = 0.1
               ,nround = 19, objective = "binary:logistic",missing = NaN) 
pred <- predict(bst, data.matrix(loandatatest[,]),missing=NA)
pred <- predict(bst, data.matrix(loandata[,]),missing=NA)

table(loandata$Loan_Status,ifelse(pred>0.5,'Y','N'))
cv<-xgb.cv(data=data.matrix(loandata[,-c(13)])
           ,label = data.matrix(ifelse(loandata$Loan_Status=='Y',1,0))
           ,nrounds = 200,eta=0.1,nfold = 5,objective="binary:logistic"
           ,metrics = 'auc')
print(cv)

table(ifelse(pred>0.5,1,0))

x = glm(Loan_Status~.,data=loandata[,-c(1)],family=binomial(link="logit"))
summary(x)
###

install.packages("missForest")
library(missForest)
x<-missForest(loandata[,-1])
summary(x$ximp)
