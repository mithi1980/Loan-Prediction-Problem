Add_Count_Feature<-function(train_df,group_by,aggregate_col,New_Feature_Name,aggregate_type,filter_clause)
{
  sqlstmt<-paste("select",group_by,",",aggregate_type,"(",aggregate_col,")",New_Feature_Name,"from",train_df,filter_clause,"group by",group_by,sep = " ")  
  tempdf<-sqldf(sqlstmt)
  eval(parse(text=paste(train_df,'$',New_Feature_Name,'<-NULL',sep="")))
  loandata<-merge(loandata,tempdf,by=group_by,all.x = T)
  eval(parse(text="loandata[,c(New_Feature_Name)]<-as.numeric(loandata[,c(New_Feature_Name)])"))
  return(loandata)
}

### start loading train and test data
loandatatrain<-read.csv('/home/mithilesh/Documents/MK/Data Science/Analytics Vidhya/AV_Loan Prediction/train_u6lujuX_CVtuZ9i.csv',header=TRUE,sep = ',')
loandatatest<-read.csv('/home/mithilesh/Documents/MK/Data Science/Analytics Vidhya/AV_Loan Prediction/test_Y3wMUE5_7gLdaTN.csv',header=TRUE,sep = ',')
### end loading train and test data
library(class)
library(rpart)
library(corrplot)
library(ROSE)
library(sqldf)
loandatatest$Loan_Status<-as.factor('C')
loandata<-rbind(loandatatrain,loandatatest)
#loandata<-ifelse()
summary(loandata)
str(loandata)
loandata$LoanAmount[is.na(loandata$LoanAmount)]<-mean(loandata$LoanAmount,na.rm = T)
#loandata$LoanAmount[loandata$LoanAmount==0]<-mean(loandata$LoanAmount) ## imputed mean
loandata$Loan_Amount_Term[is.na(loandata$Loan_Amount_Term)]<-360 ## imputed the maximum repeating categorical
loandata$Credit_History[is.na(loandata$Credit_History)]<-1 ## imputed the maximum repeating categorical

# start feature engineering
#loandata$emi<-loandata$LoanAmount/loandata$Loan_Amount_Term
#loandatatest$emi<-loandatatest$LoanAmount/loandatatest$Loan_Amount_Term
loandata$loantoincome<-loandata$LoanAmount/((loandata$ApplicantIncome+loandata$CoapplicantIncome)**2)
#loandatatest$loantoincome<-loandatatest$LoanAmount/((loandatatest$ApplicantIncome+loandatatest$CoapplicantIncome)**2)
loandata$dummy<-ifelse(loandata$Dependents==c('0','1') 
                       & loandata$Education=='Graduate' & loandata$Self_Employed=='No' 
                       & loandata$ApplicantIncome+loandata$CoapplicantIncome>6500
                       ,1,0)
loandata<-Add_Count_Feature('loandata','Credit_History','Credit_History','Credit_History_Count','count','')
loandata<-Add_Count_Feature('loandata','Loan_Amount_Term','Loan_Amount_Term','Loan_Amount_Term_Count','count','')

loandata<-loandata[order(loandata$Loan_ID),]
loandata$Loan_Seq<-seq(1,by=1,to=length(loandata$Loan_ID))
# loandatatest$dummy<-ifelse(loandatatest$Dependents==c('0','1') 
#                            & loandatatest$Education=='Graduate' & loandatatest$Self_Employed=='No' 
#                            & loandatatest$ApplicantIncome+loandatatest$CoapplicantIncome>6000
#                            ,1,0)


#end feature engineering

#start split back training and test data
loandatatrain<-loandata[loandata$Loan_Status!='C',]
loandatatest<-loandata[loandata$Loan_Status=='C',]
loandatatest$Loan_Status<- NULL

#end split back training and test data


# start to check correlation among independent variables 
cor1<-cor(loandata[sapply(loandata, is.numeric)])
corrplot(cor1)
# end to check correlation among independent variables 
### start XGBOOST
set.seed(123)
library(xgboost)
cv<-xgb.cv(data=data.matrix(loandatatrain[,-c(3,13,14)])
           ,label = data.matrix(ifelse(loandatatrain$Loan_Status=='Y',1,0))
           ,nrounds = 400,eta=0.05,nfold = 10,objective = "binary:logistic",subsample=0.75,colsample_bytree=0.80,min_child_weight=1,eval_metric="auc")


bst <- xgboost(data = data.matrix(loandatatrain[,-c(3,13,14)])
               ,label = data.matrix(ifelse(loandatatrain$Loan_Status=='Y',1,0))
               , max.depth = 6, eta = 0.05
               ,nround = 400, objective = "binary:logistic",missing = NaN
               ,subsample=0.75,colsample_bytree=0.80,min_child_weight=1,eval_metric="auc") 
xgb.importance(setdiff(colnames(loandatatrain),c("Loan_ID","Loan_Status")),model = bst)
### end XGBOOST
### fitness check
pred <- predict(bst, data.matrix(loandata[,-c(1,13)]),missing=NA)
table(loandata$Loan_Status,ifelse(pred>0.5,'Y','N'))
roc.curve(loandata$Loan_Status, pred)
# start predicting for test data
pred <- predict(bst, data.matrix(loandatatest[,-c(3)]),missing=NA)
loandatatest$Loan_Status<-(ifelse(pred>0.5,'Y','N'))
table(loandatatest$Loan_Status)
write.csv(loandatatest[,c(3,18)],file = "/home/mithilesh/Documents/MK/Data Science/Analytics Vidhya/AV_Loan Prediction/xgb/sample_submission.csv")
# end predicting for test data


