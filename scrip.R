rm(list=ls())
library("mlbench")
library('corrplot')
library('clusterSim')
library('caret')
data(Sonar)

X <- data.matrix(Sonar[,1:60])
X <- data.Normalization(X,type="n1")
Y <- 1*(Sonar[,61] == "M")

#--------------------------------------- PCA

#Transformação
xm <- colMeans(X)
X_xm <- X-matrix(xm,nrow=nrow(X),ncol=ncol(X),byrow=T)
X.cov <- cov(X_xm)
eigS <- eigen(X.cov)
projX <- X_xm %*% eigS[[2]]

#Plot da explicação da variância
plot(eigS[[1]]/sum(eigS[[1]])*100,type='b',col='red',
     xlab='PCA',ylab='Var_explained (%)',main='Variância explicada x Componente')

#Variância acumulada
cdf<-c()
aux<-0
for(i in 1:length(eigS[[1]]))
{
  cdf[i]<-aux+eigS[[1]][i]/sum(eigS[[1]])*100
  aux<-aux+eigS[[1]][i]/sum(eigS[[1]])*100
}
plot(cdf,type='b',ylim=c(0,100),xlab='PCA',ylab='Accumulated (%)',
     main='Accumulated Var_explained')

#--------------------------------------- Plot da matriz de correlação
S <- cov(X)
corrplot(S, method = "circle", type = "lower",order = "FPC")

#--------------------------------------- Histogram
for(i in 1:ncol(X))
{
  lb <- min(X[,i])
  ub <- max(X[,i])
  hist(X[Y==1,i],breaks=20,col='red',xlim=c(ub,lb),
       xlab='',ylab='',main='')
  par(new=T)
  hist(X[Y==0,i],breaks=20,col='blue',xlim=c(ub,lb),
       xlab=i,main=i)
}

#Centros para cada variável
c1 <- colMeans(X[Y==1,])
c2 <- colMeans(X[Y==0,])

s1 <- apply(X[Y==1,],2,sd)
s2 <- apply(X[Y==0,],2,sd)

#Distância entre os centros
sep <- abs(c1 - c2)/(s1+s2)
plot(sep,type='b',xlab='Índice de X',
     ylab = 'Dist_centers', main = 'Medida de separação entre classes')

#Ordenando o vetor de separabilidade e definindo as características
sep.ordered <- sort(sep, decreasing = TRUE)
Ncarac <- 5
print(sep.ordered[1:Ncarac])

#Plotando características escolhidas
#V11
lb <- min(X[,11])
ub <- max(X[,11])

hist(X[Y==1,11],breaks=40,col='green',xlim=c(lb,ub),ylim=c(0,12),
     xlab='',ylab='',main='')
par(new=T)
hist(X[Y==0,11],breaks=40,col='blue',xlim=c(lb,ub),ylim=c(0,12),
     xlab='V11',main='Histograma V11')

#V12
lb <- min(X[,12])
ub <- max(X[,12])

hist(X[Y==1,12],breaks=40,col='green',xlim=c(lb,ub),ylim=c(0,12),
     xlab='',ylab='',main='')
par(new=T)
hist(X[Y==0,12],breaks=40,col='blue',xlim=c(lb,ub),ylim=c(0,12),
     xlab='V12',main='Histograma V12')

#V45
lb <- min(X[,45])
ub <- max(X[,45])

hist(X[Y==1,45],breaks=40,col='green',xlim=c(lb,ub),ylim=c(0,12),
     xlab='',ylab='',main='')
par(new=T)
hist(X[Y==0,45],breaks=40,col='blue',xlim=c(lb,ub),ylim=c(0,12),
     xlab='V45',main='Histograma V45')

#----------------------------------------Treino, teste e classificação
X <- Sonar[,c(11,12,45,49,10)]
Y <- Sonar[,61]
classData <- cbind(X,Y)

#Train and test w/ caret
splitIdx <- createDataPartition(Y,times=1,p=0.7,list=FALSE)
trainData <- classData[splitIdx,]
testData <- classData[-splitIdx,]

ctrl <- trainControl(method = "cv", savePred=T, classProb=T)
mod <- train(Y ~., data=trainData, 
             method = "svmLinear2", trControl = ctrl)

yhat <- predict(mod,testData)

ACC.ind <- sum(diag(table(yhat,testData$Y)))/length(yhat)

#Utilizando o PCA
yt <- Sonar[,61]
xt <- projX[,1:5]
classData<-data.frame(xt,yt)

splitIdx <- createDataPartition(yt,times=1,p=0.7,list=FALSE)
trainData <- classData[splitIdx,]
testData <- classData[-splitIdx,]

ctrl <- trainControl(method = "cv", savePred=T, classProb=T)
mod2 <- train(yt ~., data=trainData, 
             method = "svmLinear2", trControl = ctrl)

yhat <- predict(mod2,testData)

ACC.PCA <- sum(diag(table(yhat,testData$yt)))/length(yhat)
