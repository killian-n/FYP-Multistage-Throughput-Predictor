setwd("C:\\Users\\knola\\Desktop\\FYP-Multistage-Throughput-Predictor")
data <- read.csv("C:\\Users\\knola\\Desktop\\FYP-Multistage-Throughput-Predictor\\Datasets\\processed_network_data.csv")
library(caret)
library(MASS)

data <- data[,-c(which(colnames(data)=="Timestamp"))]

# Example
zero <- as.matrix(data[which(data$session==0),-c(which(colnames(data)=="session"))])
zero <- scale(zero, center=T)
zero_cov <- cov(zero)
zero_cov[is.na(zero_cov)] = 0
zero_cov
zero_pca <- prcomp(zero_cov, center=F, scale=F)
zero_pca$rotation
zero_importance <- apply(abs(zero_pca$rotation), 2, which.max)
most_important_features <- colnames(zero[,(apply(abs(zero_pca$rotation), 2,
                                                 which.max))[c(0:10)]])
most_important_features


# For all Traces
sessions <-data[,which(colnames(data)=="session")]
data <- as.data.frame(scale(data, center=T))
data[,which(colnames(data)=="session")] <- sessions

most_important_features = c()
required_pcs <- numeric(135)
for (i in seq(0,134)){
  trace <- as.matrix(data[which(data$session==i), -c(which(colnames(data)=="session"))])
  dim(trace)
  trace_cov <- cov(trace)
  trace_cov[is.na(trace_cov)] = 0
  trace_pca <- prcomp(trace_cov, center=F, scale=F)
  n_pcs <- cumsum(trace_pca$sdev)/sum(trace_pca$sdev)
  required_pcs[i+1] <- min(which(n_pcs>0.99))
  column_indices <- (apply(abs(trace_pca$rotation), 2, which.max))
  trace_features <- colnames(trace[,c(column_indices[1:3])])
  most_important_features = c(most_important_features, trace_features)
}

rpcs_table <- table(required_pcs)
mp_table <- table(most_important_features)

sort(mp_table, decreasing=T)
rpcs_table



# Auto correlation
image_dir = "C:\\Users\\knola\\Desktop\\FYP-Multistage-Throughput-Predictor\\Datasets\\Images\\"
names = colnames(data[which(data$session==1), -c(which(colnames(data)=="session"))])
max_lag = 70
dl_bitrate_acf = matrix(nrow=135, ncol=max_lag+1)
for (i in 0:134) {
  trace <- as.matrix(data[which(data$session==i), -c(which(colnames(data)=="session"))])
  dl_acf <- acf(trace[,1], plot=F, lag.max = max_lag)
   dl_bitrate_acf[i+1,]<- dl_acf$acf
}

average_acf <- colMeans(dl_bitrate_acf)
average_acf
png(paste0(image_dir,"Average_RSRQ","_ACF.png"), width = 800, height = 600)
plot(average_acf, main=paste("Average RSRQ ACF over all Traces"), xlab="lag", ylab="ACF")
# Close the PNG graphics device
dev.off()


