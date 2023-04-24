setwd("C:\\Users\\Killian\\Desktop\\FYP-Multistage-Throughput-Predictor")
data <- read.csv("C:\\Users\\Killian\\Desktop\\FYP-Multistage-Throughput-Predictor\\Datasets\\unaveraged_processed_network_data.csv")
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
image_dir = "C:\\Users\\Killian\\Desktop\\FYP-Multistage-Throughput-Predictor\\Own Papers\\Undergraduate Paper\\Images\\"
names = colnames(data[which(data$session==1), -c(which(colnames(data)=="session"))])
names
max_lag = 60
dl_bitrate_acf = matrix(nrow=135, ncol=max_lag+1)
for (i in 0:134) {
  trace <- as.matrix(data[which(data$session==i), -c(which(colnames(data)=="session"))])
  dl_acf <- acf(trace[,1], plot=F, lag.max = max_lag)
   dl_bitrate_acf[i+1,]<- dl_acf$acf
}

average_acf <- colMeans(dl_bitrate_acf)
std_acf <- apply(dl_bitrate_acf, 2, sd)
std_acf
average_acf


png(paste0(image_dir,"Average_DL_bitrate","_ACF.png"),
    width = 20, height = 15, units="cm", res=300)
plot(average_acf, main=paste("Average ACF of Download Throughput over All Traces"),
     xlab="lag", ylab="ACF", pch=19, font.lab = 2, cex.lab=1.2, cex.axis=1.2)
upper <- average_acf + std_acf
lower <- average_acf - std_acf
lower

shade_x <- c(seq(1,max_lag+1), rev(seq(1,max_lag+1)))
shade_y <- c(upper, rev(lower))
polygon(shade_x, shade_y, col = rgb(1,0,0,0.2), border = NA)
# Close the PNG graphics device
x <- which(average_acf<0.4)[1]
segments(-4,0.4,x,0.4, col="blue", lwd=2)
segments(x,0,x,0.4, col="blue", lwd=2)
abline(h = 0.6, lty = 2, col = "grey")
abline(h = 0.4, lty = 2, col = "grey")
abline(h = 0.8, lty = 2, col = "grey")
axis(side = 1, at = x, labels = x,
     col.axis = "blue", col.ticks = "blue", cex.axis=1.2)
par(font.axis = 2)
dev.off()

dev.new()

max_lag = 30
cross_cors = array(dim=c(ncol(data)-2,(2*max_lag)+1, 135))
for (i in 0:134) {
  trace <- as.matrix(data[which(data$session==i), -c(which(colnames(data)=="session"))])
  for (feature in 2:ncol(trace)) {
    ccor = ccf(trace[,1], trace[,feature], lag.max = max_lag, plot=F)
    cross_cors[feature-1,,i+1] = ccor$acf
  }
}
colnames(data)
snr <- rowMeans(cross_cors[6,,], na.rm=T)
plot(x=seq(-max_lag,max_lag),y=snr, xlab="lag")


library(corrplot)
correlation_matrix = array(dim=c(20,20,135))
for (i in 0:134) {
  trace <- as.matrix(data[which(data$session==i), -c(which(colnames(data)=="session"))])
  trace[,c(4:8)] <- exp(abs(trace[,c(4:8)]))
  colnames(trace)
  trace_cor <- cor(trace)
  trace_cor[is.na(trace_cor)] = 0
  correlation_matrix[,,i+1] = trace_cor
}
colnames(data)
cor_2d <- matrix(correlation_matrix, nrow=20*20, ncol=135)

mean_cor_2d <- rowMeans(cor_2d)
average_cor_matrix = matrix(mean_cor_2d, nrow=20, ncol=20)
colnames(average_cor_matrix) = colnames(trace)
rownames(average_cor_matrix) = colnames(trace)
corrplot(average_cor_matrix)
average_cor_matrix[,2]
