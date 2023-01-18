library(dplyr)
library(ggplot2)
library(data.table) 
library( corrplot )

corr = as.data.table(fread( "results/corrmat.csv" ))
corr[,Date:=NULL]
corr2 = cor( corr )

corrplot.mixed(corr2, order = 'AOE')