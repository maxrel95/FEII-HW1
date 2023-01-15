library(dplyr)
library(ggplot2)
library(data.table) 
library( corrplot )

corr = as.data.table(fread( "/Users/maxime/Documents/Université/HEC/PhD/6.1/FE I/HW4/corrmat.csv" ))
corr[,Date:=NULL]
corr2 = cor( corr )

corrplot.mixed(corr2, order = 'AOE')