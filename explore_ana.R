## this is explory analysis of the image data set
###
rm(list=ls())
options(warn=1)
options(stringsAsFactors=FALSE)
options(digits=15)
require(stringr)
require(magrittr)
require(R.matlab)
require(ggplot2)
require(reshape2)
dirtab="/Users/mikeaalv/Dropbox (Edison_Lab@UGA)/Projects/Bioinformatics_modeling/emi_nnt_image/data/Classifications.csv"
tab=read.table(dirtab,header=TRUE,sep=",")
dim(tab)
table(tab[,2])
