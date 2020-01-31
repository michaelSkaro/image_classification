# fun with image classification :)

setwd("/Volumes/My Passport/LApops_Raw/images")


if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("EBImage")

library(keras)
library(EBImage)


# make a directory of all the files to learn 

# make an array of file names

pics <- list.files() # make sure to add the file extensions *** 

# make an empty list 

my_pic <- list()


# for loop to read them and save them

for(i in 1:length(pics)){my_pic[[i]] <- readImage(pics[i])}

# explore the data set by printing the data to screen

print(my_pic[[1]])


# if you want to look at the image you may dsiplay the loaded image 

display(my_pic[[1]])

hist(my_pic[[2]])
str(my_pic)

# Resize
for (i in 1:length(pics)){my_pic[[i]] <- resize(my_pic[[i]], 28,28)}

# Reshape
for (i in 1:length(pics)) {my_pic[[i]] <- array_reshape(my_pic[[i]], c(28, 28))}

# Row Bind for training
trainx <- NULL
for (i in 1:length(val1)) {trainx <- rbind(trainx, mypic[[i]])}
for (i in val1+1:val2) {trainx <- rbind(trainx, mypic[[i]])}
for (i in val2+1:val3) {trainx <- rbind(trainx, mypic[[i]])}
str(trainx)
testx <- ---(mypic[[leave out data1]], mypic[[leave out data2]],  mypic[[leave out data3]])
trainy <- c(0,0,0,0,0,1,1,1,1,1,2,2,2,2) # etc 
testy <- c(---, ---) # label the identities of the test set

# One Hot Encoding
trainLabels <- to_categorical(trainy)
testLabels <- to_categorical(testy)

# Model
model <- keras_model_sequential()
model %>%
  layer_dense(units = 256, activation = "relu", input_shape = c(put in the structure value for the str(trainx))) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'softmax') # three units for the classes
summary(model)

# Compile
model %>%
  compile(loss = not sure which loss function we should use, we will have to step into the code and find out ***,
          optimizer = optimizer_rmsprop(),
          metrics = c('accuracy'))

# Fit Model
history <- model %>%
  fit(trainx,
      trainLabels,
      epochs = 50,
      batch_size = 32,
      validation_split = 0.2)

# Evaluation & Prediction - train data
model %>% evaluate(---, ---)
pred <- model %>% predict_classes(trainx)
table(Predicted = pred, Actual = trainy)
prob <- model %>% predict_proba(trainx)
cbind(prob, Prected = pred, Actual= trainy)
