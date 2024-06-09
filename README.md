# tactile_slip_detection
## Tuning

#### Problems
1. Very low training accuracy 60%  (solved)
2. Very low validation accuracy 55% when training accuracy is 90%
3. 
#### Ideas

1. Change to binary crossentropy - benificial for binary classifications (Implemented) (improved training accuracy)
2. Play around with learning rate (effecting training accuracy) (tuning para)
3. remove some convolution layers from VGG16 (make a tuning parameter out of it)
4. shuffle in load data
5. adjust the labeling according to sudden change in object position
6. add dots to the tactile images
7. checkout what the depth data is
8. add a convolution layer and global average pooling instead of a dense layer for feature extraction


#### Parameters
1. sequence_of_image 
2. learning_rate 
3. reshuffle 
4. dropout1 
5. dropout2 
6. dropout3 
7. dropout4 
8. regularization_constant 
9. batch_size 
10. dense_neurons1 
11. dense_neurons2
12. remove_vgg_layers

#### Summary of tuning

1. sequence_of_image = 5
2. learning_rate = 0.001
3. reshuffle = 0
4. dropout1 = 0.5
5. dropout2 = 0.5
6. dropout3 = 0.5
7. dropout4 = 0.5
8. regularization_constant = 0.01 
9. batch_size = 64
10. dense_neurons1 = 64
11. dense_neurons2 = 8
12. remove_vgg_layers = 0

Train_acc = 60%, val_acc = 55%

* Binary_crossentropy and learning rate-
   By changing the loss function from mean squred error to binary cross entropy
   and decreasing the learning rate from 0.0001 to 0.00001 increased the training
   accuracy from 60% to 90 %

1. sequence_of_image = 5
2. learning_rate = ***0.0001***
3. reshuffle = 0
4. dropout1 = 0.5
5. dropout2 = 0.5
6. dropout3 = 0.5
7. dropout4 = 0.5
8. regularization_constant = 0.01 
9. batch_size = 64
10. dense_neurons1 = 64
11. dense_neurons2 = 8
12. remove_vgg_layers = 0

Train_acc = 90%, val_acc = 55%

* Reducing batch size and reshuffling-
  1. Main objective was to reduce overfitting but no signinficant change observed.
  2. But these changes are observable
  3. Reducing batch size decreases memory consumption but increases training time
  4. Reshuffling increases training time **significantl**

1. sequence_of_image = 5
2. learning_rate = 0.0001
3. reshuffle = ***1***
4. dropout1 = 0.5
5. dropout2 = 0.5
6. dropout3 = 0.5
7. dropout4 = 0.5
8. regularization_constant = 0.01 
9. batch_size = ***8***
10. dense_neurons1 = 64
11. dense_neurons2 = 8
12. remove_vgg_layers = 0

Train_acc = 90%, val_acc = 55%

* Removing no slip data- summary 12, 13 
  1. Tried to keep only few non-slip data
  2. observed that the output is 0.83 someting for all the datas
  3. This is wrong output
   
     self.sequence_of_image_array = [6,8,9,10]
     self.learning_rate_array = [0.00005,0.00003, 0.00004, 0.00001,0.0000008, 0.000006 ]
     self.reshuffle_array=[False, True]
     self.regularization_constant_array = [0.01, 0.05, 0.1, 0.2, 0.3]
     self.dense_neurons2_array = [8, 16, 32]
     self.vgg_layers_array= [7,11,15,19]
     self.sequence_of_images =  ***6,8***
     self.learning_rate = ***0.00003***
     self.reshuffle = ***Flase***
     self.dropout1 = 0.5
     self.dropout2 = 0.5
     self.dropout3 = 0.5
     self.dropout4 = 0.5
     self.regularization_constant = 0.001
     self.batch_size = 4
     self.dense_neurons1 = 64
     self.dense_neurons2 = 8
     self.csv_id = 0
     self.no_of_samples = 100
     self.epochs = 50
     self.vgg_layers = 19
     self.other_param='additional cnn + global average'
     self.no_of_nonslip_data = ***8***
     self.slip_instant_labels = 0.001