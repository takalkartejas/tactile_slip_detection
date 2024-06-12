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
   
  1. self.sequence_of_image_array = [6,8,9,10]
  2. self.learning_rate_array = [0.00005,0.00003, 0.00004, 0.00001,0.0000008, 0.000006 ]
  3. self.reshuffle_array=[False, True]
  4. self.regularization_constant_array = [0.01, 0.05, 0.1, 0.2, 0.3]
  5. self.dense_neurons2_array = [8, 16, 32]
  6. self.vgg_layers_array= [7,11,15,19]
  7. self.sequence_of_images =  ***6,8***
  8. self.learning_rate = ***0.00003***
  9. self.reshuffle = ***Flase***
  10. self.dropout1 = 0.5
  11. self.dropout2 = 0.5
  12. self.dropout3 = 0.5
  13. self.dropout4 = 0.5
  14. self.regularization_constant = 0.001
  15. self.batch_size = 4
  16. self.dense_neurons1 = 64
  17. self.dense_neurons2 = 8
  18. self.csv_id = 0
  19. self.no_of_samples = 100
  20. self.epochs = 50
  21. self.vgg_layers = 19
  22. self.other_param='additional cnn + global average'
  23. self.no_of_nonslip_data = ***8***
  24. self.slip_instant_labels = 0.001

* self.no_of_nonslip_data = ***30***, the predictions are more distributed but lot of 0s predicted as 1s, refer deep_nn_evaluate19.ipynb
* self.no_of_nonslip_data = ***200***, 0s are predicted incorrectly which might be due to abrubtly selected threshold value 0.003,1s are better predicted refer deep_nn_evaluate20.ipynb
* Changed the classification value to 0.0005 and maximum value to 0.05, it reduced the accuracy drastically to 60%. summary22 to summary26
* before data duplication the slip datas are classified near 0.9 and non- slip are classifed near 0.4
* implemented data balancing by data duplication, the model is classiying the slip immidiately above and a bit below 0.5 deep_nn_evaluate23.ipynb

  1. self.sequence_of_images =  6
  2. self.learning_rate = ***0.00003***
  3. self.reshuffle = Flase
  4.  self.dropout1 = 0.5
  5.  self.dropout2 = 0.5
  6.  self.dropout3 = 0.5
  7.  self.dropout4 = 0.5
  8.  self.regularization_constant = 0.001
  9.  self.batch_size = 4
  10. self.dense_neurons1 = 64
  11. self.dense_neurons2 = 8
  12. self.csv_id = 0
  13. self.no_of_samples = 450
  14. self.epochs = 40
  15. self.vgg_layers = 19
  16. self.other_param='additional cnn + global average'
  17. self.no_of_nonslip_data = 200
  18. self.slip_instant_labels = 0.0001
  19. max_labels = 0.2