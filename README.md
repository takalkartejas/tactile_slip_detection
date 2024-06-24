# tactile_slip_detection
## Tuning

#### Problems
1. Very low training accuracy 60%  (solved)
2. Very low validation accuracy 55% when training accuracy is 90%
3. 
#### Ideas

1. Change to binary crossentropy - benificial for binary classifications (Implemented) (improved training accuracy)
2. Play around with learning rate (effecting training accuracy) (tuning para)
3. remove some convolution layers from VGG16 (tried it, just puts a cap on accuracy araound 52%)
4. shuffle in load data (shuffled the data before training)
5. adjust the labeling according to sudden change in object position ( there is no sudden change in object position, the object starts slipping with a really small velocity)
6. add dots to the tactile images( it might take time, better to focus on using depth images)
7. checkout what the depth data is ( need to confirm if depth data is readily available or I need to create a code to find it in real sensor)
8. add a convolution layer and global average pooling instead of a dense layer for feature extraction (theoretically should reduce the overfitting and memory consumption. already implemented)
9. try reduing the after slip images to avoid overfitting ( increased the amount of preslip images)
10. remove some the data at the tranistion between 0 and 1  

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
<details> <summary>
Iteration1 - using small data and deep layer for feature detection
   
</summary>
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
</details>
<details> <summary>
Iteration2 - Binary_crossentropy and learning rate-
</summary>
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

</details>
<details> <summary>
Iteration3 - Reducing batch size and reshuffling-
</summary>


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

</details>
<details> <summary>
Iteration4 - Removing no slip data- summary 12, 13 
</summary>

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
</details>
<details> <summary>
Iteration5 - data balancing 
</summary>
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

</details>
<details> <summary>
Iteration6 - data size, learning rate, early stopping, summary - 67,68,69,70
</summary>
1. It was observed that using lot of samples 4559 increased the validation accuracy
2. early stopping - stops the training if validation accuracy does not increase for 2 iterations
3. the data with more than 5 mm slip is deleted, the tuning variable is called max_label
5.  sequence of images = 8
6. The main tuning variable was learning rate- 
    * 0.005 - summary 67- accuracy capped during first iteration and was early stopped
    * 0.0005 - summary 67-val accuracy capped at 0.51, early stopped
    * 0.00005 - summary 67- val accuracy increased to 0.574 early stopped after 4 epochs
    * 0.00003 - summary 67-val accuracy increased to 0.573 early stopped after 4 epochs
7. The f1 score is matching the accuracy

1. self.sequence_of_images =  8
2. self.learning_rate = ***0.005,0.0005,0.00005,0.00003***
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
13. self.no_of_samples = 4559
14. self.epochs = 40
15. self.vgg_layers = 19
16. self.other_param='additional cnn + global average'
17. self.no_of_nonslip_data = 200
18. self.slip_instant_labels = 0.0001
19. max_labels = 0.2

</details>
<details> <summary>
Iteration7 - slip_instant_labels, max_labels
</summary>
no of samples = 2279
1. summary49, slip_instant_labels=0.00005, max_labels=0.2, val_acc=0.57
2. summary50, slip_instant_labels=0.0001, max_labels=0.2, val_acc=0.57
3. summary51, slip_instant_labels=0.0001, max_labels=0.005, val_acc=0.541
4. summary52, slip_instant_labels=0.0005, max_labels=0.005, val_acc=0.545

1. self.sequence_of_images =  8
2. self.learning_rate = 0.00003
3. self.reshuffle = Flase
4.  all_dropouts = 0.5
5.  self.regularization_constant = 0.001
6.  self.batch_size = 4
7. self.dense_neurons1 = 64
8. self.dense_neurons2 = 8
9. self.no_of_samples = *** 2279 ***
10. self.epochs = 40
11. self.vgg_layers = 19
12. self.other_param='additional cnn + global average'
13. self.slip_instant_labels = *** 0.00005,0.0001,0.0005***
14. max_labels = *** 0.2, 0.005 ***

</details>
<details> <summary>
Iteration8 - increase dataset
</summary>
no of samples = 4559
1. summary53, slip_instant_labels=0.0001, max_labels=0.005, val_acc=0.5581
# it was observed that the accuracy is not increasing easily with low max_labels but val_accuracy is keeping up with accuracy for more iterations

1. self.sequence_of_images =  8
2. self.learning_rate = 0.00003
3. self.reshuffle = Flase
4.  all_dropouts = 0.5
5.  self.regularization_constant = 0.001
6.  self.batch_size = 4
7. self.dense_neurons1 = 64
8. self.dense_neurons2 = 8
9. self.no_of_samples = *** 4559 ***
10. self.epochs = 40
11. self.vgg_layers = 19
12. self.other_param='additional cnn + global average'
13. self.slip_instant_labels = 0.0001
14. max_labels = 0.005

</details>
<details> <summary>
Iteration9 - stride
</summary>
The stride 5 and stride 3 did not deliver good results, so defaulted to 1

</details>
<details> <summary>
Iteration10 - reformed data generator, reduced the applied force, correted data balancing error
</summary>
1. summary65, slip_instant_labels=0.0002, max_labels=0.02, val_acc=0.6146, no_of_train_samples = 3720
</details>
