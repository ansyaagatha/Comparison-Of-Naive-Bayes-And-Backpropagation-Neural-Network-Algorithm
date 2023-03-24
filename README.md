-- Comparison Of Naïve Bayes And Backpropagation Neural Network Algorithm In Autism Spectrum Disorder Based On Particle Swarm Optimization --

Description : 
- This research aims to analyze the classification model from data mining for autism classification. Feature selection will be applied to the classification model of this research which is expected to improve model performance

System Testing :
- The system can read files with .xlsx extension.
- Python programming language generates Confusion Matrix and ROC Curve AUC in .jpg extension.
- The system displays the data before preprocessing, the amount of missing value data, the amount of data after preprocessing, the amount of training data, the amount of data testing, the number of selected data attributes, the confusion matrix and the ROC curve auc.
- The selection of the dataset is divided into 3 categories, namely adolescent, adult and child.
- The algorithm on the web page is an algorithm that has been combined with the selection feature particle swarm optimization algorithm.

Algorithm Performance Results
![Capture](https://user-images.githubusercontent.com/77670162/227423379-f8eb7c8e-bc3b-45a2-ad3e-bfebb49e4a14.PNG)

Conclusion :
- The use of feature selection particle swarm optimization in the naïve Bayes algorithm and neural network has succeeded in increasing accuracy
a. Accuracy increased by 11% in naïve Bayes algorithm with Adolescent dataset, from 86% to 97% in 50 iterations.
b. Accuracy increases by 2% in the Naive Bayes algorithm with the Adult dataset, from 96% to 98% in 50 iterations.
c. Accuracy decreased by 6% in naïve Bayes algorithm with Child dataset, from 95% to 89% in iteration 1.
d. Accuracy increased by 7% on the neural network algorithm with the Adolescent dataset, from 83% to 90% at 30 iterations.
e. Accuracy increased by 15% on the neural network algorithm with the Adult dataset, from 83% to 98% at 50 iterations.
f. Accuracy increased by 18% on the neural network algorithm with the Child dataset, from 68% to 86% in iteration 30.

- Comparison of accuracy between the use of feature selection particle swarm optimization on naïve Bayes algorithms and neural networks is as follows.
a. The highest accuracy comparison between feature selection particle swarm optimization on naïve Bayes and neural networks in the Adolescent dataset is naïve bayes 97% with an AUC value of 0.981 which means that the use of this model is successful because the AUC value is close to 1.
b. Comparison of the highest accuracy between feature selection particle swarms
c. optimization on naïve Bayes and neural networks in the Adult dataset is a 98% neural network with an AUC value of 1,000 which means that the use of this model is successful because the AUC value is close to 1.
d. The highest accuracy comparison between feature selection particle swarm optimization on naïve Bayes and neural networks in the Child dataset is naïve bayes 89% with an AUC value of 0.964 which means that the use of this model is successful because the AUC value is close to 1

- The average use of the Naïve Bayes model is superior to the neural network.
- The more iterations carried out, particle swarm optimization does not guarantee good accuracy

1. The main page is a page that appears early when the website is run. This page has functions to go to the about page and to the next page to start running the Naïve Bayes or Neural Network algorithm
![image](https://user-images.githubusercontent.com/77670162/227422266-6d1dd361-bacc-4fd9-9268-a6e8f0feb820.png)

2. The algorithm selection page is a page that functions to choose between the Naïve Bayes algorithm and the Neural Network to be used. The algorithm selection page will appear after selecting the dataset on the previous page
![image](https://user-images.githubusercontent.com/77670162/227422451-cc315b11-54d6-4b66-b0f4-ec9b1e163c1b.png)

3. The results page is a page that functions to display the results of recall, accuracy, precision and f-measure from training and testing on the dataset and algorithm that has been selected to be used on the previous page
![image](https://user-images.githubusercontent.com/77670162/227422507-54137634-4f27-42ac-8323-b6b244cf4904.png)
![image](https://user-images.githubusercontent.com/77670162/227422517-382848d0-405a-4083-8c6a-d589ee585e5c.png)
![image](https://user-images.githubusercontent.com/77670162/227422525-63c43b0a-5b86-440a-88f7-d85ce92fca7a.png)

