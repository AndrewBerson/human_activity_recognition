# Human Activity Recognition using Smartphone Sensors

**Andrew Berson and Tayden Li**

Stanford University - Stats 315B - Modern Applied Statistics: Data Mining

### 1. Abstract

Human activity recognition (HAR) involves using sensors, which are often either embedded in wearable devices or
held by individuals, to classify activities they are performing. In this project, we compare the performance of 
multiple classification techniques used to classify what activity individuals are performing based on measurements 
taken by accelerometers and gyroscopes contained within Samsung smartphones.

### 2. Introduction

Human activity recognition (HAR) aims to utilize sensors that track our motion to predict our activity. 
HAR has many use cases in healthcare and smart homes. For example, with HAR, people can track their daily 
activities and understand their physical health better. Being able to predict activity from sensors worn 
by subjects can also help healthcare workers or family members to react if an elderly person falls 
unexpectedly. The prediction can become even more accurate and effective if used in conjunction with internet 
of things (IoT). For instance, if a smartphone classifies a subject as laying, yet the sensors contained in 
their bed and couch do not sense them as laying, their chances of having fallen are significantly increased.

Smartphones and smartwatches have become tremendously popular. Most of these devices are equipped with sensors 
such as gyroscopes and accelerometers, which record users' triaxial angular speed and acceleration, respectively. 
Given the wide use of smart gadgets, predicting human activity with sensors on these devices may have a large 
impact on many individuals' lives. In this paper, we use statistical learning approaches to predict human activity 
from data recorded by gyroscopes and accelerometers worn by subjects. We then compare the results from different 
models and conduct error analysis.

### 3. Previous work

Human activity recognition fits into the larger family of classification problems. This specific data set has been 
evaluated by numerous data scientists who have employed a variety of methods to try to solve this issue. Anguita
et al. (2012) achieved 89% accuracy using SVM as the backbone for their classification \[1\]. Roobini and 
Naomi (2019) achieved almost 94% accuracy with RNN-LSTM \[2\]. Additionally, there is a Kaggle competition 
based around this data set, and it appears that the leading contestant is able to achieve almost 97% accuracy 
using a linear support vector classifier.

### 4. Data

#### 4.1 Data collection 

The data set used in this paper was generated at the University of Genoa by Anguita et al. (2013) \[3\]. 
In the experiment, thirty volunteers aged 19-48 wore smartphones (Samsung Galaxy SII) on their waists. They 
then performed six activities: walking, walking upstairs, walking downstairs, sitting, standing, and laying. 
The embedded accelerometer and gyroscope recorded subjects' triaxial linear acceleration and angular speed at 
a rate of 50Hz. 70% of the subjects were randomly selected to be in the training set and the other 30% were 
selected to be in the test set.

#### 4.2 Data preprocessing

The sensor signals were noise-filtered and sampled in fixed-width sliding windows of 2.56 seconds with 50% 
overlap. Gravitational and body motion acceleration were separated with a Butterworth low-pass filter. Features 
were calculated from each window. The final data set contains 561 features and 10,299 observations across 30 subjects.

#### 4.3 Dimensionality reduction for visualization

Principal Component Analysis (PCA) is a dimensionality reduction method. In PCA, principal components are 
constructed such that the first component accounts for the largest possible variance. Subsequent principal 
components are orthogonal to previous principal components, and the principal components are ordered such 
that earlier principal components account for more of the variance in the features than latter principle 
components. PCA works by first standardizing all features to a comparable scale. The covariance matrix as well 
as its eigenvectors and eigenvalues are then calculated. After sorting the eigenvectors by the magnitude of 
their eigenvalues, the principal components can be ranked by the amount of variance explained.

The figure below shows a clear separation of active (walking, walking upstairs, and walking downstairs) and 
sedentary (sitting, standing, laying) activities along the first principal component.

![PCA-fig](figures/pca_2comp_hist.png)
*2-d representation of data using PCA*


### 5. Methods

The methods summarized below were tuned using 5-fold Cross Validation (CV). The folds for CV were established 
such that none of the test subjects appear in multiple folds. In this way, we ensure that the CV results most 
accurately estimate the accuracy that would be achieved on a never-before-seen subject.

#### 5.1 Multinomial logistic regression (MLR) 

**Method** MLR generalizes logistic regression to multi-class classification tasks.

**Experiments** We performed two sets of experiments using MLR. In the first experiment, we used MLR after first 
reducing the dimensionality of the data set using PCA. In this experiment, the two parameters we tuned were the 
extent of dimensionality reduction as well as the L2 regularization parameter in MLR. We found that reducing the 
dimensionality of our data with PCA prior to fitting the model did not improve classification accuracy.

In the second experiment, we did not perform any dimensionality reduction, and instead only applied L2 
regularization. 

#### 5.1 Linear discriminant analysis (LDA)

**Method** LDA assumes that each class is drawn from a multivariate Gaussian distribution, where the means of each 
feature are specific to the class, but the covariance matrix is shared across all classes. Because the covariance 
matrix is shared across classes, the decision boundary is linear.

**Experiments** When forming the LDA model, shrinkage can be applied to the estimation of the covariance matrix. 
The shrinkage parameter scales the estimate of the covariance between different features. If the shrinkage parameter 
is set equal to 1, the covariance matrix will be diagonal and all features will be assumed to be uncorrelated. 
In contrast, if the shrinkage parameter is 0, the covariance matrix will equal the empirically calculated 
covariance matrix. Practically speaking, shrinkage is helpful when there are a limited number of observations. 
However, given that the number of observations in the data set is much greater than the number of features (n >> p), 
shrinkage did not improve model accuracy. Additionally, experiments were also performed where the LDA model was 
fit after reducing the dimensionality of the data set using PCA. However, similar to MLR, LDA performs worse 
when PCA is applied to reduce the data's dimensionality.

#### 5.2 Support vector machine (SVM)

**Method** SVM aims to find hyperplanes that can best divide data by their labels. A kernel is often applied to 
transform the data. Here, we used the RBF kernel. We experimented with various regularization strengths (*C*). We 
achieved a 94.29% 5-fold CV accuracy with *C* equalling 166.8.

#### 5.3 Random Forest

**Method** Random forests ensemble multiple decisions trees trained in parallel with bagging. Bagging allows 
individual trees to be trained on subsets of training data that are randomly sampled with replacement. For any given 
tree, when deciding which feature to split on, we pick the one that decreases Gini impurity the most. 

Bagging, along with the fact that at each split only a random subset of features are considered, helps reduce the 
model variance. This tends to make random forests significantly more accurate than decision trees.

**Experiments** We tuned five different hyperparameters and obtained the highest 5-fold CV accuracy 92.92 with the 
following parameters: 
- Maximum tree depth = 15
- Maximum number of features considered per split = 10
- Maximum proportion of samples used in a tree = 0.5
- Minimum number of samples per leaf = 5
- Number of estimators = 500

A parameter-tuning example is shown in Figure \\ref{rf-param}.

### 6 Results and discussion

#### 6.1 Model selection 

Test set results} Bi-LSTM is the best performing model, with test accuracy reaching 97.79%. This test accuracy is higher than the ones achieved by previous work on this data set. LSTM was the next best model, achieving 96.64%. All other models tested achieved accuracies between 93-97% (see Table \\ref{test-acc}). Additionally, some of the simple methods that have very few parameters to tune performed very well. For example, both Multiple Linear Regression and Linear Discriminant Analysis without PCA obtained a higher accuracy than decision tree-based methods, including Random Forest and LightGBM. \\

Bi-LSTM's advantage of being able to use previous and future observations to help classify an observation is likely what set it apart from other methods tested. The data set is highly time dependent, with over 80% of observations 5 time points ahead or behind matching the observation of interest (see Figure \\ref{time-dep}).\\

It turns out that our SVM+BiLSTM model does not outperform the BiLSTM model. This might result from the fact that each BiLSTM component in the SVM+BiLSTM model was trained on a smaller data set.


BiLSTM                          & 97.79% \\
LSTM                            & 96.64% \\
LDA without PCA                 & 96.54% \\
Support vector machine          & 96.54% \\
Regularized MLR without PCA     & 96.03% \\
SVM+BiLSTM                      & 95.66% \\
LightGBM                        & 95.32% \\
MLR with PCA                    & 94.16% \\
LDA with PCA                    & 94.13% \\
Random forest                   & 93.48% \\

#### 6.2 Error analysis

Throughout all models tested, the main sources of error was confusion between walking, walking upstairs, and walking downstairs, as well as between sitting and standing. This can be easily seen in the confusion matrices (see Figure \\ref{confusion}).

#### 6.3 Performance on test set


### Conclusion
Given how widespread smart gadgets and smartphones are, predicting human activity with sensors on these devices may 
have widespread impact and applications. In our experiments, we found that bidirectional LSTM is the best at 
predicting human activity from the readings of gyroscopes and accelerometers. In addition, almost all models tested 
performed exceptionally well at separating the 6 classes into 3 groups—(walking, walking upstairs, walking 
downstairs), (sitting, standing), and (laying). However, they are less effective at differentiating classes within 
each of the three groups, likely because of similar data characteristics within groups.

#### References

\[1\] Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2012, December). Human
 activity recognition on smartphones using a multiclass hardware-friendly support vector machine. In 
 *International workshop on ambient assisted living* (pp. 216-223). Springer, Berlin, Heidelberg.

\[2\] Roobini, M. S., & Naomi, M. J. F. (2019). Smartphone sensor based human activity recognition using deep 
learning models. *Int. J. Recent Technol. Eng*, 8(1), 2740-2748.

\[3\] Anguita, D., Ghio, A., Oneto, L., Parra Perez, X., & Reyes Ortiz, J. L. (2013). A public domain dataset for 
human activity recognition using smartphones. In *Proceedings of the 21th international European symposium on 
artificial neural networks, computational intelligence and machine learning* (pp. 437-442).

\[4\] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). Lightgbm: A highly efficient
gradient boosting decision tree. *Advances in neural information processing systems*, 30.

\[5\] Gokmen, T., Rasch, M. J., & Haensch, W. (2018). Training LSTM networks with resistive cross-point devices. 
*Frontiers in neuroscience*, 745.

\[6\] Li, Y. H., Harfiya, L. N., Purwandari, K., & Lin, Y. D. (2020). Real-time cuffless continuous blood pressure 
estimation using deep learning model. *Sensors*, 20(19), 5606.

test accuracy = 96.3%
