# Machine-learning-based-efficient-prediction-of-positive-cases-of-waterborne-diseases
In this project, various machine learning classifiers are implemented which predicts waterborne disease efficiently.
We have implemented the machine learning model proposed in Hussain, M., Cifci, M. A., Sehar, T., Nabi, S., Cheikhrouhou, O., Maqsood, H., ... & Mohammad, F. (2023). Machine learning based efficient prediction of positive cases of waterborne diseases. BMC Medical Informatics and Decision Making, 23(1), 1-16 and we also implemented a few ensemble models in addition.
https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-022-02092-1

## Table of Contents
    Introduction
    Features
    Model
      Dataset
      Data pre-processing
      Feature selection
      Model selection
    Model
    Evaluation
      Performance metrics
      Comparison of accuracy of classifiers
    References
    
## Introduction
Machine learning (ML) is a discipline of artificial intelligence that provides machines with the ability to automatically learn from data and past experiences to identify patterns and make predictions with minimal human intervention. In the current study, various ML models are first trained and tested using the 10-fold cross-validation approach on the dataset that was gathered for the study. The models use the seven most important features for prediction. Typhoid and malaria patient data for the years 2017–2020 from Ayub Medical Hospital is used in this project. Further, the importance of input features in waterborne disease-positive case detection is investigated. Several machine learning algorithms are applied and compared for performance and accuracy in the prediction of waterborne diseases. Age, history, and test results were found to be significant factors in predicting cases of waterborne disease positivity, according to experimental results using the random forest as a feature selection technique. Finally, one can infer that this fascinating study might assist various health departments in predicting waterborne diseases.

## Features
The primary goal is to discover and determine the chances of spreading of water-borne disease across the region of concern and to find the machine learning model that performs the best at predicting positive cases based on patient history.
The proposed architecture includes data pre-processing, feature selection, and machine learning-based classification. The dataset used in the base paper contains 19 attributes. The Typhoid dataset has 68624 entries and malaria has 22,916 entries.

## Model
### Dataset
Typhoid and malaria patient data for the years 2017–2020 from Ayub Medical Hospital is used in this paper, which is also attached.
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/WBHUV5
### Data pre-processing:
The pre-processing procedures are applied to both the typhoid and malaria dataset. The pre-processing procedures include Data cleaning, Data balancing, Data transformation, and Data normalization.
### Feature selection
The original dataset contains 19 features, out of which the top 7 features are selected by feature selection. Feature selection is performed to select the most suitable features to train the model. In this study, it is performed using the random forest feature selection method. It is a popular method for feature selection because it is effective at identifying the most relevant features while also being relatively computationally efficient. Random Forest feature selection
is performed by training a random forest model on the entire dataset and then the importance of every feature is ranked based on the feature importance scores produced by the model.
### Model selection:
The dataset is tested and trained by various machine learning models such as RF, DT, KNN, logistic regression and SVM using ten cross-validation methods using Sklearn python library. These algorithms are easily explainable, interpretable, implemented and used in many fields with good performance, such as education and medicine. The Ensemble method such as Voting, Adaboosting, Bagging, Stacking and XGBoosting are also implemented and evaluated.
Typhoid Dataset

## Evaluation
### Performance Metrics
Accuracy is used to evaluate the performance of the model. Confusion matrix and ROC curve are also employed for each classifier to evaluate the performance
### Comparison of accuracy of all algorithms on typhoid
Model            	        Accuracy
Random forest	              0.69
Decision tree    	          0.62
K-Nearest Neighbor	        0.65
Logistic Regression	        0.55
Support Vector Machine	    0.60
Voting classifier ensemble	0.65
Adaboost	                  0.63

### Comparison of accuracy of all algorithms on Malaria
Model	Accuracy
Random forest	0.80
Decision tree	0.77
K-Nearest Neighbor	0.65
Logistic Regression	0.65
Support Vector Machine	0.51
CNN	0.59
Voting classifier ensemble	0.81
AdaBoost	0.75
Bagging	0.78
XGBoost	0.78
Stacking	0.80

## References

- Hussain, M., Cifci, M. A., Sehar, T., Nabi, S., Cheikhrouhou, O., Maqsood, H., & Mohammad, F. (2023). Machine learning based efficient prediction of positive cases of waterborne diseases. BMC Medical Informatics and Decision Making, 23(1), 1-16.

- [Scikit-learn](https://scikit-learn.org) - Pedregosa et al., 2011. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.

- [Pandas](https://pandas.pydata.org) - McKinney, W., 2010. Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference, 51-56.

We are grateful to the authors of the above-mentioned papers and the maintainers of open-source projects for their valuable work and contributions to the field of machine learning and data analysis.



