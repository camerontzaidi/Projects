# Car Insurance Claim Prediction Model - README

## Description of Data and Exploration

This project utilizes historic data on insurance claims, policyholder demographics, previous claims, and general vehicle information to build a claim status prediction model. The dataset, sourced from Kaggle by Sergey Litvinenko, consists of 27 qualitative and 12 quantitative variables. To process and standardize the input for the model, categorical variables were encoded into numeric ones, and binary variables were assigned 1s and 0s. The data was cleaned by dropping missing values and the ‘policy_id’ column, as it was not relevant for prediction.

### Data Exploration

Key features for claim status prediction were identified using the Random Forest algorithm on unsupervised clustering (DBSCAN). Adjustable steering was found to be a significant factor, likely due to its presence in newer and higher-end vehicles. Other top-ranked features include front fog lights, brake assistance, and day/night rearview mirrors, associated with advanced safety and convenience features. Subscription length also emerged as an important feature, indicating that older customers and those with larger, high-performance vehicles tend to have longer subscriptions.

## Methods

### Data Preparation

Class imbalance was addressed using SMOTE (Synthetic Minority Oversampling Technique) and the data was split into 60% training and 40% testing sets. 

### Neural Network Model

A Sequential model architecture was defined with an input layer, Batch Normalization, Dropout layers, a hidden layer, and an output layer with a sigmoid activation function. The model was compiled with the Adam optimizer and trained with the best hyperparameters found through Hyperopt:
- **Activation**: relu
- **Batch size**: 113
- **Dropout rate**: 0.24
- **Epochs**: 36
- **Neurons**: 191
- **Optimizer**: adam

### Random Forest Model

The Random Forest algorithm was applied with k-fold cross-validation to enhance model robustness. The model included 100 decision trees (n_estimators), used the square root of the number of features for max_features, optimized depth (max_depth), minimum samples for split (min_samples_split), and minimum samples for leaf (min_samples_leaf). Feature pruning was performed using prior feature importance analysis, and GridSearchCV identified the best hyperparameters for fine-tuning.

## Results

The Random Forest model achieved a high ROC-AUC score of 0.6979 and an F1-score of 0.89, indicating balanced performance in precision and recall. Extensive data preprocessing, strategic handling of class imbalances through SMOTE, and meticulous hyperparameter tuning via GridSearchCV and Hyperopt contributed to these results. The Random Forest model excelled with feature pruning and k-fold cross-validation, ensuring robustness and generalizability, outperforming simpler methodologies typically seen in other insurance claim studies.

## Discussion

This approach demonstrated several strengths, including robust data preprocessing, effective class imbalance handling through SMOTE, and meticulous hyperparameter tuning with GridSearchCV and Hyperopt. The Neural Network's ROC-AUC score of 0.6979 and the Random Forest model's overall accuracy and F1-score of 0.89 were notable. The choice of a 60/40 data split may have impacted generalizability compared to the commonly used 80/20 split. Exploring more advanced feature engineering techniques, additional ensemble methods, and deeper neural network architectures could improve the models further. Future work could focus on these enhancements and experiment with other oversampling and undersampling techniques to handle class imbalances more effectively.

## Conclusion

This project demonstrated the effectiveness of robust data preprocessing, strategic handling of class imbalances, and meticulous hyperparameter tuning. Important features such as advanced safety and convenience features and customer subscription length were influential in the claim prediction model. Despite the strengths, areas for improvement include exploring advanced feature engineering techniques and additional ensemble methods. Future enhancements could increase predictive accuracy and robustness, solidifying the models' utility in real-world insurance claim prediction scenarios.
