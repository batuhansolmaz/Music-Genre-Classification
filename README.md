# Music Genre Classification 

## Dataset

### 1. Reason for Dataset Collection
This project aims to classify music genres using machine learning techniques. To achieve this, we needed a diverse and representative dataset. By combining the dataset that we prepared with the labeled music genres dataset that we found on kaggle, we created a more varied dataset that reflects different audio clip lengths. Specifically, we included shorter 10-second clips and longer 30-second clips to ensure our model could generalize well to real-world scenarios where audio clip lengths may vary.

### 2. Data Collection Procedure
The dataset was compiled from two sources:
- **Jamendo API**: We fetched 10-second audio clips with labeled genres to introduce short audio samples into the dataset.
- **Kaggle dataset**: This dataset provided 30-second audio clips with genre labels, which added diversity and allowed us to analyze longer audio samples.

To improve the dataset, we applied filtering to remove problematic categories:
- **Blues**: Identified as a subcategory of `rock`, it caused classification confusion and was removed.
- **Electronic**: This genre also caused inconsistencies and was excluded.

The final dataset includes 6 distinct classes.

### 3. Dataset Overview
| Metric                  | Value                                      |
|-------------------------|--------------------------------------------|
| Number of samples       | 910                                       |
| Number of categories    | 6                                         |
| Samples per category    | Hiphop: 158, Pop: 155, Metal: 155, Rock: 150, Jazz: 147, Classical: 145 |
| Dimensionality of data  | 60                                        |

### 4. Feature Extraction Procedure
We extracted a total of 57 features from the audio files using the `librosa` library. These features captures the key aspects of the audio signals:
- **Chroma features**: Capture the pitch class distribution.
- **Spectral features**: Include spectral centroid, bandwidth, and rolloff, which describe frequency characteristics.
- **RMS energy**: Measures the average power of the signal.
- **MFCCs**: 20 coefficients representing the timbre of the audio, with both mean and variance values.
- **Zero crossing rate**: Reflects percussive characteristics of the audio.
- **Harmonic and percussive features**: Capture distinct audio components.
- **Tempo**: Extracted using beat tracking.

All features were retained since the number of samples exceeded the product of the number of classes and features, ensuring no overfitting due to high dimensionality.

#### Example Feature Extraction Code
```python
# Example code snippet for feature extraction
def extract_features(file_path, genre, data_type):
    y, sr = librosa.load(file_path, duration=duration)
    features = {
        'chroma_stft_mean': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        'rms_mean': np.mean(librosa.feature.rms(y=y)),
        # Additional features omitted for brevity
    }
    return features
```

### 5. Data Preprocessing
We applied several preprocessing steps to clean and prepare the data:
1. **Data Cleaning**: Removed invalid entries and replaced infinite values with NaN.
2. **Label Encoding**: Converted genre labels into numeric values for modeling.
3. **Feature Standardization**: Standardized features using `StandardScaler` to normalize the input data.
4. **Train-Test Split**: Separated the dataset into training and testing sets based on the `data_type` column.

#### Example Preprocessing Code
```python
# Example preprocessing function
def prepare_features(df):
    df = clean_numeric_columns(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    features = df.drop(['data_type', 'genre', 'length'], axis=1)
    target = df['genre']
    return features, target
```
- **Feature Correlation Heat Map and Top Features**: 
![image](https://hackmd.io/_uploads/SJn1gpVBkx.png)


### 6. Intra-Class and Inter-Class Similarities
#### Intra-Class Similarities
Below are the computed intra-class similarities. Lower the value more tightly clusters are:

| Class      | Intra-Class Similarity |
|------------|-------------------------|
| Classical  | 564,613.79             |
| Hiphop     | 2,791,285.83           |
| Jazz       | 1,818,487.06           |
| Metal      | 686,659.62             |
| Pop        | 1,987,818.72           |
| Rock       | 1,479,262.45           |

#### Observations:
- **Classical** and **Metal** genres exhibit lower intra-class similarities, indicating their samples are more consistent and tightly grouped.
- **Hiphop** and **Pop** have higher intra-class similarities, which tells more diversity within these genres.

#### Inter-Class Similarities
Below are the computed inter-class similarities:

| Class Pair           | Inter-Class Similarity |
|----------------------|-------------------------|
| Classical vs Hiphop  | 3,208,968.71           |
| Classical vs Jazz    | 1,718,715.74           |
| Classical vs Metal   | 724,186.76             |
| Classical vs Pop     | 2,989,528.43           |
| Classical vs Rock    | 1,551,230.31           |
| Hiphop vs Jazz       | 2,692,978.39           |
| Hiphop vs Metal      | 2,890,149.06           |
| Hiphop vs Pop        | 2,431,196.63           |
| Hiphop vs Rock       | 2,576,002.95           |
| Jazz vs Metal        | 1,500,165.51           |
| Jazz vs Pop          | 2,349,819.67           |
| Jazz vs Rock         | 1,650,238.10           |
| Metal vs Pop         | 2,632,922.51           |
| Metal vs Rock        | 1,314,040.08           |
| Pop vs Rock          | 2,219,985.11           |

#### Observations:
- **Classical vs Metal** has the lowest inter-class similarity, highlighting their distinct features.
- **Hiphop vs Jazz** and **Hiphop vs Metal** show high inter-class similarities, suggesting some overlap in their feature characteristics.


### 7. Feature Importance Analysis
#### Top 10 Features by ANOVA F-test
| Feature                  | F-Score |
|--------------------------|---------|
| chroma_stft_mean         | 260.16  |
| mfcc1_mean               | 260.12  |
| spectral_bandwidth_mean  | 158.89  |
| rolloff_mean             | 146.63  |
| spectral_centroid_var    | 136.28  |
| spectral_centroid_mean   | 122.02  |
| rms_mean                 | 118.04  |
| percussive_var           | 110.29  |
| rms_var                  | 106.56  |
| chroma_stft_var          | 96.98   |

#### PCA Analysis
- **Number of PCA components to preserve 95% variance**: 33
- **Note**: All features were retained as the sample size was sufficiently large compared to the number of features.

### 8. Kmeans Clustering Analysis
![Ekran Resmi 2024-12-22 14.13.06](https://hackmd.io/_uploads/ry__4dSS1g.png)

Comparing the true labels (left plot) with k-means clustering (right plot), we can observe that while the original data has 6 classes with significant overlap, particularly in the central part k-means attempts to create more geometrically defined divisions especially along Feature 1. The centroids are positioned to minimize within-cluster distances, but because the class overlap is high ,it is difficult for k-means to match the true class structure. So , unsupervised clustering alone may not be sufficient for this classification task and explaining why supervised methods might be more appropriate for this dataset.



## Supervised Learning 

## Logistic Regression 

We implemented a logistic regression classifier from scratch using gradient descent optimization.Looking at the Loss History graph, both training and validation loss curves follow very similar patterns and decrease smoothly over iterations. The small gap between training and validation loss curve indicates that the model is not suffering from significant overfitting. The model training was stopped using a convergence criterion at iteration 827. The stopping criteria was based on monitoring the change in loss between iterations.If the change in training loss remains smaller than the tolerance threshold as we choose it 1e-4 for a specified number of consecutive iterations we stop![Ekran Resmi 2024-12-20 19.57.43](https://hackmd.io/_uploads/BJ2OOXXS1e.png)

The final model achieved the following accuracies:

Test Accuracy: 0.7198 (71.98%)
Training Accuracy: 0.7253 (72.53%)

The small different indicates that we are not suffering from overfitting.

#### Comparing scratch implementation and skicit logistic regression

Custom Implementation:
Training time: 27.1781 seconds
Test accuracy: 0.7198
Average AUROC: 0.9180
Average F1: 0.4618
Average Precision: 0.4097
Average Recall: 0.8467

Scikit-learn Implementation:
Training time: 0.0699 seconds
Test accuracy: 0.7198
Average AUROC: 0.9180
Average F1: 0.4618
Average Precision: 0.4097
Average Recall: 0.8467


Scikit-learn Implementation:
Training time: 0.0699 seconds
Test accuracy: 0.7198
Average AUROC: 0.9180
Average F1: 0.4618
Average Precision: 0.4097
Average Recall: 0.8467

##### Comparison
Both the custom and Scikit-learn logistic regression models achieve similar performance metrics, including test accuracy, AUROC, Precision, Recall, and F1 score. There is no noticeable difference in terms of classification quality.Scikit-learn's implementation is much faster, taking only 0.07 seconds, whereas the custom implementation takes 27.18 seconds, making it around 388 times slower.


##### Custom Implementation Metrics:
Overall Accuracy: 0.7198

Here's the data formatted as a markdown table:

| Class      | AUROC  | Precision | Recall | F1     |
|------------|--------|-----------|--------|--------|
| classical  | 0.9899 | 0.4475    | 0.9086 | 0.5009 |
| hiphop     | 0.9254 | 0.4320    | 0.8477 | 0.4778 |
| jazz       | 0.9457 | 0.4151    | 0.8712 | 0.4676 |
| metal      | 0.9364 | 0.4297    | 0.8628 | 0.4790 |
| pop        | 0.8949 | 0.4137    | 0.8243 | 0.4567 |
| rock       | 0.8155 | 0.3202    | 0.7659 | 0.3887 |
| **Average**| 0.9180 | 0.4097    | 0.8467 | 0.4618 |

Training time: 27.1781 seconds

##### Scikit-learn Implementation Metrics:

Overall Accuracy: 0.7198

Here's both the header and table formatted in Markdown:

### Per-Class Metrics:

| Class      | AUROC  | Precision | Recall | F1     |
|------------|--------|-----------|--------|--------|
| classical  | 0.9899 | 0.4475    | 0.9086 | 0.5009 |
| hiphop     | 0.9254 | 0.4320    | 0.8477 | 0.4778 |
| jazz       | 0.9457 | 0.4151    | 0.8712 | 0.4676 |
| metal      | 0.9364 | 0.4297    | 0.8628 | 0.4790 |
| pop        | 0.8949 | 0.4137    | 0.8243 | 0.4567 |
| rock       | 0.8155 | 0.3202    | 0.7659 | 0.3887 |
| **Average**| 0.9180 | 0.4097    | 0.8467 | 0.4618 |

Training time: 0.0699 seconds

## Support Vector Machines

We implemented a linear soft-margin SVM using CVXOPT's quadratic programming solver. The implementation follows the one-vs-all strategy for multi-class classification. The quadratic programming formulation is:

![Ekran Resmi 2024-12-22 21.20.55](https://hackmd.io/_uploads/S1viu0BH1g.png)


### Model Evaluation

1. Classification Accuracy (ACC):
Training ACC: 0.857
Testing ACC: 0.731

2. Area Under ROC Curve (AUROC):
classical: 0.938
hiphop: 0.829
jazz: 0.870
metal: 0.812
pop: 0.809
rock: 0.777
Average AUROC: 0.839

Training:
              
              precision    recall  f1-score   support

    classical     0.96      0.97      0.96       116
    hiphop        0.93      0.89      0.91       126
    jazz          0.84      0.86      0.85       118
    metal         0.82      0.85      0.83       124
    pop           0.83      0.86      0.85       124
    rock          0.77      0.72      0.74       120
    accuracy                          0.86       728
    macro avg     0.86      0.86      0.86       728
    weighted avg  0.86      0.86      0.86       728


Testing:
              
                 precision    recall  f1-score   support

    classical     0.90        0.90      0.90        29
    hiphop        0.72        0.72      0.72        32
    jazz          0.74        0.79      0.77        29
    metal         0.72        0.68      0.70        31
    pop           0.70        0.68      0.69        31
    rock          0.61        0.63      0.62        30
    accuracy                            0.73       182
    acro avg      0.73        0.73      0.73       182
    weighted avg  0.73        0.73      0.73       182


Per-Class Performance (Test Set):
                   
                 
              precision    recall  f1-score   support

    classical      0.90      0.90      0.90        29
      hiphop       0.72      0.72      0.72        32
        jazz       0.74      0.79      0.77        29
       metal       0.72      0.68      0.70        31
         pop       0.70      0.68      0.69        31
        rock       0.61      0.63      0.62        30
    accuracy                           0.73       182
    macro avg      0.73      0.73      0.73       182
    weighted avg   0.73      0.73      0.73       182


6. Confusion Matrix (Test Set):

Confusion Matrix:
True\Pred	
                       
                       classic	hiphop	jazz	metal	pop	rock	

           classic	     26	          0	  1	   0	 1	 1	
           hiphop	     0	         23	  2	   1	 4	 2	
            jazz	     2	         0	 23	   1	 3	 0	
            metal	     0	         2	  0	   21	 1	 7	
            pop	     1	         4	  2	   1	21	 2	
            rock	     0	         3	  3	   5	 0	19


### Scikit-learn SVMs with Different Kernels


Classification Report for linear kernel (Training):
              
              precision    recall  f1-score   support

           0       0.97      0.99      0.98       116
           1       0.95      0.96      0.96       126
           2       0.92      0.94      0.93       118
           3       0.90      0.90      0.90       124
           4       0.94      0.93      0.93       124
           5       0.90      0.86      0.88       120
    accuracy                           0.93       728
    macro avg      0.93      0.93      0.93       728
    weighted avg   0.93      0.93      0.93       728



Classification Report for linear kernel (Test):
                
                precision    recall  f1-score   support

           0       0.84      0.90      0.87        29
           1       0.70      0.81      0.75        32
           2       0.69      0.76      0.72        29
           3       0.74      0.65      0.69        31
           4       0.73      0.61      0.67        31
           5       0.72      0.70      0.71        30

    accuracy                           0.74       182
    macro avg      0.74      0.74      0.73       182
    weighted avg   0.74      0.74      0.73       182
    

AUROC Scores:
0: 0.976
1: 0.952
2: 0.950
3: 0.946
4: 0.937
5: 0.921
Average AUROC for linear: 0.947

Training SVM with rbf kernel...

Classification Report for rbf kernel (Training):
              
              precision    recall  f1-score   support

           0       0.93      0.97      0.95       116
           1       0.95      0.92      0.94       126
           2       0.90      0.92      0.91       118
           3       0.91      0.85      0.88       124
           4       0.90      0.90      0.90       124
           5       0.80      0.83      0.82       120

    accuracy                           0.90       728
    macro avg      0.90      0.90      0.90       728
    weighted avg   0.90      0.90      0.90       728


Classification Report for rbf kernel (Test):
    
                precision    recall  f1-score   support

           0       0.89      0.83      0.86        29
           1       0.68      0.88      0.77        32
           2       0.79      0.79      0.79        29
           3       0.88      0.68      0.76        31
           4       0.90      0.61      0.73        31
           5       0.62      0.83      0.71        30

    accuracy                            0.77       182
    macro avg       0.79      0.77      0.77       182
    weighted avg    0.79      0.77      0.77       182
    
    
AUROC Scores:
0: 0.986
1: 0.953
2: 0.973
3: 0.953
4: 0.943
5: 0.935
Average AUROC for rbf: 0.957

Training SVM with poly kernel...

Classification Report for poly kernel (Training):
              
               precision    recall  f1-score   support

           0       0.98      0.91      0.95       116
           1       1.00      0.82      0.90       126
           2       0.76      0.95      0.85       118
           3       0.96      0.73      0.83       124
           4       0.98      0.70      0.82       124
           5       0.60      0.94      0.74       120

    accuracy                           0.84       728
    macro avg       0.88      0.84      0.85       728
    weighted avg    0.88      0.84      0.84       728


Classification Report for poly kernel (Test):
              
              precision    recall  f1-score   support

           0       0.92      0.83      0.87        29
           1       0.88      0.47      0.61        32
           2       0.65      0.90      0.75        29
           3       0.94      0.48      0.64        31
           4       0.82      0.58      0.68        31
           5       0.41      0.83      0.55        30

    accuracy                            0.68       182
    macro avg       0.77      0.68      0.68       182
    weighted avg    0.77      0.68      0.68       182


AUROC Scores:
0: 0.988
1: 0.929
2: 0.973
3: 0.951
4: 0.933
5: 0.895
Average AUROC for poly: 0.945

Training SVM with sigmoid kernel...

Classification Report for sigmoid kernel (Training):
              
              precision    recall  f1-score   support

           0       0.86      0.87      0.86       116
           1       0.74      0.71      0.72       126
           2       0.70      0.74      0.72       118
           3       0.69      0.76      0.72       124
           4       0.65      0.70      0.68       124
           5       0.63      0.50      0.56       120

    accuracy                            0.71       728
    macro avg       0.71      0.71      0.71       728
    weighted avg    0.71      0.71      0.71       728


Classification Report for sigmoid kernel (Test):
                 
                 precision    recall  f1-score   support

           0       0.80      0.83      0.81        29
           1       0.63      0.75      0.69        32
           2       0.81      0.76      0.79        29
           3       0.69      0.77      0.73        31
           4       0.64      0.58      0.61        31
           5       0.58      0.47      0.52        30

    accuracy                           0.69       182
    macro avg      0.69      0.69      0.69       182
    weighted avg   0.69      0.69      0.69       182
    

AUROC Scores:
0: 0.969
1: 0.920
2: 0.971
3: 0.944
4: 0.892
5: 0.882
Average AUROC for sigmoid: 0.930


### Hyperparamter Tuning

Fitting 5 folds for each of 24 candidates, totalling 120 fits
Best parameters: {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
Best cross-validation score: 0.7569

Best Model Configuration:
Kernel: rbf
C: 1
Gamma: scale

### Final Evaluation

Training Accuracy: 0.897
Testing Accuracy : 0.769

2. Area Under ROC Curve (AUROC):
classical: 0.904
hiphop: 0.894
jazz: 0.877
metal: 0.829
pop: 0.800
rock: 0.867
Average AUROC: 0.862
Training:
                 
                 precision    recall  f1-score   support

           0       0.93      0.97      0.95       116
           1       0.95      0.92      0.94       126
           2       0.90      0.92      0.91       118
           3       0.91      0.85      0.88       124
           4       0.90      0.90      0.90       124
           5       0.80      0.83      0.82       120

       accuracy                        0.90       728
       macro avg   0.90       0.90     0.90       728
       weighted avg 0.90      0.90     0.90       728


Testing:
                 
                 precision    recall  f1-score   support

           0       0.89      0.83      0.86        29
           1       0.68      0.88      0.77        32
           2       0.79      0.79      0.79        29
           3       0.88      0.68      0.76        31
           4       0.90      0.61      0.73        31
           5       0.62      0.83      0.71        30

    accuracy                            0.77       182
    macro avg       0.79      0.77      0.77       182
    weighted avg    0.79      0.77      0.77       182



Execution Time:

Custom SVM: 4.02 seconds
Sklearn SVM: 0.04 seconds
Speed Difference: 114.0x longer for custom implementation


Performance Metrics:

Custom SVM:
- Accuracy: 0.7308
- F1-Score: 0.7306

Sklearn SVM:
- Accuracy: 0.7692
- F1-Score: 0.7701

Accuracy Difference: 3.8% higher for sklearn

#### Comparison between Skicit-learn and Scratch SVM
Scikit-learn includes specialized optimizations and is implemented in highly efficient languages like C++, while our custom implementation relies on the Python-based CVXOPT solver. This difference leads to limitations in handling higher-dimensional data efficiently for the CVXOPT solver. As a result, it is normal that our custom implementation is 114 times slower than Scikit-learn’s SVM.

Moreover, Scikit-learn utilizes advanced algorithms, such as Sequential Minimal Optimization (SMO), which are specifically designed for efficient and scalable SVM training. In contrast, our implementation employs a generic quadratic programming solver, which leads significantly higher computational overhead. While our approach is good for understanding how SVM works , it cannot match the performance and efficiency of Scikit-learn’s optimized implementation.



## Random Forest

### Hyperparameter Tuning
Best parameters: {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 16, 'min_samples_split': 16, 'n_estimators': 200}
Best cross-validation score: 0.7156636750118092

### Runtime
Training time: 0.77 seconds

### Evaluation 
Training Metrics:
Accuracy: 0.8255
AUROC: 0.9745
Precision: 0.8247
Recall: 0.8255
F1-Score: 0.8244

Test Metrics:
Accuracy: 0.7143
AUROC: 0.9404
Precision: 0.7179
Recall: 0.7143
F1-Score: 0.7150


## KNN classifier

Cross-validation results:
k=3: 0.7225
k=5: 0.7294
k=7: 0.7376
k=9: 0.7417
k=11: 0.7170

Best k value: 9



Training Metrics:
Overall Accuracy: 0.8104

Per-Class Metrics:

        Class                AUROC  Precision     Recall         F1

        classical           0.9949     0.7947     0.7923     0.6962
        hiphop              0.9841     0.8291     0.5902     0.5511
        jazz                0.9572     0.7590     0.5362     0.4562
        metal               0.9786     0.8091     0.6195     0.5798
        pop                 0.9597     0.7264     0.6503     0.5597
        rock                0.9443     0.6776     0.6038     0.4645
      
        Average             0.9698     0.7660     0.6320     0.5512

Test Metrics:
Overall Accuracy: 0.7088

Per-Class Metrics:
        
        Class                AUROC  Precision     Recall         F1
        classical           0.9706     0.7558     0.7429     0.6400
        hiphop              0.9461     0.7636     0.5188     0.4575
        jazz                0.9492     0.7447     0.5110     0.4054
        metal               0.9661     0.8112     0.5419     0.5085
        pop                 0.9091     0.7286     0.5894     0.5220
        rock                0.8979     0.5798     0.6030     0.4127
        Average             0.9398     0.7306     0.5845     0.4910

Training time: 0.0001 seconds
## Performance and Runtime Comparison

In this section, we contrast the results of all the supervised learning models we trained—Logistic Regression, Support Vector Machine (SVM), Random Forest, and K-Nearest Neighbors (KNN)—in terms of their key performance metrics and training times. But more detailed results are at the above section. While each model has its strengths, our goal is to identify how they differ in accuracy, F1 score, AUROC, and computational efficiency.

**1. Logistic Regression**  
- **Accuracy**: Both our custom implementation and Scikit-learn’s version reached about 72% on the test set.  
- **F1 Score**: Approximately 0.46 (averaged across classes).  
- **AUROC**: High at around 0.918, meaning it ranks classes reasonably well despite the moderate accuracy.  
- **Runtime**: The custom version took about 27 seconds, whereas Scikit-learn’s implementation took only ~0.07 seconds—a 388× speed difference.  

**2. Support Vector Machines (SVM)**  
- **Custom SVM (CVXOPT)**:  
  - Test accuracy of roughly 73%.  
  - Training took about 4 seconds.  
  - Relies on generic quadratic programming, so it’s slower and slightly less accurate than the optimized library.  
- **Scikit-learn SVM**:  
  - **Linear Kernel**: ~74% accuracy, ~0.947 average AUROC.  
  - **RBF Kernel (Tuned)**: ~77% accuracy (best among all SVM variants), high AUROC (~0.86–0.96 depending on the specific evaluation).  
  - **Polynomial and Sigmoid Kernels**: Accuracy hovered around 68–69%.  
  - **Runtime**: Generally very fast (~0.04 seconds for training), thanks to specialized SMO-based solvers and compiled optimizations.

**3. Random Forest**  
- **Accuracy**: About 71% on the test set after hyperparameter tuning.  
- **AUROC**: ~0.94 on the test set, indicating strong ranking capability.  
- **F1 Score**: ~0.715 on the test set.  
- **Runtime**: Moderate (~0.77 seconds) for training.  
- Random Forest provides good interpretability via feature importances and can handle complex feature interactions well, though its accuracy here did not surpass the best SVM.

**4. K-Nearest Neighbors (KNN)**  
- **Best \(k\) Found**: 9  
- **Accuracy**: ~81% on the training set, ~71% on the test set.  
- **AUROC**: ~0.94 on the test set, which is quite high.  
- **Runtime**: Training is extremely fast (~0.0001 seconds), but inference can be slower at scale since KNN must compare each new point to all training samples.

**Summary of Comparisons**  
- **Highest Accuracy**: RBF SVM (~77%).  
- **Overall AUROC**: Many models (Logistic Regression, Random Forest, KNN, SVM) show high AUROCs (~0.90+), but SVM with RBF remains the strongest in accuracy and robust ranking metrics.  
- **Fastest Training**: Scikit-learn’s implementations (Logistic Regression, SVM) and KNN’s minimal “training.”  
- **Slowest Implementations**: Custom Logistic Regression (~27 seconds) and custom SVM (~4 seconds).

---

## Best-Performing Model and Reasoning

Among all the classifiers tested, the **Scikit-learn SVM with the RBF kernel** achieved the highest test accuracy (~77%) and maintained a consistently strong AUROC across all classes. Several factors contribute to this:

1. **Adaptive Nonlinear Boundaries**  
   Using the RBF kernel allows the SVM to flexibly shape its decision boundaries, which is really helpful when different genres overlap in the raw feature space. Instead of drawing straight lines or simple curves between classes, the RBF kernel can “bend” the boundaries to better separate music genres that share similar characteristics.

2. **Hyperparameter Tuning**  
   We used grid search to find the best `C` (regularization) and `gamma` (kernel coefficient) values. This step made sure the model didn’t overfit by being too strict (high `C`) or too loose (low `C`). With the right combination of `C` and `gamma`, the SVM managed to learn a good boundary without memorizing the training data too closely.

3. **Efficient Implementation**  
   Scikit-learn’s SVM runs on an optimized solver (SMO) written in efficient, compiled code. This gives it two main benefits: fast training times and stable convergence even with a relatively large number of features. As a result, the RBF SVM not only handled our dataset’s complexity but also did so faster than our custom implementations or less optimized libraries.

Overall, the RBF SVM turned out to be the best performer because it strikes a balance between flexibility, speed, and accuracy. It handles the overlapping clusters in our dataset better than simpler methods, making it the top choice among all the classifiers we tested.




## Evaluation of ACC, AUROC , Average Precision , Recall and F1-score

 Classification Accuracy(ACC) measures the ratio of correct predictions to total predictions. While, it can be misleading with imbalanced classes. Area Under ROC Curve (AUROC) evaluates the model's ability to distinguish between classes across different thresholds. AUROC is important because it is not sensitive to class imbalance which is more powerful than ACC. It helps evaluate model performance without choosing a specific threshold. Average Precision on the other hand , focuses on accuracy of positive predictions, indicates how many of our predicted positive cases were actually correct. Average Recall, tells us how many actual positive cases we successfully identified and F1-score is the harmonic mean of precision and recall, it provides a balanced measure of both metrics. 
 
 AUROC and F1-score generally provide more realistic assessments of model performance when there is class balance this is because AUROC is insensitive to class imbalance as we disscussed , on the other hand F1-score balances precision and recall. Simple Accuraccy might be misleading when there is class imbalance since predicting by majority class it gets high accuracy which might be misleading .
 
Our models' performanced can be assessed using accuracy since our dataset has balanced classes across music genres. The similar scores across metrics and between training and test sets suggest our model is learning meaningful patterns in the music genre classification task without overfitting.
 



## The Challenges we Faced

Before extracting and removing electronics and blues from our dataset, its performance was poor. The blues and rock features were performing similarly, and the accuracy (ACC) was relatively low. On the other hand, the electronic genre had a broader distribution and did not exhibit any meaningful differentiation. The SVM from scracth was another difficult part . We get confused to write it mathematically correct in some parts . Another challanges was chosing two feature and two genre for plotting the decision boundary . We tried to choose the best for visualization . Also we tried to fix endless bugs and errors while writing the code which was very time consuming.


# Plotting Decision Boundary

## Logistic Regression 
![Ekran Resmi 2024-12-20 22.51.48](https://hackmd.io/_uploads/HJckormS1x.png)

## Linear Soft Margin SVM
![Ekran Resmi 2024-12-20 22.52.42](https://hackmd.io/_uploads/BJc4iHXHJl.png)

## Both on the same scatter
![Ekran Resmi 2024-12-20 22.53.49](https://hackmd.io/_uploads/rJ_Ojrmr1l.png)


### Differences

They have different angles for the two boundaries above which might indicate they've found different optimal separating hyperplanes The SVM boundary appears to create a wider separation between the classes, prioritizing margin maximization Logistic Regression's boundary seems to be more influenced by the overall class distribution
SVM creates a more strict boundary with explicit margins Logistic Regression provides a smoother transition between classes The overlap region shows this is not a perfectly separable case
SVM appears less influenced by outliers due to its maximum-margin approach Logistic Regression's boundary shows more sensitivity to the overall distribution, including outliers.

## References

- [Jamendo API](https://www.google.com/search?client=safari&rls=en&q=Jamendo+api&ie=UTF-8&oe=UTF-8) which was the main part of our dataset collection
- [Kaggle Gtzan dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) which was our insight when we decide to do this project . Also we used some music files from here for improving our dataset.
- [Skicit Learn Documentation](https://scikit-learn.org/stable/modules/svm.html) Helped us to use skicit-learn
- [CVXOPT Documentation](https://cvxopt.org/userguide/index.html) helped us when writing SVM from scratch
- [Geeksforgeeks](https://www.geeksforgeeks.org/) which has insigthful tutorials helped us how we going to do the development part.


## URL for Zip file
https://drive.google.com/file/d/1OuilZC-waDvS53u6f-uUhYeXCo7pfVtx/view?usp=sharing


