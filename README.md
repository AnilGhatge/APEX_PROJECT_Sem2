Advanced Apex Project-1 (M.Sc. DS & AI)
Week-9 Final Submission Notebook
 
1. Predicting Credit Risk:
A Baseline Classification Model for German Credit Applicants
Student Name: Anil Ghatge
Student ID: 2025EM1200153
 
Problem Statement
Financial institutions face the challenge of accurately assessing the creditworthiness of loanapplicants. Misclassifying a 'Bad' credit risk as 'Good' can lead to significant financial losses, while incorrectly rejecting a 'Good' applicant can result in missed business opportunities. The problem is to build a predictive model that can effectively discriminate between 'Good' and 'Bad' credit applicants based on their provided profile information, minimizing the higher cost associated with false positives (approving a bad risk). 17/07/2026, 22:56 Jupyter Notebook — generated with runcell about:blank 1/31

Business Goal
The primary business objective is to develop a reliable baseline machine learning
model for credit
risk assessment. This model aims to:
* Improve credit decision-making: Provide adata-driven approach to classify
applicants.
* Mitigate financial risk: Reduce losses incurred from approving high-risk
applicants by
accurately identifying 'Bad' credit risks.
* Establish a performance benchmark: Create aninitial model that serves as a
baseline for
future, more sophisticated credit risk models, with a focus on optimizing metrics
aligned
with the given cost matrix (e.g., minimizing false positives for 'Bad' risks).
 
2. Dataset Overview
  The dataset used for this project is the German Credit Data from the UCI Machine
Learning Repository. This dataset contains information on 1000 loan applicants,
classified as either 'Good' or 'Bad' credit risks.
Key Characteristics:
* Source: UCI Machine Learning Repository (ID: 144)
* Number of Instances: 1000
* Number of Features: 20
* Target Variable: 'class' (binary: 1 = Good, 2 = Bad)
* Feature Types: A mix of numerical and categorical attributes describing various
aspects of the applicants, such as their checking account status, duration of credit,
credit history, purpose of credit, credit amount, savings account status,
employment status, personal status and sex, and age.
This dataset is commonly used for credit risk assessment and provides a realistic
scenario for developing and evaluating classification models.


From the df.describe() output, we can observe the summary statistics for
numerical features, such as Attribute2 (Duration), Attribute5 (Credit amount),
and Attribute13 (Age). This gives us an idea of their range, mean, and standard
deviation.
The df['class'].value_counts() output shows the distribution of the target
variable. We can see that there are 700 'Good' credit applicants (represented by 1)
and 300 'Bad' credit applicants (represented by 2), indicating a class imbalance.

Initial Understanding & Observations
The dataset contains 1000 observations, with 20 features and a single target
variable (class).
A significant number of features are categorical, represented by codes (e.g., 'A11',
'A32'), which
will require proper encoding for most machine learning algorithms.
The problem is a binary classification task. The target variable needs to be mapped
to a standard
numerical representation (e.g., 0 and 1).
The project description highlights a crucial cost matrix: misclassifying a 'Bad' credit
risk as 'Good'
is 5 times more costly than misclassifying a 'Good' risk as 'Bad'. This implies that
traditional
accuracy might not be the most appropriate primary evaluation metric; precision,
recall, and F1-
score, especially for the 'Bad' class, or custom cost-sensitive metrics, will be
important.
There are no reported missing values according to the dataset metadata.
 
Class Distribution of Target Variable
The target variable y needs to be analyzed for class imbalance. A balanced
dataset is crucial for model training, especially given the cost matrix where
misclassifying 'Bad' credit applicants is more expensive.
 
Identifying Numerical and Categorical Features
Based on german_credit.variables and initial inspection, we can identify which
features are numerical and which are categorical. This is important for applying
appropriate preprocessing steps like one-hot encoding for categorical features and
potential scaling for numerical features.

Unique Values in Categorical Features
Many categorical features are represented by codes (e.g., 'A11', 'A32'). Examining
the unique values for these features will provide a better understanding of their
diversity and help confirm if they are indeed categorical and how many unique
categories each feature has, which is important for encoding strategies.
 
Duplicate Checks
It's important to check for duplicate rows in the dataset, as these can bias model
training and evaluation. We will identify and count any exact duplicate rows.
 
Outlier Analysis
Outliers can significantly impact model performance. We will use box plots to
visualize potential outliers in numerical features and calculate the Interquartile
Range (IQR) to quantify them.
 
Skewness Analysis
Skewness indicates the asymmetry of the probability distribution of a real-valued
random variable about its mean. Understanding skewness is important for some
models that assume normal distribution or can be sensitive to skewed data. We will
calculate the skewness coefficient and visualize distributions using histograms.
 
Visualizing Target Variable Distribution
It is crucial to understand the distribution of the target variable ('class') to assess
class imbalance, which is vital for this credit risk prediction problem.

Visualizing Important Numerical Features
Let's visualize the distributions of key numerical features such as Attribute2
(Duration), Attribute5 (Credit amount), and Attribute13 (Age) and their
relationship with the target variable.
 
Visualizing Important Categorical Features
Now, let's look at the distributions of important categorical features like
Attribute1 (Status of existing checking account), Attribute3 (Credit history),
Attribute4 (Purpose), and Attribute9 (Personal status and sex), and how their
categories are distributed across the target variable.
 
Metric Interpretation in the Context of Misclassification Costs
Given the problem statement's emphasis on the high cost of misclassifying a 'Bad'
credit risk as 'Good' (5 times more costly than the reverse), our future metric
interpretations will explicitly prioritize this business objective.
Key considerations for model evaluation and metric selection will include:
* False Negatives (FN): These represent 'Bad' credit applicants incorrectly
classified as 'Good'. Minimizing FNs will be paramount due to their high financial
cost.
* Recall (Sensitivity) for the 'Bad' Class (Class 2): High recall for the 'Bad' class
means the model is effective at identifying most of the actual 'Bad' applicants, thus
reducing costly false positives.
* Precision for the 'Good' Class (Class 1): While prioritizing 'Bad' class recall, we
also need to maintain reasonable precision for the 'Good' class to avoid rejecting
too many creditworthy applicants.
* F1-Score and Custom Cost-Sensitive Metrics: We will explore F1-score and
potentially develop custom metrics that directly incorporate the specified cost
matrix to provide a more business-relevant evaluation of model performance.
All subsequent model building and evaluation steps will aim to strike a balance that
reflects this cost asymmetry, ensuring that the model's performance aligns with the
financial institution's risk mitigation goals.

4. Data Preparation & Preprocessing
 
Target Variable Remapping
Many machine learning algorithms expect binary target variables to be represented
as 0 and 1. Currently, our 'class' variable is encoded as 1 (Good) and 2 (Bad). We
will remap 'Good' to 0 and 'Bad' to 1 to align with common practices, especially
when dealing with cost-sensitive learning where the positive class (often the rare or
more costly class, 'Bad' credit in our case) is typically represented by 1.
 
Outlier Treatment (Capping)
Outliers, as identified in our EDA (e.g., in Attribute2 , Attribute5 , Attribute13 ,
Attribute16 , Attribute18 ), can significantly distort statistical analyses and
model training. One common strategy to handle them is capping, where extreme
values are replaced with values at a specified percentile.
We will cap the values at the 5th and 95th percentiles for the identified numerical
columns. This means any value below the 5th percentile will be set to the 5th
percentile value, and any value above the 95th percentile will be set to the 95th
percentile value.
 
Skewness Transformation (Log Transformation)
Many machine learning models perform better when numerical features are
normally distributed or at least symmetrically distributed. Our EDA showed that
several numerical features are positively skewed. Log transformation (using
np.log1p which applies log(1+x) to handle zero values) is a powerful technique
to reduce this skewness.
We will apply log transformation to the numerical columns that exhibited significant
positive skewness: Attribute2 (Duration) and Attribute5 (Credit amount).

Why Log Transformation (np.log1p) in this Case?
The choice of transformation depends on the nature and degree of skewness, as
well as the desired properties of the transformed data. Here's a comparison and
why log transformation was a suitable choice here:
* Log Transformation ( np.log1p ):
* How it works: This transformation, specifically log(1+x) , is excellent for highly
positively skewed data where values can be zero. It severely compresses large
values while expanding smaller values. This is why you saw a significant reduction in
skewness for Attribute2 (Duration) and Attribute5 (Credit amount), which were
highly skewed.
* Advantages: It's very effective for exponential-like distributions, common in
financial or duration data. It's also relatively easy to understand and interpret
conceptually (e.g., changes in log values represent proportional changes in the
original scale).
* Suitability here: Given the high positive skewness observed in features like
'Duration' and 'Credit amount', log transformation was a strong and effective choice
to bring their distributions closer to symmetry.
* Square Root Transformation ( np.sqrt ):
* How it works: This is a milder transformation than the logarithm. It also helps
reduce positive skewness but to a lesser degree. It's often suitable for data where
the variance is proportional to the mean, such as count data (e.g., Poisson
distributed data).
* Advantages: Simpler to compute than log.
Suitability here: For the highly* skewed features we encountered, a square root
transformation might not have been strong enough to achieve the desired level of
symmetry, making log a more appropriate first step.
* Box-Cox Transformation:
* How it works: This is a more general family of power transformations that
includes log and square root as special cases. It's parameterized by λ (lambda), and
the transformation finds the optimal λ that minimizes the skewness of the data,
often aiming for the most normal-like distribution.
* Advantages: It's data-driven and can often find a more 'optimal' transformation
than a pre-defined one like log or square root, potentially leading to a more
perfectly normal distribution.
* Disadvantages/Why not chosen here (initially):
* It requires all data values to be positive. If a feature contains zeros or negative
values, it cannot be directly applied without shifting the data first. Our np.log1p
handles zeros gracefully.
* It adds a layer of complexity; you need to compute and interpret the optimal
lambda. For initial EDA and feature engineering, a well-understood and effective
transformation like log is often chosen first if it clearly addresses the problem.
* While Box-Cox could have been used, the clear improvement seen with log
transformation for the highly skewed features indicated it was sufficient for the
current phase, balancing effectiveness with simplicity.
In summary, log transformation ( np.log1p ) was a practical and effective choice for
the observed high positive skewness, offering a good balance between efficacy and
interpretability. Box-Cox could be explored later if further fine-tuning for normality
is critical and computational complexity is less of a concern.
 
Categorical Feature Encoding (One-Hot Encoding)
Most machine learning algorithms cannot directly work with categorical data. One
common technique to convert these features into a numerical format is One-Hot
Encoding. This process creates new binary (dummy) variables for each category
within a categorical feature. We will apply this to our categorical_cols .
 
Scaling Numerical Features
Many machine learning algorithms, particularly those based on distance metrics
(e.g., K-Nearest Neighbors, Support Vector Machines) or gradient descent (e.g.,
Logistic Regression, Neural Networks), are sensitive to the scale and range of input
features. Features with larger values might dominate the learning process.
Standardization (using StandardScaler ) transforms the data so that it has a mean
of 0 and a standard deviation of 1. This helps in bringing all numerical features to a
comparable scale without distorting differences in the ranges of values.

Feature Interaction
Based on our EDA, specific interactions between numerical and categorical features
were identified as potentially important predictors. Creating interaction terms can
help capture more complex relationships where the effect of one feature on the
target variable is dependent on another feature.
We will create the following interaction terms:
1. Attribute2 (Transformed Duration) * Attribute1 (Checking Account
Status)
2. Attribute5 (Transformed Credit Amount) * Attribute3 (Credit History)
To do this, we first need to ensure our DataFrame is in the right state after all
previous transformations.
 
Feature Importance Analysis for Interaction Terms
After creating the interaction terms, it's crucial to evaluate their significance. We
will train a simple classification model (e.g., RandomForestClassifier ) and use its
built-in feature importance mechanism to understand how much these new features
contribute to predicting the target variable, relative to the original features.
 
Feature Selection using Tree-based Importance
Based on the RandomForestClassifier feature importances, we can now proceed
with selecting the most impactful features. This step is crucial for reducing noise,
preventing overfitting, and potentially improving model performance and
interpretability, especially after increasing dimensionality through one-hot encoding
and interaction terms.
We will use sklearn.feature_selection.SelectFromModel which allows us to
select features based on their importance weights (coefficients) from a trained
estimator. We will set a threshold for the feature importances to keep only the most
relevant features.

Model Evaluation Before and After Feature Selection
To assess the impact of feature selection, we will:
1. Train and evaluate a RandomForestClassifier using the original feature set
( X_train , X_test ).
2. Train and evaluate a RandomForestClassifier using the selected feature set
( X_train_selected , X_test_selected ).
3. Compare the performance metrics (e.g., accuracy, precision, recall, F1-score) to
determine if feature selection has improved model efficiency or predictive power

Does Feature Selection Help? Interpretation of Results
While Precision for predicting 'Bad' (Class 1) decreased from approximately 0.77
to 0.59, the Recall for 'Bad' (Class 1) significantly increased from 0.40 to 0.48.
Let's break down why this trade-off is likely beneficial for our business goal:
1. Recall is Key for 'Bad' Credit Risk:
* The problem statement explicitly states: "misclassifying a 'Bad' credit risk as
'Good' is 5 times more costly than incorrectly rejecting a 'Good' applicant." This
means False Negatives (FN) – classifying a truly 'Bad' applicant as 'Good' – are
extremely expensive.
* Recall measures the proportion of actual 'Bad' applicants that the model correctly
identified (True Positives). An increase in recall for the 'Bad' class means the model
is now catching a higher percentage of the truly risky applicants, directly
reducing those high-cost false negatives. This is a very positive outcome from a
business perspective, as it directly mitigates the most expensive error.
2. The Trade-off: Precision vs. Recall:
* A decrease in precision means that among all applicants the model predicts as
'Bad', a larger proportion might actually be 'Good' (i.e., more False Positives).
However, in a scenario with asymmetric costs, prioritizing the reduction of the most
costly error (False Negatives) often leads to accepting a slightly higher rate of False
Positives.
3. Overall F1-Score Improvement:
* The F1-Score, which is the harmonic mean of precision and recall, saw a slight
increase (0.5275 to 0.5321). This suggests that even with the drop in precision, the
gain in recall was enough to marginally improve the overall balance between the two
metrics.
Conclusion:
Yes, this feature selection likely helps us significantly by making the model
more effective at identifying 'Bad' credit applicants. This aligns directly with our
primary business objective of minimizing the severe financial losses associated with
approving high-risk individuals. The increase in recall for the 'Bad' class, despite a
precision drop, indicates a shift towards a model that is more cost-effective given
the defined cost matrix. Further tuning and exploring different classification
thresholds could optimize this balance even more.

Data Preprocessing Decisions and Observations
Following the data preprocessing steps, we have transformed the dataset into a
format suitable for machine learning models, and several key insights emerged:
1. Target Variable Remapping ( class ):
* The remapping of the target variable from (1: Good, 2: Bad) to (0: Good, 1: Bad) is
crucial. It aligns the dataset with standard machine learning conventions where '1'
often represents the positive or 'event' class, which in our cost-sensitive context, is
the 'Bad' credit risk. This prepares the data for models that naturally interpret '1' as
the class of interest, especially for metrics like recall, precision, and F1-score
focused on the minority (or more costly) class.
2. Categorical Feature Encoding (One-Hot Encoding):
* One-hot encoding successfully converted the 13 categorical features into
numerical representations. This process significantly increased the
dimensionality of our dataset (from 21 columns to 49 columns, including the
target). This increased feature space is now accessible to algorithms but also
highlights a potential need for feature selection or dimensionality reduction in later
stages to combat the curse of dimensionality.
* The drop_first=True argument in pd.get_dummies prevented multicollinearity
among the dummy variables, which is important for some linear models.
3. Numerical Feature Scaling ( StandardScaler ):
* Applying StandardScaler to the 7 numerical features normalized their
distributions to have a mean of 0 and a standard deviation of 1. This step is vital for
algorithms sensitive to feature scales, such as Logistic Regression, Support Vector
Machines, Neural Networks, and K-Nearest Neighbors. It ensures that features with
larger original ranges (e.g., Attribute5 - Credit amount) do not disproportionately
influence the model compared to features with smaller ranges.
* The describe() output for scaled features confirms that their mean is near zero
and standard deviation is near one, validating the scaling process.
In summary, the preprocessing has effectively prepared the dataset for modeling by
handling categorical data, normalizing numerical scales, and aligning the target
variable. The increase in dimensionality and the initial class imbalance remain
important considerations for subsequent model building and evaluation.

5. Exploratory Data Analysis
 
Visualizing All Feature Distributions
To get a comprehensive understanding of the dataset, let's visualize the
distributions of all numerical features using histograms and all categorical features
using bar plots. This will help us observe their individual patterns, skewness, and
category frequencies.
  #### Numerical Feature Distributions (Histograms)
  #### Categorical Feature Distributions (Bar Plots)
 
Analyzing Relationships between Features and Target
Variable
Now, let's analyze the relationships between our features and the target variable
( class ). This will help us identify features that are most predictive of credit risk.
  #### Correlation Matrix for Numerical Features
A correlation matrix is a great way to visualize the linear relationships between
numerical variables. We'll include the target variable ( class ) to see how strongly
each numerical feature correlates with credit risk.
  #### Relationship between All Categorical Features and Target Variable
For categorical features, we can use stacked bar plots to visualize how the
distribution of 'Good' and 'Bad' credit applicants varies across the categories of
each feature. This helps in understanding which categories might be more prone to
'Bad' credit.

Identifying Potential Correlations Between Features
Beyond the feature-target relationships, it's also important to understand the
relationships between the features themselves. This helps in identifying
multicollinearity, potential redundant features, or interesting interaction effects.
We've already looked at numerical-numerical correlations. Now, let's explore
categorical-categorical associations.
  #### Association Between Categorical Features (Cramer's V)
For categorical variables, a common measure of association is Cramer's V. It is a
measure of association between two nominal variables, ranging from 0 (no
association) to 1 (perfect association). We will compute a Cramer's V matrix for all
our categorical features and visualize it with a heatmap.
 
Associations Between Numerical and Categorical Features
To identify potential associations between numerical and categorical features, one
effective method is to use box plots or violin plots. These plots allow us to visually
inspect how the distribution of a numerical feature varies across different
categories of a categorical feature. If there are significant differences in the median,
spread, or overall distribution of the numerical feature across categories, it
suggests a strong association.
Let's explicitly visualize the relationship between some of the most influential
numerical and categorical features identified during our EDA. These plots will help
us understand if certain categories of a feature are associated with higher or lower
values of a numerical feature, which can be indicative of predictive power or
interesting data patterns.
We will examine the following pairs:
1. Attribute2 (Duration) vs Attribute1 (Checking Account Status): How credit
duration varies based on the applicant's checking account status.
2. Attribute5 (Credit amount) vs Attribute3 (Credit history): The distribution
of credit amounts across different credit history categories.
3. Attribute13 (Age) vs Attribute9 (Personal status and sex): How age
distributes among different personal status and sex groups.

Interpretation of Box Plots
These box plots provide visual insights into how numerical features differ across
categories of our chosen categorical features:
* For Attribute2 (Duration) vs Attribute1 (Checking Account Status): We can
observe if certain checking account statuses are associated with significantly
longer or shorter credit durations. For example, a higher median duration for certain
account types might indicate higher risk or different credit products.
* For Attribute5 (Credit amount) vs Attribute3 (Credit history): This plot
helps us understand if applicants with a particular credit history tend to request
larger or smaller credit amounts. A 'critical account' history might show a different
credit amount profile than 'no credits taken'.
* For Attribute13 (Age) vs Attribute9 (Personal status and sex): We can see
if there are noticeable age differences or variations in age distribution across
different personal status and sex categories. This can reveal demographic patterns
related to credit applications.

Key Trends, Patterns, and Insights from Exploratory
Data Analysis (EDA)
Based on our comprehensive exploratory data analysis, we've gathered several
important insights:
1. Data Integrity & Missingness:
* No duplicate rows were found in the dataset, ensuring data uniqueness.
* No missing values were identified across any features or the target variable,
simplifying initial data handling.
2. Target Variable Distribution (Credit Risk):
* A significant class imbalance exists in the target variable ( class ). The dataset
contains 70% 'Good' credit applicants (re-mapped to 0) and 30% 'Bad' credit
applicants (re-mapped to 1).
* This imbalance is critical for model training, as directly optimizing for accuracy
might lead to models biased towards the majority class. We must consider cost￾sensitive learning and evaluate models using metrics like recall, precision, and F1-
score for the 'Bad' class (minority class).
3. Numerical Features:
* Outliers: Several numerical features, particularly Attribute2 (Duration),
Attribute5 (Credit amount), Attribute13 (Age), Attribute16 (Number of
existing credits), and Attribute18 (Number of dependents), exhibit a notable
number of outliers as evidenced by box plots and IQR calculations. These outliers
could significantly influence model training and might require specific handling
(e.g., Winsorization, robust scaling, or outlier removal).
* Skewness: Many numerical features, such as Attribute2 (Duration) and
Attribute5 (Credit amount), show positive skewness. This indicates a long tail
towards higher values. Transformations (e.g., log or square root) might be beneficial
for models that assume normally distributed data or are sensitive to skewness.
* Relationships with Target: Box plots comparing numerical features against the
target variable ( class ) revealed potential differences in distributions between
'Good' and 'Bad' credit applicants. For example, 'Bad' credit applicants tend to have
longer Attribute2 (Duration) and higher Attribute5 (Credit amount) on average,
suggesting these features are strong indicators of risk.
17/07/2026, 22:56 Jupyter Notebook — generated with runcell
about:blank

* Numerical Feature Correlations: The correlation heatmap showed varying
degrees of linear relationships among numerical features. For instance,
Attribute2 (Duration) and Attribute5 (Credit amount) have a moderate positive
correlation, which is intuitive. Correlations with the target variable highlight features
like Attribute2 and Attribute5 as having the strongest linear relationship with
credit risk.
4. Categorical Features:
* Diversity: Categorical features have diverse numbers of unique values, ranging
from 2 (e.g., Attribute19 , Attribute20 ) to 10 ( Attribute4 - Purpose).
* Relationships with Target: Stacked bar plots demonstrated that the distribution
of 'Good' and 'Bad' credit risk varies significantly across categories of many
features. For example:
* Attribute1 (Checking Account Status): Certain statuses (e.g., 'A14' - no
checking account) show a higher proportion of 'Bad' credit applicants.
* Attribute3 (Credit History): Categories like 'A34' (critical account) are strongly
associated with a much higher proportion of 'Bad' credit, indicating its high
predictive power.
* Attribute4 (Purpose): Specific purposes (e.g., 'A43' - radio/television, 'A40' -
car (new)) might have different risk profiles.
* Categorical Feature Associations (Cramer's V): The Cramer's V heatmap
revealed strong associations between certain categorical features. For example,
Attribute9 (Personal status and sex) and Attribute17 (Job) show some
association, which is expected due to demographic relationships. High associations
between two features might suggest redundancy or potential multicollinearity if
both are used in models without proper handling.
5. Numerical vs. Categorical Feature Associations (Box Plots):
* Specific box plots confirmed visual associations, such as:
* Attribute2 (Duration) often being higher for certain Attribute1 (Checking
Account Status) categories, implying that the length of credit might depend on the
account status.
* Attribute5 (Credit amount) showing different distributions across Attribute3
(Credit history) categories, indicating that credit history influences the amount of
credit requested or approved.
* Attribute13 (Age) distribution varied across Attribute9 (Personal status and
sex) categories, revealing demographic patterns in age distributions relative to
personal status and sex.
These insights are invaluable for informing subsequent steps, including feature
engineering, model selection, and hyperparameter tuning, all while keeping the
critical cost of misclassification in mind.

5. Feature Engineering
 
Why Feature Selection Using RandomForestClassifier Feature
Importances?
We chose to leverage the feature importances from our RandomForestClassifier
for feature selection over other methods for several compelling reasons, especially
in the context of our preprocessed and engineered dataset:
1. Handling Complex Feature Interactions and Non-linearity:
* RandomForestClassifier is a tree-based ensemble method, inherently capable of
capturing non-linear relationships and complex interactions between features.
This is particularly relevant given that we explicitly engineered interaction terms
(e.g., Att2_x_Attribute1_A12 , Att5_x_Attribute3_A32 ). Linear models, for
example, might struggle to accurately assess the importance of these complex
terms.
* Other methods might only identify individual feature importance, whereas tree￾based methods can often inherently value features that contribute to splits, even if
their individual linear correlation is low but their interaction effect is strong.
2. Robustness to Multicollinearity:
* After one-hot encoding and creating interaction terms, our feature space
significantly expanded. This increased the likelihood of multicollinearity (high
correlation between features). While not entirely immune, tree-based models like
Random Forests are generally less sensitive to multicollinearity compared to
linear models (e.g., Logistic Regression), which can have unstable coefficients in
such scenarios. The importances derived are thus more robust.
3. Built-in Feature Importance Mechanism:
* Random Forests provide a direct and easily interpretable measure of feature
importance (typically Gini importance or mean decrease in impurity). This makes it
straightforward to rank features and apply a threshold (like the mean importance we
used) to select the most relevant ones. This is more intuitive than trying to interpret
coefficients from highly regularized linear models or complex model-agnostic
methods if not carefully implemented.
4. Scaling Invariance:
* Unlike distance-based methods (e.g., PCA) or some regularization techniques that
are sensitive to feature scaling, RandomForestClassifier is a scale-invariant
algorithm. Its decision-making process is based on relative order and thresholds,
not the absolute magnitude of values. This simplifies the feature selection process,
as the importance scores are not biased by the units or ranges of the original

5. Effective for High-Dimensional Data:
* Our one-hot encoding expanded the dataset to 55 features. Random Forests are
well-suited for high-dimensional data, and their feature importance mechanism
provides an efficient way to reduce this dimensionality, helping to combat the
curse of dimensionality and prevent overfitting by focusing on the most
informative signals.
6. Alignment with Problem Context (Cost Sensitivity):
* By using a RandomForestClassifier trained with class_weight='balanced' , we
already incorporated awareness of the class imbalance. The feature importances
derived from such a model are therefore more likely to reflect the features that are
crucial for distinguishing the minority (and costly) 'Bad' class, aligning with our
business objective.
In summary, RandomForestClassifier 's ability to handle non-linearities,
interactions, and high-dimensional data, combined with its direct and robust feature
importance mechanism, made it an ideal choice for performing feature selection
effectively after our extensive preprocessing and engineering steps.

Summary of Created/Transformed Features and Justification
for Current Feature Engineering State
We've undertaken a comprehensive set of feature engineering and transformation
steps to prepare our dataset for modeling. Here's a summary of the key changes
and the rationale behind them:
#### 1. Target Variable Remapping
* Transformation: The original target variable ('class' with values 1:Good, 2:Bad)
was remapped to (0:Good, 1:Bad). The 'Bad' credit risk was assigned 1 to align
with common machine learning practices where the positive or 'event' class, often
the minority or more critical class (in our case, the costly one), is represented by
1 .
* Impact: Essential for algorithms that expect binary targets (0/1) and for
consistent interpretation of classification metrics (e.g., recall, precision, F1-score
for the positive class).
#### 2. One-Hot Encoding of Categorical Features
* Transformation: All 13 original categorical features were converted into numerical
(binary) format using one-hot encoding. drop_first=True was used to prevent
multicollinearity.
* Impact: Expanded the feature space significantly, making categorical data
digestible for most machine learning algorithms. This increased dimensionality also
prompted the need for subsequent feature selection.
#### 3. Numerical Feature Scaling
* Transformation: Numerical features were standardized using StandardScaler ,
resulting in a mean of 0 and a standard deviation of 1.
* Impact: Ensured that features with larger ranges do not dominate the learning
process of algorithms sensitive to feature scale (e.g., SVMs, Logistic Regression,
Neural Networks).
#### 4. Outlier Treatment (Capping)
* Transformation: Outliers in Attribute2 , Attribute5 , Attribute13 ,
Attribute16 , and Attribute18 were capped at the 5th and 95th percentiles.
* Impact: Reduced the undue influence of extreme values on model training,
leading to more robust models without removing valuable data points.
#### 5. Skewness Transformation (Log Transformation)
* Transformation: Highly skewed numerical features ( Attribute2 , Attribute5 ,
Attribute16 , Attribute18 ) underwent np.log1p (log(1+x)) transformation.
* Impact: Brought the distributions of these features closer to symmetry, which can
improve the performance of models that assume or perform better with normally
distributed inputs.

#### 6. Feature Interactions
* Creation: New interaction terms were created based on EDA insights:
* Attribute2 (transformed duration) multiplied by Attribute1 (one-hot encoded
checking account status dummies).
* Attribute5 (transformed credit amount) multiplied by Attribute3 (one-hot
encoded credit history dummies).
* Impact: Allowed the model to capture more complex, non-linear relationships
where the effect of one feature is conditional on another, potentially enhancing
predictive power.
#### Why Additional Feature Engineering is Not Required at This Stage:
At this point, extensive feature engineering has been completed, addressing several
key aspects identified during EDA:
1. Comprehensive Preprocessing: We have handled target remapping, categorical
encoding, and numerical scaling, making the data fully compatible with a wide
range of ML models.
2. Robust Outlier and Skewness Handling: Outliers were capped, and highly
skewed features were transformed, mitigating common data quality issues that can
degrade model performance.
3. Explored Interactions: Key feature interactions, hypothesized to be important
based on EDA, have been explicitly created and added to the dataset.
4. Feature Selection Applied: Crucially, we've already performed feature
selection using RandomForestClassifier importances, which implicitly suggests
that the most informative features (including our engineered ones) are now isolated.
This process inherently reduces noise and redundancy by focusing on features that
already demonstrate predictive value within a robust model context.
5. Baseline Model Objective: The primary goal at this stage is to establish a
reliable baseline machine learning model. The current set of engineered features
is rich and diverse enough to support this objective. Over-engineering features
beyond this point might lead to diminishing returns, increased complexity, or even
overfitting, which we are actively trying to prevent.
Therefore, the current feature set is well-prepared and optimized for building and
evaluating our baseline classification models. Any further feature engineering would
likely be considered during advanced model tuning or iteration, after the
performance of the current baseline is thoroughly understood.

6. Model Implementation: Baseline
RandomForestClassifier
Now that we have prepared our data and selected the most relevant features, we
will implement a baseline RandomForestClassifier . This model is chosen for its
robustness, ability to handle both numerical and categorical features, and its
inherent mechanism for feature importance. We will train it on the
X_train_selected and y_train data and evaluate its performance on the
X_test_selected and y_test data.
 
Confusion Matrix
A confusion matrix provides a detailed breakdown of correct and incorrect
classifications, which is especially insightful for imbalanced datasets and when
specific misclassification costs are involved.
 
7. Model Optimization: Hyperparameter Tuning with
GridSearchCV
To further improve the performance of our RandomForestClassifier , especially in
light of the class imbalance and the business objective of minimizing costly false
negatives, we will perform hyperparameter tuning. GridSearchCV systematically
works through multiple combinations of parameter values, cross-validating each
combination to determine which set of parameters best optimizes a chosen
evaluation metric.
For our credit risk problem, we will focus on optimizing for recall of the 'Bad' class
(Class 1), as misclassifying a 'Bad' credit risk as 'Good' is significantly more costly.

Evaluation of the Tuned Model
After finding the best hyperparameters, let's evaluate the performance of the
best_rf_model on the unseen test set ( X_test_selected ) using the same metrics
as before, paying close attention to recall for the 'Bad' class and the confusion
matrix.

8. Basic Evaluation & Interpretation
Observations and Findings:
* Initial Data State: The dataset initially presented a class imbalance (70% 'Good',
30% 'Bad' credit risks) and a mix of numerical and categorical features. Several
numerical features showed skewness and outliers.
* Preprocessing Impact: Our comprehensive preprocessing pipeline addressed
these issues through outlier capping, log transformations for skewed features, one￾hot encoding for categorical variables, and standardization for numerical features.
This created a robust feature set ready for modeling.
* Feature Engineering & Selection: Interaction terms were created to capture
complex relationships. Feature selection using RandomForestClassifier 's
importances effectively reduced the dimensionality from 55 to 13 features, focusing
on the most predictive signals.
* Baseline Model Performance: The initial RandomForestClassifier (using
selected features and class_weight='balanced' ) yielded an accuracy of 0.73,
precision for 'Bad' class of 0.5682, and recall for 'Bad' class of 0.4167. The
confusion matrix highlighted 35 false negatives (costly misclassifications).
Classification Insights:
* Impact of Hyperparameter Tuning: Hyperparameter tuning with GridSearchCV ,
optimizing for recall of the 'Bad' class, significantly improved the model's ability to
identify 'Bad' credit risks.
* Recall for 'Bad' class increased from 0.4167 (baseline) to 0.6833 (tuned).
This is a crucial improvement, directly mitigating the most expensive type of error
(False Negatives).
* Precision for 'Bad' class saw a slight decrease from 0.5682 to 0.5256. This is
an expected trade-off when prioritizing recall in imbalanced, cost-sensitive
scenarios. The gain in correctly identified 'Bad' cases outweighs the slight increase
in 'Good' cases being incorrectly flagged as 'Bad'.
* F1-Score for 'Bad' class improved from 0.4808 to 0.5942, indicating an overall
better balance between precision and recall for the critical 'Bad' class.
* Confusion Matrix Analysis (Tuned Model): The tuned model's confusion matrix
shows a reduction in False Negatives (from 35 to 19), which was our primary
business objective. This means the model is now much better at catching actual
high-risk applicants.

Conclusion:
The hyperparameter-tuned RandomForestClassifier is a substantially more
effective model for this credit risk assessment problem, particularly given the
asymmetric costs of misclassification. By prioritizing and achieving a higher recall
for 'Bad' credit applicants, the model is better equipped to reduce the financial
institution's exposure to high-risk loans, aligning well with the stated business goal.






