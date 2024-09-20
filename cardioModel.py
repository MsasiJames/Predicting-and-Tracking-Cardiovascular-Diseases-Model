import pandas as pd
import numpy as np
import seaborn as sns
import random
import matplotlib.pyplot as plt

data = pd.read_csv("D:\Year 3\Investigations\cardiovascular_diseases\cardio_data_processed.csv")

print(data.head(5))                                                             # print out first 5 rows of the data

print(data.info())

# the last two columns are basically the same need to drop one of them
num = len(data["bp_category_encoded"])
different_data_count = 0
for i in range(num):
    if data["bp_category_encoded"][i] != data["bp_category"][i]:
        different_data_count += 1
        
print(different_data_count)                                                             # to if data in all the columns are the same or not

if(different_data_count > 0):
    print("These two columns are not the same")
else:
    print("The two columns are the same.")
    data.drop("bp_category_encoded", axis=1, inplace=True)                      # drop the last column, no use
    
print(data.head(5))                                                             # see if the column has been dropped
                                                    
data.drop("age", axis=1, inplace=True)                                          # remove age in days column from dataset, since it has no use
data.drop("id", axis=1, inplace=True)                                           # remove id column from dataset, since it has no use

missing_values = data.isna().sum()                                              # check if any of the columns had missing values

result_df = pd.DataFrame(missing_values, columns=['missing values'])

result_df.reset_index(inplace=True)

result_df.columns = ['column', 'missing values']

print(result_df)

print(data.describe())                                                          # get summary of statistics

positive = data[data["cardio"] == 1]
negative = data[data["cardio"] == 0]

print("Cardio presence", positive["cardio"].value_counts())
print("Cardio not present", negative["cardio"].value_counts())

# Graph one - piechart on the distribution of gender in ths data
plt.figure(figsize=(12, 7))
plt.pie(data["gender"].value_counts(), autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
plt.title("Gender spread\n\n1: Women, 2:Men")
plt.show()

# Graph one part two - how cardiovascular diseases are distributed between the two genders
plt.figure(figsize=(10, 6))

men_data = data[data['gender'] == 2]
women_data = data[data['gender'] == 1]

men_cvd_counts = men_data['cardio'].value_counts()
women_cvd_counts = women_data['cardio'].value_counts()

sns.barplot(x=['Men (0)', 'Men (1)', 'Women (0)', 'Women (1)'],
            y=[men_cvd_counts[0], men_cvd_counts[1], women_cvd_counts[0], women_cvd_counts[1]],
            palette=['#66b3ff', '#ff9999', '#66b3ff', '#ff9999'])
plt.title('Distribution of Cardiovascular Disease by Gender')
plt.ylabel('Count')
plt.xlabel('Gender and Cardiovascular Disease Status')
plt.show()

# Graph two - Blood pressure category
target_column = data["bp_category"]
counts = target_column.value_counts()

counts.plot(kind='bar', color=['#ff9999', '#66b3ff'])
plt.title('Blood pressure category distribution')
plt.xlabel("Blood pressure category")
plt.ylabel("Count")
plt.show()

# Graph two part one - Blood pressure with exercising factored in.
active_data = data[data['active'] == 1]
inactive_data = data[data['active'] == 0]

active_counts = active_data['bp_category'].value_counts()
inactive_counts = inactive_data['bp_category'].value_counts()

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
active_counts.plot(kind='bar', color='#66b3ff')
plt.title('Blood Pressure Category Distribution (Active)')
plt.xlabel("Blood Pressure Category")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
inactive_counts.plot(kind='bar', color='#ff9999')
plt.title('Blood Pressure Category Distribution (Inactive)')
plt.xlabel("Blood Pressure Category")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# Graph three - Cholesterol level count
sns.countplot(data=data, x='cholesterol')
plt.title("Cholesterol level count plot")
plt.xlabel("1:Normal, 2:Above normal, 3:Well above normal")
plt.show()

# Graph three part one - Cholesterol level count with cardio factored in
plt.figure(figsize=(10, 6))

sns.countplot(data=data, x='cholesterol', hue='cardio', palette={0: '#66b3ff', 1: '#ff9999'})

plt.title("Cholesterol Level Count Plot by Cardiovascular Disease")
plt.xlabel("Cholesterol Level\n(1: Normal, 2: Above Normal, 3: Well Above Normal)")
plt.ylabel("Count")
plt.legend(title='Cardio', labels=['No CVD', 'CVD'])

plt.show()

# Graph four - Active subjects count
data["active"].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
plt.title("Number of active subjects")
plt.xlabel("0: Don't do exercises  1: Do exercises")
plt.show()

# Graph five - Cardiovascular diseases distribution
data["cardio"].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
plt.title("Cardiovascular presence distribution")
plt.xlabel("0: Absent  1: Present")
plt.show()



# Ordinal encoding
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

OE = OrdinalEncoder(categories=[["Normal", "Elevated", "Hypertension Stage 1", "Hypertension Stage 2"]])
OE.fit(data[["bp_category"]])
data["bp_encoded"] = OE.transform(data[["bp_category"]])

data.drop("bp_category", axis=1, inplace=True)                                  # now this column is not needed after encoding

print(data.head(20))



sns.pairplot(data,
             hue="cardio",
             x_vars=["ap_hi", "ap_lo", "bmi", "gluc", "active", "cholesterol"], 
             y_vars=["ap_hi", "ap_lo", "bmi", "gluc", "active", "cholesterol"],
             markers=["o", "s"]
)
plt.show()

sns.pairplot(data,
             hue="cardio",
             x_vars=["ap_hi", "ap_lo", "bmi"], y_vars=["smoke", "alco", "gender"],
             markers=["o", "s"]
)
plt.show()

sns.pairplot(data,
             hue="cardio",
             x_vars=["bmi", "ap_hi", "ap_lo"],
             y_vars=["cholesterol", "smoke", "alco", "active"],
             markers=['o', 's'])
plt.show()

# Correlation matrix
matrix = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title("Correlation matrix")
plt.show()


from scipy import stats
cardio_positive = data[data["cardio"] == 1]
cardio_negative = data[data["cardio"] == 0]

risk_factors = ["cholesterol", "smoke", "alco", "active", "bmi", "ap_hi", "ap_lo"]

for factor in risk_factors:
    data_yes = cardio_positive[factor]
    data_no = cardio_negative[factor]
    t_stat, p_value = stats.ttest_ind(data_yes, data_no)
    print(f"Test for factor {factor}")
    print("T-statistic", t_stat)
    print("P-value", p_value)

sample_size = int(len(data) * 0.01)
sample = data.sample(n=sample_size, random_state=42)

statistic, p_value = stats.shapiro(sample)
alpha = 0.05
print("Shapiro-Wilk Test:")
print("Statistic:", statistic, "p-value:", p_value)
if p_value > alpha:
    print("Data looks Gaussian (fail to reject H0)")
else:
    print("Data does not look Gaussian (reject H0)")
    
from scipy.stats import chi2_contingency

risk_factors = ["cholesterol", "smoke", "alco", "active", "bmi", "ap_hi", "ap_lo"]

for factor in risk_factors:
    contingency_table = pd.crosstab(data[factor], data["cardio"])
    
    chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
    
    print(f"Chi-square test for {factor}:")
    print(f"Chi-square statistic: {chi2_stat}")
    print(f"P-value: {p_val}")
    
    if p_val < 0.05:
        print("There is a significant association between", factor, "and cardiovascular disease.")
    else:
        print("There is no significant association between", factor, "and cardiovascular disease.")
    print()

    
# ----------------------------------------------------------------------MODEL BUILDING FROM HERE ------------------------------------------------------------
    
# Split data to train and test
x = data.drop('cardio', axis=1)
y = data['cardio']


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


#-----------------------------------------------------------------Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn import metrics

logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)

log_pred = logmodel.predict(x_test)
print("Logistic Regression")
print("Accuracy of the model: ", accuracy_score(log_pred, y_test))
print(classification_report(y_test, log_pred))

#-----------------------------------------------------------------Decision Tree
from sklearn.tree import DecisionTreeClassifier


tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)

tree_pred = tree.predict(x_test)
print("Decision Tree")
print("Accuracy of the model: ", accuracy_score(tree_pred, y_test))
print(classification_report(y_test, tree_pred))


predictionsTwo = tree.predict(x_test.head(1))       # this is how you feed the data
print(x_test.head(1))
print(f"Predicted outcome: {predictionsTwo}")                               # see its outcome


# Retrieve feature importances from the trained Decision Tree model
importances = tree.feature_importances_

# Map feature importances to corresponding feature names
feature_importance_map = dict(zip(x_test.columns, importances))

# Sort feature importances in descending order
sorted_importances = sorted(feature_importance_map.items(), key=lambda x: x[1], reverse=True)

# Print or visualize the sorted feature importances
for feature, importance in sorted_importances:
    print(f"Feature: {feature}, Importance: {importance}")




#----------------------------------------------------------------- Random forest
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier()
forest.fit(x_train, y_train)

forest_pred = forest.predict(x_test)
print("Random Forest")
print("Accuracy of the model: ", accuracy_score(forest_pred, y_test))
print(classification_report(y_test, forest_pred))

#-----------------------------------------------------------------Support Vector Machine
from sklearn.svm import SVC

svm_model = SVC(probability=True)
svm_model.fit(x_train, y_train)
svm_pred = svm_model.predict(x_test)
print("Support Vector Machine")
print("Accuracy of the model: ", accuracy_score(svm_pred, y_test))
print(classification_report(y_test, svm_pred))

#-----------------------------------------------------------------Naive bayes
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(x_train, y_train)
nb_pred = nb_model.predict(x_test)
print("Naive bayes")
print("Accuracy of the model: ", accuracy_score(nb_pred, y_test))
print(classification_report(y_test, nb_pred))

#-----------------------------------------------------------------Gradient Boosting Machine
from sklearn.ensemble import GradientBoostingClassifier
import joblib

gbm_model = GradientBoostingClassifier()
gbm_model.fit(x_train, y_train)
gbm_predict = gbm_model.predict(x_test)
print("Gradient Boosting Machine")
print("Accuracy of the model: ", accuracy_score(gbm_predict, y_test))
print(classification_report(y_test, gbm_predict))

joblib.dump(gbm_model, 'gbm_model.pkl')

# Calculate ROC curve and AUC for the initial model
fpr_gbm, tpr_gbm, _ = roc_curve(y_test, gbm_model.predict_proba(x_test)[:, 1])
roc_auc_gbm = auc(fpr_gbm, tpr_gbm)

#-----------------------------------------------------------------Neural Network
from sklearn.neural_network import MLPClassifier

nn_model = MLPClassifier()
nn_model.fit(x_train,y_train)
nn_predict = nn_model.predict(x_test)
print("Neural Network")
print("Accuracy of the model: ", accuracy_score(nn_predict, y_test))
print(classification_report(y_test, nn_predict))


# # #----------------------------------------------------------------- Overfitting test
from sklearn.model_selection import cross_val_score

gbm_model_test = gbm_model.predict(x_train)
train_accuracy = accuracy_score(gbm_model_test, y_train)

cv_scores = cross_val_score(gbm_model, x, y, cv=10)

mean_cv_score = np.mean(cv_scores)
std_cv_score = np.std(cv_scores)

print("Mean CV Score:", mean_cv_score)
print("Standard Deviation of CV Scores:", std_cv_score)

disparity = train_accuracy - mean_cv_score
print(disparity)


# # ----------------------------------------------------------------- SHAP analysis
# GBM model------------------------------------------------------------------------------------------------------------
import shap
shap.initjs()
np.random.seed(42)      # to make same sampling reproducible
background = shap.sample(x_train, 100)  # reduce size for performance

# get shap values
explainer = shap.Explainer(gbm_model)       # load the model
shap_values = explainer(x_test)           # see what you want to explain

shap.summary_plot(shap_values, x_test, feature_names=x.columns)

shap.plots.bar(shap_values)

# waterfall plot for one observation
shap.plots.waterfall(shap_values[0])

shap.plots.waterfall(shap_values[10])

shap.plots.waterfall(shap_values[200])

shap.plots.waterfall(shap_values[300])

# Naive Bayes ------------------------------------------------------------------------------------------------------------

# Define a prediction function that returns the prediction probabilities
explainer = shap.Explainer(nb_model.predict_proba, background)
shap_values = explainer(x_test)

# # since shap summary plot is very computational intensive, another method must be adopted
from sklearn.preprocessing import StandardScaler
def calculate_feature_importance(model, X, y, feature_names):
    class_means = []
    for class_label in np.unique(y):
        class_means.append(np.mean(X[y == class_label], axis=0))
    
    mean_diff = np.abs(class_means[1] - class_means[0])
    
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_diff
    })
    importance = importance.sort_values('importance', ascending=False).reset_index(drop=True)
    
    return importance

# Get feature names before scaling
feature_names = x_train.columns.tolist()

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x_train)

# Calculate feature importance
importance = calculate_feature_importance(nb_model, X_scaled, y_train, feature_names)

# Plot the feature importance
plt.figure(figsize=(12, 6))
plt.bar(importance['feature'][:10], importance['importance'][:10])
plt.title('Top 10 Feature Importance for Naive Bayes Model')
plt.xlabel('Features')
plt.ylabel('Importance (Absolute difference in means)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# waterfall plot for one observation
shap.plots.waterfall(shap_values[0][:, 0])

shap.plots.waterfall(shap_values[10][:, 0])

shap.plots.waterfall(shap_values[0][:, 1])

shap.plots.waterfall(shap_values[10][:, 1])

# Neural Network ------------------------------------------------------------------------------------------------------------

explainer = shap.Explainer(nn_model.predict_proba, background)
shap_values = explainer(x_test)

shap.plots.waterfall(shap_values[0][:, 0])

shap.plots.waterfall(shap_values[10][:, 0])

shap.plots.waterfall(shap_values[0][:, 1])

shap.plots.waterfall(shap_values[10][:, 1])

# Logistic Regression ------------------------------------------------------------------------------------------------------------

explainer = shap.Explainer(logmodel.predict_proba, background)
shap_values = explainer(x_test)

shap.plots.waterfall(shap_values[0][:, 0])

shap.plots.waterfall(shap_values[10][:, 0])

shap.plots.waterfall(shap_values[0][:, 1])

shap.plots.waterfall(shap_values[10][:, 1])

# ------------------------------------------------------------------ GRADIENT BOOSTING MODEL HYPERPARAMETER TUNING
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 1.0]
}


gbm_model = GradientBoostingClassifier(random_state=1)
grid_search = GridSearchCV(estimator=gbm_model, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2, scoring='accuracy')


grid_search.fit(x_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

best_gbm_model = grid_search.best_estimator_
gbm_predict = best_gbm_model.predict(x_test)

print("Gradient Boosting Machine")
print("Accuracy of the model: ", accuracy_score(gbm_predict, y_test))
print(classification_report(y_test, gbm_predict))

fpr_gbm_tuned, tpr_gbm_tuned, _ = roc_curve(y_test, best_gbm_model.predict_proba(x_test)[:, 1])
roc_auc_gbm_tuned = auc(fpr_gbm_tuned, tpr_gbm_tuned)

# ------------------------------------------------------------------ PLOTTING THE ROC CURVES

plt.figure()

# Plot ROC curve for the initial model
plt.plot(fpr_gbm, tpr_gbm, color='blue', lw=2, label='Initial GBM (AUC = %0.2f)' % roc_auc_gbm)

# Plot ROC curve for the tuned model
plt.plot(fpr_gbm_tuned, tpr_gbm_tuned, color='red', lw=2, label='Tuned GBM (AUC = %0.2f)' % roc_auc_gbm_tuned)

# Plot diagonal line for reference
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison for Initial and Tuned GBM Models')
plt.legend(loc="lower right")
plt.show()
