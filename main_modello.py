from data_diabete import diabetes_data
from confronto_modello import *
from modello import *

data = diabetes_data()
default_test_size = 0.5

data.plot_diabetes()
data.plot_ages()

# logisitc regression to predict diabetes
model_1 = diabetes_logistic_regression(data, 100, default_test_size)
model_1.predict()

print("\nLogistic Regression metrics")
model_1.print_metrics()
model_1.get_confusion_matrix()

print("\n\n")

# decision tree to predict diabetes
model_2 = diabetes_decision_tree(data, 50, default_test_size)
model_2.predict()

print("Decision tree metrics")
model_2.print_metrics()
model_2.get_confusion_matrix()

print("\n\n")

# knn to predict diabetes
model_3 = diabetes_knn(data, default_test_size, 21)
model_3.predict()

print("Knn tree metrics")
model_3.print_metrics()
model_3.get_confusion_matrix()

# model comparisons
metrics_graph_lr(data,default_test_size)
metrics_graph_dt(data,default_test_size)
metrics_graph_knn(data,default_test_size)
comparison_metrics_models(data,default_test_size)



