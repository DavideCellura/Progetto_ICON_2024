import numpy as np
from sklearn.model_selection import *

from confronto_modello import *
from modello import *

data = diabetes_data()
std_list=[]

#Setup della tabella
X = data.get_data().drop('diabetes', axis='columns').values
Y = data.get_data()['diabetes'].values


# logisitc regression to predict diabetes

# Liste per memorizzare le metriche
logistic_regression_accuracy_scores = []
logistic_regression_precision_scores = []
logistic_regression_recall_scores = []
logistic_regression_f1_scores = []
logistic_regression_confusion_matrices = []

#Divido il dataset con la Kfold
kf_logistic = KFold(n_splits=5, shuffle=True, random_state=42)

# Esegui la K-Fold Cross Validation
logistic_model = LogisticRegression(max_iter=1000)

#scores = cross_val_score(logistic_model, X, Y, cv=kf_logistic)

nfold = 1 
print("Logistic Regression")

# Ciclo su ogni fold
for train_index, test_index in kf_logistic.split(X):
    
    
    #model_1.predict()
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    scaler = StandardScaler()
    x_train_scaled=scaler.fit_transform(X_train)
    x_test_scaled=scaler.fit_transform(X_test)

    # Allena il modello
    logistic_model.fit(x_train_scaled ,  y_train)

    # Fai le previsioni
    y_pred = logistic_model.predict(x_test_scaled)

    # Calcola le metriche
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Salva i risultati
    logistic_regression_precision_scores.append(precision)
    logistic_regression_recall_scores.append(recall)
    logistic_regression_f1_scores.append(f1)
    logistic_regression_accuracy_scores.append(accuracy)
    
    
    print("K fold",nfold)
    print(f"Accuracy: {float(accuracy):.6f}")
    print(f"Precision : {float(precision):.6f}")
    print(f"Recall: {float(recall):.6f}")
    print(f"F1-Score: {float(f1):.6f}\n")

    # Calcola e salva la matrice di confusione
    cm = confusion_matrix(y_test, y_pred)
    cm_percentage = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100  # Matrice di confusione in percentuale
    logistic_regression_confusion_matrices.append(cm_percentage)
    
    # Crea una griglia di subplot con 1 riga e 2 colonne
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot della matrice di confusione normale
    disp_matrix = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp_matrix.plot(ax=axes[0])
    axes[0].grid(False)
    axes[0].set_title("Confusion matrix Logistic Regression")
    
    # Plot della matrice di confusione in percentuale
    disp_matrix2 = ConfusionMatrixDisplay(confusion_matrix=cm_percentage)
    disp_matrix2.plot(ax=axes[1])
    axes[1].grid(False)
    axes[1].set_title("Confusion matrix percentage Logistic Regression")
    
    plt.show()
    
    nfold = nfold+1
    
  

# Calcola la media e la deviazione standard delle metriche
mean_accuracy_logistic_regression = np.mean(logistic_regression_accuracy_scores)
std_accuracy = np.std(logistic_regression_accuracy_scores)
mean_precision_logistic_regression = np.mean(logistic_regression_precision_scores)
std_precision = np.std(logistic_regression_precision_scores)
mean_recall_logistic_regression = np.mean(logistic_regression_recall_scores)
std_recall = np.std(logistic_regression_recall_scores)
mean_f1_logistic_regression = np.mean(logistic_regression_f1_scores)
std_f1 = np.std(logistic_regression_f1_scores)

std_list.append([std_accuracy,std_f1,std_precision,std_recall])


# Mostra i risultati

print("\nMetriche Logistic Regression")
print(f"Accuracy media: {float(mean_accuracy_logistic_regression):.6f}, Deviazione standard: {float(std_accuracy):.6f}")
print(f"Precisione media: {float(mean_accuracy_logistic_regression):.6f}, Deviazione standard: {float(std_precision):.6f}")
print(f"Recall media: {float(mean_recall_logistic_regression):.6f}, Deviazione standard: {float(std_recall):.6f}")
print(f"F1-Score medio: {float(mean_f1_logistic_regression):.6f}, Deviazione standard: {float(std_f1):.6f}")

print("\n\n")


#______________________________________________________________________________


# decision tree to predict diabetes


# Liste per memorizzare le metriche
decision_tree_accuracy_scores = []
decision_tree_precision_scores = []
decision_tree_recall_scores = []
decision_tree_f1_scores = []
decision_tree_confusion_matrices = []

kf_decision = KFold(n_splits=5, shuffle=True, random_state=42)

decision_model = tree.DecisionTreeClassifier(max_depth=100)

nfold = 1 
print("\nDecision Tree")

for train_index, test_index in kf_decision.split(X):

    #model_1.predict()
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    scaler = StandardScaler()
    x_train_scaled=scaler.fit_transform(X_train)
    x_test_scaled=scaler.fit_transform(X_test)

    # Allena il modello
    decision_model.fit(x_train_scaled ,  y_train)

    # Fai le previsioni
    y_pred = decision_model.predict(x_test_scaled)

    # Calcola le metriche
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Salva i risultati
    decision_tree_accuracy_scores.append(accuracy)
    decision_tree_precision_scores.append(precision)
    decision_tree_recall_scores.append(recall)
    decision_tree_f1_scores.append(f1)
    
    print("K fold",nfold)
    print(f"Accuracy: {float(accuracy):.6f}")
    print(f"Precision : {float(precision):.6f}")
    print(f"Recall: {float(recall):.6f}")
    print(f"F1-Score: {float(f1):.6f}\n")


    # Calcola e salva la matrice di confusione
    cm = confusion_matrix(y_test, y_pred)
    cm_percentage = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100  # Matrice di confusione in percentuale
    decision_tree_confusion_matrices.append(cm_percentage)

    # Crea una griglia di subplot con 1 riga e 2 colonne
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot della matrice di confusione normale
    disp_matrix = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp_matrix.plot(ax=axes[0])
    axes[0].grid(False)
    axes[0].set_title("Confusion matrix Decision Tree")
    
    # Plot della matrice di confusione in percentuale
    disp_matrix2 = ConfusionMatrixDisplay(confusion_matrix=cm_percentage)
    disp_matrix2.plot(ax=axes[1])
    axes[1].grid(False)
    axes[1].set_title("Confusion matrix percentage Decision Tree")
    
    plt.show()
    
    nfold = nfold+1

# Calcola la media e la deviazione standard delle metriche
mean_accuracy_decision_tree = np.mean(decision_tree_accuracy_scores)
std_accuracy = np.std(decision_tree_accuracy_scores)
mean_precision_decision_tree = np.mean(decision_tree_precision_scores)
std_precision = np.std(decision_tree_precision_scores)
mean_recall_decision_tree = np.mean(decision_tree_recall_scores)
std_recall = np.std(decision_tree_recall_scores)
mean_f1_decision_tree = np.mean(decision_tree_f1_scores)
std_f1 = np.std(decision_tree_f1_scores)

std_list.append([std_accuracy,std_f1,std_precision,std_recall])

# Mostra i risultati
print("\nMetriche Decision Tree")
print(f"Accuracy media: {float(mean_accuracy_decision_tree):.6f}, Deviazione standard: {float(std_accuracy):.6f}")
print(f"Precisione media: {float(mean_accuracy_decision_tree):.6f}, Deviazione standard: {float(std_precision):.6f}")
print(f"Recall media: {float(mean_recall_decision_tree):.6f}, Deviazione standard: {float(std_recall):.6f}")
print(f"F1-Score medio: {float(mean_f1_decision_tree):.6f}, Deviazione standard: {float(std_f1):.6f}")


print("\n\n")

#______________________________________________________________________________

# knn to predict diabetes

# Liste per memorizzare le metriche
knn_accuracy_scores = []
knn_precision_scores = []
knn_recall_scores = []
knn_f1_scores = []
knn_confusion_matrices = []

kf_knn = KFold(n_splits=5, shuffle=True, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=50)

nfold = 1 
print("\nKnn")

for train_index, test_index in kf_knn.split(X):

    #model_1.predict()
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    scaler = StandardScaler()
    x_train_scaled=scaler.fit_transform(X_train)
    x_test_scaled=scaler.fit_transform(X_test)

    # Allena il modello
    knn_model.fit(x_train_scaled ,  y_train)

    # Fai le previsioni
    y_pred = knn_model.predict(x_test_scaled)

    # Calcola le metriche
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Salva i risultati
    knn_accuracy_scores.append(accuracy)
    knn_precision_scores.append(precision)
    knn_recall_scores.append(recall)
    knn_f1_scores.append(f1)
    
    print("K fold",nfold)
    print(f"Accuracy: {float(accuracy):.6f}")
    print(f"Precision : {float(precision):.6f}")
    print(f"Recall: {float(recall):.6f}")
    print(f"F1-Score: {float(f1):.6f}\n")
    

    # Calcola e salva la matrice di confusione
    cm = confusion_matrix(y_test, y_pred)
    cm_percentage = cm.astype(float) / cm.sum(axis=1, keepdims=True) # Matrice di confusione in percentuale
    knn_confusion_matrices.append(cm_percentage)


    # Crea una griglia di subplot con 1 riga e 2 colonne
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot della matrice di confusione normale
    disp_matrix = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp_matrix.plot(ax=axes[0])
    axes[0].grid(False)
    axes[0].set_title("Confusion matrix Knn")
    
    # Plot della matrice di confusione in percentuale
    disp_matrix2 = ConfusionMatrixDisplay(confusion_matrix=cm_percentage)
    disp_matrix2.plot(values_format=".2%",ax=axes[1])
    axes[1].grid(False)
    axes[1].set_title("Confusion matrix percentage Knn")
    
    plt.show()
    
    nfold = nfold+1


# Calcola la media e la deviazione standard delle metriche
mean_accuracy_knn = np.mean(knn_accuracy_scores)
std_accuracy = np.std(knn_accuracy_scores)
mean_precision_knn = np.mean(knn_precision_scores)
std_precision = np.std(knn_precision_scores)
mean_recall_knn = np.mean(knn_recall_scores)
std_recall = np.std(knn_recall_scores)
mean_f1_knn = np.mean(knn_f1_scores)
std_f1 = np.std(knn_f1_scores)

std_list.append([std_accuracy,std_f1,std_precision,std_recall])

# Mostra i risultati
print("\nMetriche Knn")
print(f"Accuracy media: {float(mean_accuracy_knn):.6f}, Deviazione standard: {float(std_accuracy):.6f}")
print(f"Precisione media: {float(mean_accuracy_knn):.6f}, Deviazione standard: {float(std_precision):.6f}")
print(f"Recall media: {float(mean_recall_knn):.6f}, Deviazione standard: {float(std_recall):.6f}")
print(f"F1-Score medio: {float(mean_f1_knn):.6f}, Deviazione standard: {float(std_f1):.6f}")

print("\n\n")

#______________________________________________________________________________



#Grafico di confronto tra le medie delle misure dei modelli
a, graph_lr = plt.subplots(2, 2)
a.tight_layout(pad=4.0)

precision_data_dict = {"Logistic_Regression": mean_precision_logistic_regression, "Decision_Tree": mean_precision_decision_tree, "K-Nearest-Neighbor": mean_precision_knn}

recall_data_dict = {"Logistic_Regression": mean_recall_logistic_regression, "Decision_Tree": mean_recall_decision_tree, "K-Nearest-Neighbor": mean_recall_knn}

f1_data_dict = {"Logistic_Regression": mean_f1_logistic_regression, "Decision_Tree": mean_f1_decision_tree, "K-Nearest-Neighbor": mean_f1_knn}

accurancy_data_dict = {"Logistic_Regression": mean_accuracy_logistic_regression, "Decision_Tree": mean_accuracy_decision_tree, "K-Nearest-Neighbor": mean_accuracy_knn}

models_names = list(precision_data_dict.keys())
models_precision_data = list(precision_data_dict.values())
models_recall_data = list(recall_data_dict.values())
models_f1_data = list(f1_data_dict.values())
models_accurancy_data = list(accurancy_data_dict.values())

graph_lr[0, 0].bar(models_names, models_precision_data, color="red")
graph_lr[0, 0].set_title("Mean Precision")

graph_lr[0, 1].bar(models_names, models_recall_data, color="green")
graph_lr[0, 1].set_title("Mean Recall")

graph_lr[1, 0].bar(models_names, models_f1_data, color="purple")
graph_lr[1, 0].set_title("Mean F1-precision")

graph_lr[1, 1].bar(models_names, models_accurancy_data, color="blue")
graph_lr[1, 1].set_title("Mean Accurancy")

plt.show()

#Grafico di confronto tra le deviazioni standard delle misure dei modelli
a, graph_lr = plt.subplots(2, 2)
a.tight_layout(pad=4.0)

precision_data_dict = {"Logistic_Regression": std_list[0][2], "Decision_Tree": std_list[1][2], "K-Nearest-Neighbor": std_list[2][2]}

recall_data_dict = {"Logistic_Regression": std_list[0][3], "Decision_Tree": std_list[1][3], "K-Nearest-Neighbor": std_list[2][3]}

f1_data_dict = {"Logistic_Regression": std_list[0][1], "Decision_Tree": std_list[1][1], "K-Nearest-Neighbor": std_list[2][1]}

accurancy_data_dict = {"Logistic_Regression": std_list[0][0], "Decision_Tree": std_list[1][0], "K-Nearest-Neighbor": std_list[2][0]}

models_names = list(precision_data_dict.keys())
models_precision_data = list(precision_data_dict.values())
models_recall_data = list(recall_data_dict.values())
models_f1_data = list(f1_data_dict.values())
models_accurancy_data = list(accurancy_data_dict.values())

graph_lr[0, 0].bar(models_names, models_precision_data, color="red")
graph_lr[0, 0].set_title("Standard Deviation Precision")

graph_lr[0, 1].bar(models_names, models_recall_data, color="green")
graph_lr[0, 1].set_title("Standard Deviation Recall")

graph_lr[1, 0].bar(models_names, models_f1_data, color="purple")
graph_lr[1, 0].set_title("Standard Deviation F1-precision")

graph_lr[1, 1].bar(models_names, models_accurancy_data, color="blue")
graph_lr[1, 1].set_title("Standard Deviation Accurancy")

plt.show()