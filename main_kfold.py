import numpy as np
from sklearn.model_selection import *
from confronto_modello import *
from modello import *
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import randint


data = diabetes_data()
std_list=[]

#Setup della tabella
X = data.get_data().drop('diabetes', axis='columns').values
Y = data.get_data()['diabetes'].values


# Imposta la griglia di iperparametri
param_grid = {
    'C': [0.01, 0.1, 1, 10],           # Diversi valori per il parametro di regolarizzazione
    'penalty': ['l1', 'l2'],                # Proviamo sia L1 che L2 (tipo regolarizazione)
    'solver': ['liblinear', 'saga'],        # Solver che supportano L1 e L2 (Algoritmo di ottimizazione)
}

# Dichiara il modello di regressione logistica
logistic_model = LogisticRegression(max_iter=5000)

grid_search = GridSearchCV(estimator=logistic_model, param_grid=param_grid, 
                           cv=5, n_jobs=-1, scoring='accuracy', verbose=1)


# Esegui la Grid Search sull'intero dataset
grid_search.fit(X, Y)


# Migliori iperparametri trovati dalla Grid Search
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Migliori iperparametri trovati dalla Grid Search per Logistic Regression:\n {best_params}\n")


# Liste per memorizzare le metriche
logistic_regression_accuracy_scores = []
logistic_regression_precision_scores = []
logistic_regression_recall_scores = []
logistic_regression_f1_scores = []
logistic_regression_confusion_matrices = []

#Divido il dataset con la Kfold
kf_logistic = KFold(n_splits=5, shuffle=True, random_state=42)


nfold = 1 
print("Logistic Regression ottimizato con Grid Search")

# Ciclo su ogni fold
for train_index, test_index in kf_logistic.split(X):
    
    
    #model_1.predict()
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    # Standardizza i dati
    scaler = StandardScaler()
    x_train_scaled=scaler.fit_transform(X_train)
    x_test_scaled=scaler.fit_transform(X_test)

    # Allena il miglior modello trovato dalla Grid Search
    best_model.fit(x_train_scaled, y_train)

    # Fai le previsioni
    y_pred = best_model.predict(x_test_scaled)

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



decision_model = tree.DecisionTreeClassifier(max_depth=42)

# Definizione della distribuzione casuale di iperparametri
param_dist = {
    'max_depth': randint(3, 100),#(Profondità massima dell'albero)
    'min_samples_split': randint(2, 20),#(Minimo numero di campioni per effettuare uno split)
    'min_samples_leaf': randint(1, 10),#(Minimo numero di campioni in una foglia)
    'max_leaf_nodes': randint(10, 100),#(Numero massimo di nodi foglia)
    'criterion': ['gini', 'entropy'],#(Numero massimo di caratteristiche considerate per lo split)
}

# Definizione di Randomized Search con cross-validation
random_search = RandomizedSearchCV(estimator=decision_model, param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy', random_state=42)

# Esegui la Randomized Search
random_search.fit(X, Y)

# Migliori iperparametri trovati
best_params = random_search.best_params_
print("Migliori iperparametri trovati dalla Randomized Search per Decision Tree:\n", best_params)

# Ora possiamo usare i migliori iperparametri trovati nella K-Fold Validation
best_decision_model = DecisionTreeClassifier(**best_params)


# Liste per memorizzare le metriche
decision_tree_accuracy_scores = []
decision_tree_precision_scores = []
decision_tree_recall_scores = []
decision_tree_f1_scores = []
decision_tree_confusion_matrices = []

kf_decision = KFold(n_splits=5, shuffle=True, random_state=42)

nfold = 1 
print("\nDecision Tree ottimizzato con Randomized Search")

for train_index, test_index in kf_decision.split(X):

    #model_1.predict()
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    scaler = StandardScaler()
    x_train_scaled=scaler.fit_transform(X_train)
    x_test_scaled=scaler.fit_transform(X_test)

    # Allena il modello con i migliori iperparametri
    best_decision_model.fit(x_train_scaled, y_train)

    # Fai le previsioni
    y_pred = best_decision_model.predict(x_test_scaled)

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


# Definisci lo spazio degli iperparametri da esplorare con Random Search
param_dist = {
    'n_neighbors': np.arange(1, 10),
    'weights': ['uniform', 'distance'],
    'p': [1, 2],  # Parametro per la distanza Minkowski: 1 = distanza di Manhattan, 2 = distanza Euclidea
    'algorithm': ['auto','kd_tree']  # Algoritmi per la ricerca dei vicini
}

knn_model = KNeighborsClassifier()

# Randomized Search con 50 iterazioni, 5-fold cross validation
random_search = RandomizedSearchCV(knn_model, param_distributions=param_dist, n_iter=50, cv=5, random_state=42, n_jobs=-1)

# Fitting dei dati (con Randomized Search)
random_search.fit(X, Y)

# Migliori parametri trovati
print(f"Migliori iperparametri trovati con la Random Search per Knn:\n {random_search.best_params_}")

# Liste per memorizzare le metriche
knn_accuracy_scores = []
knn_precision_scores = []
knn_recall_scores = []
knn_f1_scores = []
knn_confusion_matrices = []

kf_knn = KFold(n_splits=5, shuffle=True, random_state=42)

# Usa il modello KNN con i migliori iperparametri trovati
best_knn_model = random_search.best_estimator_


nfold = 1 
print("\nKnn ottimizzato con Randomized Search")

for train_index, test_index in kf_knn.split(X):

    #model_1.predict()
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    scaler = StandardScaler()
    x_train_scaled=scaler.fit_transform(X_train)
    x_test_scaled=scaler.fit_transform(X_test)

   # Allena il modello con i migliori iperparametri
    best_knn_model.fit(x_train_scaled, y_train)

    # Fai le previsioni
    y_pred = best_knn_model.predict(x_test_scaled)

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