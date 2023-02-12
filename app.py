import numpy as np
import pandas as pd
import seaborn as sns
import pyswarms as ps
import matplotlib.pyplot as plt

from sklearn import metrics
from flask import Flask, render_template
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from pyswarms.utils.functions import single_obj as fx

app = Flask(__name__)
app.secret_key = '171401059'

@app.route('/')
def utama():
    return render_template('index.html')

@app.route('/beranda')
def beranda():
    return render_template('index.html')

@app.route('/tentang')
def tentang():
    return render_template('tentang.html')

@app.route('/pilih_dataset')
def pilih_dataset():
    return render_template('pilih_dataset.html')

@app.route('/Adolescent')
def Adolescent():
    return render_template('Adolescent.html')

@app.route('/Adolescent_NB.html')
def Adolescent_NB():
    # Dataset Adolescent 
    dataset = pd.read_excel("D:\Py\skripsi\static\Adolescent_.xlsx")
    jum = len(dataset)
    dataset = dataset.dropna(axis = 0, how ='any')
    juma = len(dataset)
    X = np.array(dataset.drop(['Class/ASD'], 1))
    y = np.array(dataset['Class/ASD'])

    # Metode Naive Bayes 
    metode = GaussianNB()

    # Naive Bayes (NB)
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=.70, random_state=42)
    modelNB = metode.fit(X_train, y_train)
    hasilNB_Xtest = modelNB.predict(X_test)

    # Confusion Matrix (NB)
    cmnb_Adolescent = metrics.confusion_matrix(y_test, hasilNB_Xtest)
    print(classification_report(y_test, hasilNB_Xtest))
    # Confusion Matrix (NB)
    cm_Adolescent = metrics.confusion_matrix(y_test, hasilNB_Xtest)
    plt.figure(1)
    plot_confusion_matrix(metode, X_test, y_test, cmap=plt.cm.Pastel1)
    plt.savefig('D:\Py\skripsi\static\Hasil Confusion Matrix NB Tanpa PSO - Adolescent.png')
    plt.close()

    # Nilai Confusion Matrix (NB)
    akurasi_nb = int(np.around(metrics.accuracy_score(y_test, hasilNB_Xtest)*100))
    precision_nb = int(np.around(metrics.precision_score(y_test, hasilNB_Xtest)*100))
    recall_nb = int(np.around(metrics.recall_score(y_test, hasilNB_Xtest)*100))
    fmeasure_nb = int(np.around(metrics.f1_score(y_test, hasilNB_Xtest)*100))

    #AUC Score (NB)
    nb_probs = metode.predict_proba(X_test)
    random_probs = [0 for _ in range(len(y_test))]
    nb_probs = nb_probs[:,1] 
    #Prediction probability berisi '0'
    nb_auc = metrics.roc_auc_score(y_test, nb_probs)
    random_auc = roc_auc_score(y_test, random_probs)
    print('Naive Bayes : %.3f' % (nb_auc))
    print('Random Prediction : AUCROC = %.3f' % (random_auc))
    
    #Hitung ROC Curves (NB)
    nb_tpr, nb_fpr, _ = metrics.roc_curve(y_test, nb_probs)
    random_fpr, random_tpr, _ = roc_curve(y_test, random_probs)
    
    #ROC Curve (NB)
    plt.figure(2)
    plt.plot(nb_tpr, nb_fpr, marker='.', color='salmon', label ='Naive Bayes : %.3f' % nb_auc)
    plt.plot(random_fpr, random_tpr, color='crimson', marker='.', label='Random : (AUCROC = %.3f)' % random_auc)
    plt.title('ROC Curve Naive Bayes')
    plt.ylabel('True Positive Rate', size = 20)
    plt.xlabel('False Positive Rate', size = 20)
    plt.legend()
    plt.savefig('D:\Py\skripsi\static\ROC Curve Adolescent - NB.png')
    plt.close()

    # Particle Swarm Optimization (Pergerakan partikel (atribut))
    def f_per_particle(m, alpha):
        total_features = X.shape[1]
        if np.count_nonzero(m) == 0:
            X_subset = X
        else:
            X_subset = X[:,m==1]
        X_train, X_test, y_train, y_test = train_test_split(X_subset,y,train_size=.70, random_state=42)
        metode.fit(X_train, y_train)
        P = (metode.predict(X_test) == y_test).mean()
        j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
        return j
    
    def f(x, alpha=0.88):
        n_particles = x.shape[0]
        j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
        return np.array(j)

    #Initialize swarm, arbitrary
    options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 1  , 'p':2}
    dimensions = X.shape[1] # banyak fitur
    optimizer = ps.discrete.BinaryPSO(n_particles=2, dimensions=dimensions,options=options) 
    #Optimization
    cost, pos = optimizer.optimize(fx.sphere, iters=20)
    
    # Naive Bayes (NB - PSO)
    fiturSeleksiX = X[:,pos==1]
    X_train, X_test, y_train, y_test = train_test_split(fiturSeleksiX,y,train_size=.70, random_state=42)
    modelNBPSO = metode.fit(X_train, y_train)
    hasilNBPSO_Xtest = modelNBPSO.predict(X_test) # Menentukan hasil prediksi y_pred
    
    # Confusion Matrix (NB - PSO)
    cm_Adolescent = metrics.confusion_matrix(y_test, hasilNBPSO_Xtest)
    plt.figure(1)
    plot_confusion_matrix(metode, X_test, y_test, cmap=plt.cm.Pastel1)
    plt.savefig('D:\Py\skripsi\static\Hasil Confusion Matrix NB - Adolescent.png')
    plt.close()

    #AUC Score (NB - PSO)
    nb_probs = metode.predict_proba(X_test)
    random_probs = [0 for _ in range(len(y_test))]
    nb_probs = nb_probs[:,1] 
    #Prediction probability berisi '0'
    nb_auc = metrics.roc_auc_score(y_test, nb_probs)
    random_auc = roc_auc_score(y_test, random_probs)
    print('Naive Bayes - PSO : %.3f' % (nb_auc))
    print('Random Prediction : AUCROC = %.3f' % (random_auc))
    
    #Hitung ROC Curves (NB - PSO)
    nb_tpr, nb_fpr, _ = metrics.roc_curve(y_test, nb_probs)
    random_fpr, random_tpr, _ = roc_curve(y_test, random_probs)
    
    #ROC Curve (NB - PSO)
    plt.figure(2)
    plt.plot(nb_tpr, nb_fpr, marker='.', color='salmon', label ='Naive Bayes : %.3f' % nb_auc)
    plt.plot(random_fpr, random_tpr, color='crimson', marker='.', label='Random : (AUCROC = %.3f)' % random_auc)
    plt.title('ROC Curve Naive Bayes - PSO')
    plt.ylabel('True Positive Rate', size = 20)
    plt.xlabel('False Positive Rate', size = 20)
    plt.legend()
    plt.savefig('D:\Py\skripsi\static\ROC Curve Adolescent - NBPSO.png')
    plt.close()

    # Nilai Confusion Matrix (NB - PSO)
    akurasi = int(np.around(metrics.accuracy_score(y_test, hasilNBPSO_Xtest)*100))
    precision = int(np.around(metrics.precision_score(y_test, hasilNBPSO_Xtest)*100))
    recall = int(np.around(metrics.recall_score(y_test, hasilNBPSO_Xtest)*100))
    fmeasure = int(np.around(metrics.f1_score(y_test, hasilNBPSO_Xtest)*100))

    # Menampilkan Atribut Terpilih (NB - PSO)
    atribut = ["id","A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score","age","gender","ethnicity","jundice","austim","contry_of_res","used_app_before","age_desc","relation","Class/ASD"]
    atributArr = np.array(atribut)
    atributSel = atributArr[pos==1]
    atributTerpilih = str(sum((pos == 1)*1))
    atributTotal = str(len(pos))
    dataTraining = str(len(X_train))
    dataTesting = str(len(X_test))
    dataMissing = jum - juma

    print (classification_report(y_test, hasilNBPSO_Xtest))
    print('Hasil Confusions Matrix NB : \n', cmnb_Adolescent)
    print('Akurasi NB : ', akurasi_nb)
    print('Precision NB : ', precision_nb)
    print('Recall NB : ', recall_nb)
    print('F-Measure NB : ', fmeasure_nb)
    print('Fitur Terseleksi : \n' + str(sum((pos == 1)*1)) + '/' + str(len(pos)))
    print('Hasil Confusion Matrix NB - PSO: \n', cm_Adolescent)
    print('Akurasi NB - PSO : ', akurasi)
    print('Precision NB - PSO : ', precision)
    print('Recall NB - PSO : ', recall)
    print('F-Measure NB - PSO : ', fmeasure)
    print('Jumlah Dataset : ', jum)
    print('Jumlah Dataset setelah penanganan missing value: ', juma)
    print('Jumlah Atribut Yang Terpilih :', atributTerpilih)
    print('Atribut Yang Terpilih :', atributSel)

    return render_template('Adolescent_NB.html', atributSel=atributSel, fmeasure=fmeasure, recall=recall, precision=precision, akurasi=akurasi,jum=jum,juma=juma, atributTerpilih=atributTerpilih, atributTotal=atributTotal, dataTraining=dataTraining, dataTesting=dataTesting, dataMissing=dataMissing)

@app.route('/Adolescent_NN.html')
def Adolescent_NN():
    # Dataset Adolescent 
    dataset = pd.read_excel("D:\Py\skripsi\static\Adolescent_.xlsx")
    jum = len(dataset)
    dataset = dataset.dropna(axis = 0, how ='any')
    juma = len(dataset)
    X = np.array(dataset.drop(['Class/ASD'], 1))
    y = np.array(dataset['Class/ASD'])

    # Metode Neural Network 
    metode = MLPClassifier(hidden_layer_sizes=(6,), random_state=42)

    # Naive Bayes (NN)
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=.70, random_state=42)
    modelNN = metode.fit(X_train, y_train)
    hasilNN_Xtest = modelNN.predict(X_test)

    # Confusion Matrix (NN)
    cmnn_Adolescent = metrics.confusion_matrix(y_test, hasilNN_Xtest)
    print(classification_report(y_test, hasilNN_Xtest))
    # Confusion Matrix (NN)
    cm_Adolescent = metrics.confusion_matrix(y_test, hasilNN_Xtest)
    plt.figure(1)
    plot_confusion_matrix(metode, X_test, y_test, cmap=plt.cm.Pastel1)
    plt.savefig('D:\Py\skripsi\static\Hasil Confusion Matrix NN Tanpa PSO - Adolescent.png')
    plt.close()

    # Nilai Confusion Matrix (NN)
    akurasi_nn = int(np.around(metrics.accuracy_score(y_test, hasilNN_Xtest)*100))
    precision_nn = int(np.around(metrics.precision_score(y_test, hasilNN_Xtest)*100))
    recall_nn = int(np.around(metrics.recall_score(y_test, hasilNN_Xtest)*100))
    fmeasure_nn = int(np.around(metrics.f1_score(y_test, hasilNN_Xtest)*100))

    #AUC Score (NN)
    nn_probs = metode.predict_proba(X_test)
    random_probs = [0 for _ in range(len(y_test))]
    nn_probs = nn_probs[:,1] 
    #Prediction probability berisi '0'
    nn_auc = metrics.roc_auc_score(y_test, nn_probs)
    random_auc = roc_auc_score(y_test, random_probs)
    print('Neural Network: %.3f' % (nn_auc))
    print('Random Prediction : AUCROC = %.3f' % (random_auc))
    
    #Hitung ROC Curves (NN)
    nn_tpr, nn_fpr, _ = metrics.roc_curve(y_test, nn_probs)
    random_fpr, random_tpr, _ = roc_curve(y_test, random_probs)
    
    #ROC Curve (NN)
    plt.figure(2)
    plt.plot(nn_tpr, nn_fpr, marker='.', color='salmon', label ='Neural Network : %.3f' % nn_auc)
    plt.plot(random_fpr, random_tpr, color='crimson', marker='.', label='Random : (AUCROC = %.3f)' % random_auc)
    plt.title('ROC Curve Neural Network')
    plt.ylabel('True Positive Rate', size = 20)
    plt.xlabel('False Positive Rate', size = 20)
    plt.legend()
    plt.savefig('D:\Py\skripsi\static\ROC Curve Adolescent - NN.png')
    plt.close()

    # Particle Swarm Optimization
    def f_per_particle(m, alpha):
        total_features = X.shape[1]
        if np.count_nonzero(m) == 0:
            X_subset = X
        else:
            X_subset = X[:,m==1]
        X_train, X_test, y_train, y_test = train_test_split(X_subset,y,train_size=.70, random_state=42)
        metode.fit(X_train, y_train)
        P = (metode.predict(X_test) == y_test).mean()
        j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
        return j
    
    def f(x, alpha=0.88):
        n_particles = x.shape[0]
        j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
        return np.array(j)

    #Initialize swarm, arbitrary
    options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 1  , 'p':2}
    dimensions = X.shape[1] # dimensions should be the number of features
    optimizer = ps.discrete.BinaryPSO(n_particles=2, dimensions=dimensions,options=options) 
    #Optimization
    cost, pos = optimizer.optimize(fx.sphere, iters=20)
    
    # Naive Bayes (NN - PSO)
    fiturSeleksiX = X[:,pos==1]
    X_train, X_test, y_train, y_test = train_test_split(fiturSeleksiX,y,train_size=.70, random_state=42)
    modelNNPSO = metode.fit(X_train, y_train)
    hasilNNPSO_Xtest = modelNNPSO.predict(X_test) # Menentukan hasil prediksi y_pred
    
    # Confusion Matrix (NN - PSO)
    cm_Adolescent = metrics.confusion_matrix(y_test, hasilNNPSO_Xtest)
    plt.figure(1)
    plot_confusion_matrix(metode, X_test, y_test, cmap=plt.cm.Pastel1)
    plt.savefig('D:\Py\skripsi\static\Hasil Confusion Matrix NN - Adolescent.png')
    plt.close()

    #AUC Score (NN - PSO)
    nn_probs = metode.predict_proba(X_test)
    random_probs = [0 for _ in range(len(y_test))]
    nn_probs = nn_probs[:,1] 
    #Prediction probability berisi '0'
    nn_auc = metrics.roc_auc_score(y_test, nn_probs)
    random_auc = roc_auc_score(y_test, random_probs)
    print('Neural Network - PSO : %.3f' % (nn_auc))
    print('Random Prediction : AUCROC = %.3f' % (random_auc))
    
    #Hitung ROC Curves (NN - PSO)
    nn_tpr, nn_fpr, _ = metrics.roc_curve(y_test, nn_probs)
    random_fpr, random_tpr, _ = roc_curve(y_test, random_probs)
    
    #ROC Curve (NN - PSO)
    plt.figure(2)
    plt.plot(nn_tpr, nn_fpr, marker='.', color='salmon', label ='Neural Network - PSO : %.3f' % nn_auc)
    plt.plot(random_fpr, random_tpr, color='crimson', marker='.', label='Random : (AUCROC = %.3f)' % random_auc)
    plt.title('ROC Curve Neural Network - PSO')
    plt.ylabel('True Positive Rate', size = 20)
    plt.xlabel('False Positive Rate', size = 20)
    plt.legend()
    plt.savefig('D:\Py\skripsi\static\ROC Curve Adolescent - NNPSO.png')
    plt.close()

    # Nilai Confusion Matrix (NN - PSO)
    akurasi = int(np.around(metrics.accuracy_score(y_test, hasilNNPSO_Xtest)*100))
    precision = int(np.around(metrics.precision_score(y_test, hasilNNPSO_Xtest)*100))
    recall = int(np.around(metrics.recall_score(y_test, hasilNNPSO_Xtest)*100))
    fmeasure = int(np.around(metrics.f1_score(y_test, hasilNNPSO_Xtest)*100))

    # Menampilkan Atribut Terpilih (NN - PSO)
    atribut = ["id","A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score","age","gender","ethnicity","jundice","austim","contry_of_res","used_app_before","age_desc","relation","Class/ASD"]
    atributArr = np.array(atribut)
    atributSel = atributArr[pos==1]
    atributTerpilih = str(sum((pos == 1)*1))
    atributTotal = str(len(pos))
    dataTraining = str(len(X_train))
    dataTesting = str(len(X_test))
    dataMissing = jum - juma

    print (classification_report(y_test, hasilNNPSO_Xtest))
    print('Hasil Confusions Matrix NN : \n', cmnn_Adolescent)
    print('Akurasi NN : ', akurasi_nn)
    print('Precision NN: ', precision_nn)
    print('Recall NN: ', recall_nn)
    print('F-Measure NN: ', fmeasure_nn)
    print('Fitur Terseleksi : \n' + str(sum((pos == 1)*1)) + '/' + str(len(pos)))
    print('Hasil Confusion Matrix NN - PSO: \n', cm_Adolescent)
    print('Akurasi NN - PSO : ', akurasi)
    print('Precision NN - PSO : ', precision)
    print('Recall NN - PSO : ', recall)
    print('F-Measure NN - PSO : ', fmeasure)
    print('Jumlah Dataset : ', jum)
    print('Jumlah Dataset setelah penanganan missing value: ', juma)
    print('Atribut Yang Terpilih :', atributTerpilih)
    print('Atribut Yang Terpilih :', atributSel)


    return render_template('Adolescent_NN.html', atributSel=atributSel, fmeasure=fmeasure, recall=recall, precision=precision, akurasi=akurasi,jum=jum,juma=juma, atributTerpilih=atributTerpilih, atributTotal=atributTotal, dataTraining=dataTraining, dataTesting=dataTesting, dataMissing=dataMissing)

@app.route('/Adult')
def Adult():
    return render_template('Adult.html')

@app.route('/Adult_NB.html')
def Adult_NB():
    # Dataset Adult
    dataset = pd.read_excel("D:\Py\skripsi\static\Adult_.xlsx")
    jum = len(dataset)
    dataset = dataset.dropna(axis = 0, how ='any')
    juma = len(dataset)
    X = np.array(dataset.drop(['Class/ASD'], 1))
    y = np.array(dataset['Class/ASD'])

    # Metode Naive Bayes 
    metode = GaussianNB()

    # Naive Bayes (NB)
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=.70, random_state=42)
    modelNB = metode.fit(X_train, y_train)
    hasilNB_Xtest = modelNB.predict(X_test)

    # Confusion Matrix (NB)
    cmnb_Adolescent = metrics.confusion_matrix(y_test, hasilNB_Xtest)
    print(classification_report(y_test, hasilNB_Xtest))
    # Confusion Matrix (NB)
    cm_Adolescent = metrics.confusion_matrix(y_test, hasilNB_Xtest)
    plt.figure(1)
    plot_confusion_matrix(metode, X_test, y_test, cmap=plt.cm.Pastel1)
    plt.savefig('D:\Py\skripsi\static\Hasil Confusion Matrix NB Tanpa PSO - Adult.png')
    plt.close()

    # Nilai Confusion Matrix (NB)
    akurasi_nb = int(np.around(metrics.accuracy_score(y_test, hasilNB_Xtest)*100))
    precision_nb = int(np.around(metrics.precision_score(y_test, hasilNB_Xtest)*100))
    recall_nb = int(np.around(metrics.recall_score(y_test, hasilNB_Xtest)*100))
    fmeasure_nb = int(np.around(metrics.f1_score(y_test, hasilNB_Xtest)*100))

    #AUC Score (NB)
    nb_probs = metode.predict_proba(X_test)
    random_probs = [0 for _ in range(len(y_test))]
    nb_probs = nb_probs[:,1] 
    #Prediction probability berisi '0'
    nb_auc = metrics.roc_auc_score(y_test, nb_probs)
    random_auc = roc_auc_score(y_test, random_probs)
    print('Naive Bayes : %.3f' % (nb_auc))
    print('Random Prediction : AUCROC = %.3f' % (random_auc))
    
    #Hitung ROC Curves (NB)
    nb_tpr, nb_fpr, _ = metrics.roc_curve(y_test, nb_probs)
    random_fpr, random_tpr, _ = roc_curve(y_test, random_probs)
    
    #ROC Curve (NB)
    plt.figure(2)
    plt.plot(nb_tpr, nb_fpr, marker='.', color='salmon', label ='Naive Bayes : %.3f' % nb_auc)
    plt.plot(random_fpr, random_tpr, color='crimson', marker='.', label='Random : (AUCROC = %.3f)' % random_auc)
    plt.title('ROC Curve Naive Bayes')
    plt.ylabel('True Positive Rate', size = 20)
    plt.xlabel('False Positive Rate', size = 20)
    plt.legend()
    plt.savefig('D:\Py\skripsi\static\ROC Curve Adult - NB.png')
    plt.close()

    # Particle Swarm Optimization
    def f_per_particle(m, alpha):
        total_features = X.shape[1]
        if np.count_nonzero(m) == 0:
            X_subset = X
        else:
            X_subset = X[:,m==1]
        X_train, X_test, y_train, y_test = train_test_split(X_subset,y,train_size=.70, random_state=42)
        metode.fit(X_train, y_train)
        P = (metode.predict(X_test) == y_test).mean()
        j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
        return j
    
    def f(x, alpha=0.88):
        n_particles = x.shape[0]
        j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
        return np.array(j)

    #Initialize swarm, arbitrary
    options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 1  , 'p':2}
    dimensions = X.shape[1] # dimensions should be the number of features
    optimizer = ps.discrete.BinaryPSO(n_particles=2, dimensions=dimensions,options=options) 
    #Optimization
    cost, pos = optimizer.optimize(fx.sphere, iters=20)
    
    # Naive Bayes (NB - PSO)
    fiturSeleksiX = X[:,pos==1]
    X_train, X_test, y_train, y_test = train_test_split(fiturSeleksiX,y,train_size=.70, random_state=42)
    modelNBPSO = metode.fit(X_train, y_train)
    hasilNBPSO_Xtest = modelNBPSO.predict(X_test) # Menentukan hasil prediksi y_pred
    
    # Confusion Matrix (NB - PSO)
    cm_Adolescent = metrics.confusion_matrix(y_test, hasilNBPSO_Xtest)
    plt.figure(1)
    plot_confusion_matrix(metode, X_test, y_test, cmap=plt.cm.Pastel1)
    plt.savefig('D:\Py\skripsi\static\Hasil Confusion Matrix NB - Adult.png')
    plt.close()

    #AUC Score (NB - PSO)
    nb_probs = metode.predict_proba(X_test)
    random_probs = [0 for _ in range(len(y_test))]
    nb_probs = nb_probs[:,1] 
    #Prediction probability berisi '0'
    nb_auc = metrics.roc_auc_score(y_test, nb_probs)
    random_auc = roc_auc_score(y_test, random_probs)
    print('Naive Bayes - PSO : %.3f' % (nb_auc))
    print('Random Prediction : AUCROC = %.3f' % (random_auc))
    
    #Hitung ROC Curves (NB - PSO)
    nb_tpr, nb_fpr, _ = metrics.roc_curve(y_test, nb_probs)
    random_fpr, random_tpr, _ = roc_curve(y_test, random_probs)
    
    #ROC Curve (NB - PSO)
    plt.figure(2)
    plt.plot(nb_tpr, nb_fpr, marker='.', color='salmon', label ='Naive Bayes : %.3f' % nb_auc)
    plt.plot(random_fpr, random_tpr, color='crimson', marker='.', label='Random : (AUCROC = %.3f)' % random_auc)
    plt.title('ROC Curve Naive Bayes - PSO')
    plt.ylabel('True Positive Rate', size = 20)
    plt.xlabel('False Positive Rate', size = 20)
    plt.legend()
    plt.savefig('D:\Py\skripsi\static\ROC Curve Adult - NBPSO.png')
    plt.close()

    # Nilai Confusion Matrix (NB - PSO)
    akurasi = int(np.around(metrics.accuracy_score(y_test, hasilNBPSO_Xtest)*100))
    precision = int(np.around(metrics.precision_score(y_test, hasilNBPSO_Xtest)*100))
    recall = int(np.around(metrics.recall_score(y_test, hasilNBPSO_Xtest)*100))
    fmeasure = int(np.around(metrics.f1_score(y_test, hasilNBPSO_Xtest)*100))

    # Menampilkan Atribut Terpilih (NB - PSO)
    atribut = ["id","A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score","age","gender","ethnicity","jundice","austim","contry_of_res","used_app_before","age_desc","relation","Class/ASD"]
    atributArr = np.array(atribut)
    atributSel = atributArr[pos==1]
    atributTerpilih = str(sum((pos == 1)*1))
    atributTotal = str(len(pos))
    dataTraining = str(len(X_train))
    dataTesting = str(len(X_test))
    dataMissing = jum - juma

    print (classification_report(y_test, hasilNBPSO_Xtest))
    print('Hasil Confusions Matrix NB : \n', cmnb_Adolescent)
    print('Akurasi NB : ', akurasi_nb)
    print('Precision NB : ', precision_nb)
    print('Recall NB : ', recall_nb)
    print('F-Measure NB : ', fmeasure_nb)
    print('Fitur Terseleksi : \n' + str(sum((pos == 1)*1)) + '/' + str(len(pos)))
    print('Hasil Confusion Matrix NB - PSO: \n', cm_Adolescent)
    print('Akurasi NB - PSO : ', akurasi)
    print('Precision NB - PSO : ', precision)
    print('Recall NB - PSO : ', recall)
    print('F-Measure NB - PSO : ', fmeasure)
    print('Jumlah Dataset : ', jum)
    print('Jumlah Dataset setelah penanganan missing value: ', juma)
    print('Atribut Yang Terpilih :', atributTerpilih)
    print('Atribut Yang Terpilih :', atributSel)

    return render_template('Adult_NB.html', atributSel=atributSel, fmeasure=fmeasure, recall=recall, precision=precision, akurasi=akurasi,jum=jum,juma=juma, atributTerpilih=atributTerpilih, atributTotal=atributTotal, dataTraining=dataTraining, dataTesting=dataTesting, dataMissing=dataMissing)

@app.route('/Adult_NN')
def Adult_NN():
    # Dataset Adult
    dataset = pd.read_excel("D:\Py\skripsi\static\Adult_.xlsx")
    jum = len(dataset)
    dataset = dataset.dropna(axis = 0, how ='any')
    juma = len(dataset)
    X = np.array(dataset.drop(['Class/ASD'], 1))
    y = np.array(dataset['Class/ASD'])

    # Metode Neural Network 
    metode = MLPClassifier(hidden_layer_sizes=(6,), random_state=42)

    # Neural Network (NN)
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=.70, random_state=42)
    modelNN = metode.fit(X_train, y_train)
    hasilNN_Xtest = modelNN.predict(X_test)

    # Confusion Matrix (NN)
    cmnn_Adolescent = metrics.confusion_matrix(y_test, hasilNN_Xtest)
    print(classification_report(y_test, hasilNN_Xtest))
    # Confusion Matrix (NN)
    cm_Adolescent = metrics.confusion_matrix(y_test, hasilNN_Xtest)
    plt.figure(1)
    plot_confusion_matrix(metode, X_test, y_test, cmap=plt.cm.Pastel1)
    plt.savefig('D:\Py\skripsi\static\Hasil Confusion Matrix NN Tanpa PSO - Adolescent.png')
    plt.close()

    # Nilai Confusion Matrix (NN)
    akurasi_nn = int(np.around(metrics.accuracy_score(y_test, hasilNN_Xtest)*100))
    precision_nn = int(np.around(metrics.precision_score(y_test, hasilNN_Xtest)*100))
    recall_nn = int(np.around(metrics.recall_score(y_test, hasilNN_Xtest)*100))
    fmeasure_nn = int(np.around(metrics.f1_score(y_test, hasilNN_Xtest)*100))

    #AUC Score (NN)
    nn_probs = metode.predict_proba(X_test)
    random_probs = [0 for _ in range(len(y_test))]
    nn_probs = nn_probs[:,1] 
    #Prediction probability berisi '0'
    nn_auc = metrics.roc_auc_score(y_test, nn_probs)
    random_auc = roc_auc_score(y_test, random_probs)
    print('Neural Network: %.3f' % (nn_auc))
    print('Random Prediction : AUCROC = %.3f' % (random_auc))
    
    #Hitung ROC Curves (NN)
    nn_tpr, nn_fpr, _ = metrics.roc_curve(y_test, nn_probs)
    random_fpr, random_tpr, _ = roc_curve(y_test, random_probs)
    
    #ROC Curve (NN)
    plt.figure(2)
    plt.plot(nn_tpr, nn_fpr, marker='.', color='salmon', label ='Neural Network : %.3f' % nn_auc)
    plt.plot(random_fpr, random_tpr, color='crimson', marker='.', label='Random : (AUCROC = %.3f)' % random_auc)
    plt.title('ROC Curve Neural Network')
    plt.ylabel('True Positive Rate', size = 20)
    plt.xlabel('False Positive Rate', size = 20)
    plt.legend()
    plt.savefig('D:\Py\skripsi\static\ROC Curve Adult - NN.png')
    plt.close()

    # Particle Swarm Optimization
    def f_per_particle(m, alpha):
        total_features = X.shape[1]
        if np.count_nonzero(m) == 0:
            X_subset = X
        else:
            X_subset = X[:,m==1]
        X_train, X_test, y_train, y_test = train_test_split(X_subset,y,train_size=.70, random_state=42)
        metode.fit(X_train, y_train)
        P = (metode.predict(X_test) == y_test).mean()
        j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
        return j
    
    def f(x, alpha=0.88):
        n_particles = x.shape[0]
        j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
        return np.array(j)

    #Initialize swarm, arbitrary
    options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 1  , 'p':2}
    dimensions = X.shape[1] # dimensions should be the number of features
    optimizer = ps.discrete.BinaryPSO(n_particles=2, dimensions=dimensions,options=options) 
    #Optimization
    cost, pos = optimizer.optimize(fx.sphere, iters=20)
    
    # Naive Bayes (NN - PSO)
    fiturSeleksiX = X[:,pos==1]
    X_train, X_test, y_train, y_test = train_test_split(fiturSeleksiX,y,train_size=.70, random_state=42)
    modelNNPSO = metode.fit(X_train, y_train)
    hasilNNPSO_Xtest = modelNNPSO.predict(X_test) # Menentukan hasil prediksi y_pred
    
    # Confusion Matrix (NN - PSO)
    cm_Adolescent = metrics.confusion_matrix(y_test, hasilNNPSO_Xtest)
    plt.figure(1)
    plot_confusion_matrix(metode, X_test, y_test, cmap=plt.cm.Pastel1)
    plt.savefig('D:\Py\skripsi\static\Hasil Confusion Matrix NN - Adult.png')
    plt.close()

    #AUC Score (NN - PSO)
    nn_probs = metode.predict_proba(X_test)
    random_probs = [0 for _ in range(len(y_test))]
    nn_probs = nn_probs[:,1] 
    #Prediction probability berisi '0'
    nn_auc = metrics.roc_auc_score(y_test, nn_probs)
    random_auc = roc_auc_score(y_test, random_probs)
    print('Neural Network - PSO : %.3f' % (nn_auc))
    print('Random Prediction : AUCROC = %.3f' % (random_auc))
    
    #Hitung ROC Curves (NN - PSO)
    nn_tpr, nn_fpr, _ = metrics.roc_curve(y_test, nn_probs)
    random_fpr, random_tpr, _ = roc_curve(y_test, random_probs)
    
    #ROC Curve (NN - PSO)
    plt.figure(2)
    plt.plot(nn_tpr, nn_fpr, marker='.', color='salmon', label ='Neural Network - PSO : %.3f' % nn_auc)
    plt.plot(random_fpr, random_tpr, color='crimson', marker='.', label='Random : (AUCROC = %.3f)' % random_auc)
    plt.title('ROC Curve Neural Network - PSO')
    plt.ylabel('True Positive Rate', size = 20)
    plt.xlabel('False Positive Rate', size = 20)
    plt.legend()
    plt.savefig('D:\Py\skripsi\static\ROC Curve Adult - NNPSO.png')
    plt.close()

    # Nilai Confusion Matrix (NN - PSO)
    akurasi = int(np.around(metrics.accuracy_score(y_test, hasilNNPSO_Xtest)*100))
    precision = int(np.around(metrics.precision_score(y_test, hasilNNPSO_Xtest)*100))
    recall = int(np.around(metrics.recall_score(y_test, hasilNNPSO_Xtest)*100))
    fmeasure = int(np.around(metrics.f1_score(y_test, hasilNNPSO_Xtest)*100))

    # Menampilkan Atribut Terpilih (NN - PSO)
    atribut = ["id","A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score","age","gender","ethnicity","jundice","austim","contry_of_res","used_app_before","age_desc","relation","Class/ASD"]
    atributArr = np.array(atribut)
    atributSel = atributArr[pos==1]
    atributTerpilih = str(sum((pos == 1)*1))
    atributTotal = str(len(pos))
    dataTraining = str(len(X_train))
    dataTesting = str(len(X_test))
    dataMissing = jum - juma

    print (classification_report(y_test, hasilNNPSO_Xtest))
    print('Hasil Confusions Matrix NN : \n', cmnn_Adolescent)
    print('Akurasi NN : ', akurasi_nn)
    print('Precision NN: ', precision_nn)
    print('Recall NN: ', recall_nn)
    print('F-Measure NN: ', fmeasure_nn)
    print('Fitur Terseleksi : \n' + str(sum((pos == 1)*1)) + '/' + str(len(pos)))
    print('Hasil Confusion Matrix NN - PSO: \n', cm_Adolescent)
    print('Akurasi NN - PSO : ', akurasi)
    print('Precision NN - PSO : ', precision)
    print('Recall NN - PSO : ', recall)
    print('F-Measure NN - PSO : ', fmeasure)
    print('Jumlah Dataset : ', jum)
    print('Jumlah Dataset setelah penanganan missing value: ', juma)
    print('Atribut Yang Terpilih :', atributTerpilih)
    print('Atribut Yang Terpilih :', atributSel)

    return render_template('Adult_NN.html', atributSel=atributSel, fmeasure=fmeasure, recall=recall, precision=precision, akurasi=akurasi,jum=jum,juma=juma, atributTerpilih=atributTerpilih, atributTotal=atributTotal, dataTraining=dataTraining, dataTesting=dataTesting, dataMissing=dataMissing)

@app.route('/Child')
def Child():
    return render_template('Child.html')

@app.route('/Child_NB')
def Child_NB():
    # Dataset Child 
    dataset = pd.read_excel("D:\Py\skripsi\static\Child_.xlsx")
    jum = len(dataset)
    dataset = dataset.dropna(axis = 0, how ='any')
    juma = len(dataset)
    X = np.array(dataset.drop(['Class'], 1))
    y = np.array(dataset['Class'])

    # Metode Naive Bayes 
    metode = GaussianNB()

    # Naive Bayes (NB)
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=.70, random_state=42)
    modelNB = metode.fit(X_train, y_train)
    hasilNB_Xtest = modelNB.predict(X_test)

    # Confusion Matrix (NB)
    cmnb_Adolescent = metrics.confusion_matrix(y_test, hasilNB_Xtest)
    print(classification_report(y_test, hasilNB_Xtest))
    # Confusion Matrix (NB)
    cm_Adolescent = metrics.confusion_matrix(y_test, hasilNB_Xtest)
    plt.figure(1)
    plot_confusion_matrix(metode, X_test, y_test, cmap=plt.cm.Pastel1)
    plt.savefig('D:\Py\skripsi\static\Hasil Confusion Matrix NB Tanpa PSO - Child.png')
    plt.close()

    # Nilai Confusion Matrix (NB)
    akurasi_nb = int(np.around(metrics.accuracy_score(y_test, hasilNB_Xtest)*100))
    precision_nb = int(np.around(metrics.precision_score(y_test, hasilNB_Xtest)*100))
    recall_nb = int(np.around(metrics.recall_score(y_test, hasilNB_Xtest)*100))
    fmeasure_nb = int(np.around(metrics.f1_score(y_test, hasilNB_Xtest)*100))

    #AUC Score (NB)
    nb_probs = metode.predict_proba(X_test)
    random_probs = [0 for _ in range(len(y_test))]
    nb_probs = nb_probs[:,1] 
    #Prediction probability berisi '0'
    nb_auc = metrics.roc_auc_score(y_test, nb_probs)
    random_auc = roc_auc_score(y_test, random_probs)
    print('Naive Bayes: %.3f' % (nb_auc))
    print('Random Prediction : AUCROC = %.3f' % (random_auc))
    
    #Hitung ROC Curves (NB)
    nb_tpr, nb_fpr, _ = metrics.roc_curve(y_test, nb_probs)
    random_fpr, random_tpr, _ = roc_curve(y_test, random_probs)
    
    #ROC Curve (NB - PSO)
    plt.figure(2)
    plt.plot(nb_tpr, nb_fpr, marker='.', color='salmon', label ='Naive Bayes : %.3f' % nb_auc)
    plt.plot(random_fpr, random_tpr, color='crimson', marker='.', label='Random : (AUCROC = %.3f)' % random_auc)
    plt.title('ROC Curve Naive Bayes')
    plt.ylabel('True Positive Rate', size = 20)
    plt.xlabel('False Positive Rate', size = 20)
    plt.legend()
    plt.savefig('D:\Py\skripsi\static\ROC Curve Child - NB.png')
    plt.close()

    # Particle Swarm Optimization
    def f_per_particle(m, alpha):
        total_features = X.shape[1]
        if np.count_nonzero(m) == 0:
            X_subset = X
        else:
            X_subset = X[:,m==1]
        X_train, X_test, y_train, y_test = train_test_split(X_subset,y,train_size=.70, random_state=42)
        metode.fit(X_train, y_train)
        P = (metode.predict(X_test) == y_test).mean()
        j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
        return j
    
    def f(x, alpha=0.88):
        n_particles = x.shape[0]
        j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
        return np.array(j)

    #Initialize swarm, arbitrary
    options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 1  , 'p':2}
    dimensions = X.shape[1] # dimensions should be the number of features
    optimizer = ps.discrete.BinaryPSO(n_particles=2, dimensions=dimensions,options=options) 
    #Optimization
    cost, pos = optimizer.optimize(fx.sphere, iters=20)
    
    # Naive Bayes (NB - PSO)
    fiturSeleksiX = X[:,pos==1]
    X_train, X_test, y_train, y_test = train_test_split(fiturSeleksiX,y,train_size=.70, random_state=42)
    modelNBPSO = metode.fit(X_train, y_train)
    hasilNBPSO_Xtest = modelNBPSO.predict(X_test) # Menentukan hasil prediksi y_pred
    
    # Confusion Matrix (NB - PSO)
    cm_Adolescent = metrics.confusion_matrix(y_test, hasilNBPSO_Xtest)
    plt.figure(1)
    plot_confusion_matrix(metode, X_test, y_test, cmap=plt.cm.Pastel1)
    plt.savefig('D:\Py\skripsi\static\Hasil Confusion Matrix NB - Child.png')
    plt.close()

    #AUC Score (NB - PSO)
    nb_probs = metode.predict_proba(X_test)
    random_probs = [0 for _ in range(len(y_test))]
    nb_probs = nb_probs[:,1] 
    #Prediction probability berisi '0'
    nb_auc = metrics.roc_auc_score(y_test, nb_probs)
    random_auc = roc_auc_score(y_test, random_probs)
    print('Naive Bayes - PSO : %.3f' % (nb_auc))
    print('Random Prediction : AUCROC = %.3f' % (random_auc))
    
    #Hitung ROC Curves (NB - PSO)
    nb_tpr, nb_fpr, _ = metrics.roc_curve(y_test, nb_probs)
    random_fpr, random_tpr, _ = roc_curve(y_test, random_probs)
    
    #ROC Curve (NB - PSO)
    plt.figure(2)
    plt.plot(nb_tpr, nb_fpr, marker='.', color='salmon', label ='Naive Bayes : %.3f' % nb_auc)
    plt.plot(random_fpr, random_tpr, color='crimson', marker='.', label='Random : (AUCROC = %.3f)' % random_auc)
    plt.title('ROC Curve Naive Bayes - PSO')
    plt.ylabel('True Positive Rate', size = 20)
    plt.xlabel('False Positive Rate', size = 20)
    plt.legend()
    plt.savefig('D:\Py\skripsi\static\ROC Curve Child - NBPSO.png')
    plt.close()

    # Nilai Confusion Matrix (NB - PSO)
    akurasi = int(np.around(metrics.accuracy_score(y_test, hasilNBPSO_Xtest)*100))
    precision = int(np.around(metrics.precision_score(y_test, hasilNBPSO_Xtest)*100))
    recall = int(np.around(metrics.recall_score(y_test, hasilNBPSO_Xtest)*100))
    fmeasure = int(np.around(metrics.f1_score(y_test, hasilNBPSO_Xtest)*100))

    # Menampilkan Atribut Terpilih (NB - PSO)
    atribut = ["id","A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score","age","gender","ethnicity","jundice","austim","contry_of_res","used_app_before","age_desc","relation","Class/ASD"]
    atributArr = np.array(atribut)
    atributSel = atributArr[pos==1]
    atributTerpilih = str(sum((pos == 1)*1))
    atributTotal = str(len(pos))
    dataTraining = str(len(X_train))
    dataTesting = str(len(X_test))
    dataMissing = jum - juma

    print (classification_report(y_test, hasilNBPSO_Xtest))
    print('Hasil Confusions Matrix NB : \n', cmnb_Adolescent)
    print('Akurasi NB : ', akurasi_nb)
    print('Precision NB : ', precision_nb)
    print('Recall NB : ', recall_nb)
    print('F-Measure NB : ', fmeasure_nb)
    print('Fitur Terseleksi : \n' + str(sum((pos == 1)*1)) + '/' + str(len(pos)))
    print('Hasil Confusion Matrix NB - PSO: \n', cm_Adolescent)
    print('Akurasi NB - PSO : ', akurasi)
    print('Precision NB - PSO : ', precision)
    print('Recall NB - PSO : ', recall)
    print('F-Measure NB - PSO : ', fmeasure)
    print('Jumlah Dataset : ', jum)
    print('Jumlah Dataset setelah penanganan missing value: ', juma)
    print('Atribut Yang Terpilih :', atributTerpilih)
    print('Atribut Yang Terpilih :', atributSel)


    return render_template('Child_NB.html', atributSel=atributSel, fmeasure=fmeasure, recall=recall, precision=precision, akurasi=akurasi,jum=jum,juma=juma, atributTerpilih=atributTerpilih, atributTotal=atributTotal, dataTraining=dataTraining, dataTesting=dataTesting, dataMissing=dataMissing)

@app.route('/Child_NN')
def Child_NN():
    # Dataset Child 
    dataset = pd.read_excel("D:\Py\skripsi\static\Child_.xlsx")
    jum = len(dataset)
    dataset = dataset.dropna(axis = 0, how ='any')
    juma = len(dataset)
    X = np.array(dataset.drop(['Class'], 1))
    y = np.array(dataset['Class'])

    # Metode Neural Network 
    metode = MLPClassifier(hidden_layer_sizes=(6,), random_state=42)

    # Neural Network (NN)
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=.70, random_state=42)
    modelNN = metode.fit(X_train, y_train)
    hasilNN_Xtest = modelNN.predict(X_test)

    # Confusion Matrix (NN)
    cmnn_Adolescent = metrics.confusion_matrix(y_test, hasilNN_Xtest)
    print(classification_report(y_test, hasilNN_Xtest))
    # Confusion Matrix (NN)
    cm_Adolescent = metrics.confusion_matrix(y_test, hasilNN_Xtest)
    plt.figure(1)
    plot_confusion_matrix(metode, X_test, y_test, cmap=plt.cm.Pastel1)
    plt.savefig('D:\Py\skripsi\static\Hasil Confusion Matrix NN Tanpa PSO - Child.png')
    plt.close()

    # Nilai Confusion Matrix (NN)
    akurasi_nn = int(np.around(metrics.accuracy_score(y_test, hasilNN_Xtest)*100))
    precision_nn = int(np.around(metrics.precision_score(y_test, hasilNN_Xtest)*100))
    recall_nn = int(np.around(metrics.recall_score(y_test, hasilNN_Xtest)*100))
    fmeasure_nn = int(np.around(metrics.f1_score(y_test, hasilNN_Xtest)*100))

    #AUC Score (NN)
    nn_probs = metode.predict_proba(X_test)
    random_probs = [0 for _ in range(len(y_test))]
    nn_probs = nn_probs[:,1] 
    #Prediction probability berisi '0'
    nn_auc = metrics.roc_auc_score(y_test, nn_probs)
    random_auc = roc_auc_score(y_test, random_probs)
    print('Neural Network: %.3f' % (nn_auc))
    print('Random Prediction : AUCROC = %.3f' % (random_auc))
    
    #Hitung ROC Curves (NN)
    nn_tpr, nn_fpr, _ = metrics.roc_curve(y_test, nn_probs)
    random_fpr, random_tpr, _ = roc_curve(y_test, random_probs)
    
    #ROC Curve (NN)
    plt.figure(2)
    plt.plot(nn_tpr, nn_fpr, marker='.', color='salmon', label ='Neural Network : %.3f' % nn_auc)
    plt.plot(random_fpr, random_tpr, color='crimson', marker='.', label='Random : (AUCROC = %.3f)' % random_auc)
    plt.title('ROC Curve Neural Network')
    plt.ylabel('True Positive Rate', size = 20)
    plt.xlabel('False Positive Rate', size = 20)
    plt.legend()
    plt.savefig('D:\Py\skripsi\static\ROC Curve Child - NN.png')
    plt.close()

    # Particle Swarm Optimization
    def f_per_particle(m, alpha):
        total_features = X.shape[1]
        if np.count_nonzero(m) == 0:
            X_subset = X
        else:
            X_subset = X[:,m==1]
        X_train, X_test, y_train, y_test = train_test_split(X_subset,y,train_size=.70, random_state=42)
        metode.fit(X_train, y_train)
        P = (metode.predict(X_test) == y_test).mean()
        j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
        return j
    
    def f(x, alpha=0.88):
        n_particles = x.shape[0]
        j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
        return np.array(j)

    #Initialize swarm, arbitrary
    options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 1  , 'p':2}
    dimensions = X.shape[1] # dimensions should be the number of features
    optimizer = ps.discrete.BinaryPSO(n_particles=2, dimensions=dimensions,options=options) 
    #Optimization
    cost, pos = optimizer.optimize(fx.sphere, iters=20)
    
    # Naive Bayes (NN - PSO)
    fiturSeleksiX = X[:,pos==1]
    X_train, X_test, y_train, y_test = train_test_split(fiturSeleksiX,y,train_size=.70, random_state=42)
    modelNNPSO = metode.fit(X_train, y_train)
    hasilNNPSO_Xtest = modelNNPSO.predict(X_test) # Menentukan hasil prediksi y_pred
    
    # Confusion Matrix (NN - PSO)
    cm_Adolescent = metrics.confusion_matrix(y_test, hasilNNPSO_Xtest)
    plt.figure(1)
    plot_confusion_matrix(metode, X_test, y_test, cmap=plt.cm.Pastel1)
    plt.savefig('D:\Py\skripsi\static\Hasil Confusion Matrix NN - Child.png')
    plt.close()

    #AUC Score (NN - PSO)
    nn_probs = metode.predict_proba(X_test)
    random_probs = [0 for _ in range(len(y_test))]
    nn_probs = nn_probs[:,1] 
    #Prediction probability berisi '0'
    nn_auc = metrics.roc_auc_score(y_test, nn_probs)
    random_auc = roc_auc_score(y_test, random_probs)
    print('Neural Network - PSO : %.3f' % (nn_auc))
    print('Random Prediction : AUCROC = %.3f' % (random_auc))
    
    #Hitung ROC Curves (NN - PSO)
    nn_tpr, nn_fpr, _ = metrics.roc_curve(y_test, nn_probs)
    random_fpr, random_tpr, _ = roc_curve(y_test, random_probs)
    
    #ROC Curve (NN - PSO)
    plt.figure(2)
    plt.plot(nn_tpr, nn_fpr, marker='.', color='salmon', label ='Neural Network - PSO : %.3f' % nn_auc)
    plt.plot(random_fpr, random_tpr, color='crimson', marker='.', label='Random : (AUCROC = %.3f)' % random_auc)
    plt.title('ROC Curve Neural Network - PSO')
    plt.ylabel('True Positive Rate', size = 20)
    plt.xlabel('False Positive Rate', size = 20)
    plt.legend()
    plt.savefig('D:\Py\skripsi\static\ROC Curve Child - NNPSO.png')
    plt.close()

    # Nilai Confusion Matrix (NN - PSO)
    akurasi = int(np.around(metrics.accuracy_score(y_test, hasilNNPSO_Xtest)*100))
    precision = int(np.around(metrics.precision_score(y_test, hasilNNPSO_Xtest)*100))
    recall = int(np.around(metrics.recall_score(y_test, hasilNNPSO_Xtest)*100))
    fmeasure = int(np.around(metrics.f1_score(y_test, hasilNNPSO_Xtest)*100))

    # Menampilkan Atribut Terpilih (NN - PSO)
    atribut = ["id","A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score","age","gender","ethnicity","jundice","austim","contry_of_res","used_app_before","age_desc","relation","Class/ASD"]
    atributArr = np.array(atribut)
    atributSel = atributArr[pos==1]
    atributTerpilih = str(sum((pos == 1)*1))
    atributTotal = str(len(pos))
    dataTraining = str(len(X_train))
    dataTesting = str(len(X_test))
    dataMissing = jum - juma

    print (classification_report(y_test, hasilNNPSO_Xtest))
    print('Hasil Confusions Matrix NN : \n', cmnn_Adolescent)
    print('Akurasi NN : ', akurasi_nn)
    print('Precision NN: ', precision_nn)
    print('Recall NN: ', recall_nn)
    print('F-Measure NN: ', fmeasure_nn)
    print('Fitur Terseleksi : \n' + str(sum((pos == 1)*1)) + '/' + str(len(pos)))
    print('Hasil Confusion Matrix NN - PSO: \n', cm_Adolescent)
    print('Akurasi NN - PSO : ', akurasi)
    print('Precision NN - PSO : ', precision)
    print('Recall NN - PSO : ', recall)
    print('F-Measure NN - PSO : ', fmeasure)
    print('Jumlah Dataset : ', jum)
    print('Jumlah Dataset setelah penanganan missing value: ', juma)
    print('Atribut Yang Terpilih :', atributTerpilih)
    print('Atribut Yang Terpilih :', atributSel)

    return render_template('Child_NN.html', atributSel=atributSel, fmeasure=fmeasure, recall=recall, precision=precision, akurasi=akurasi,jum=jum,juma=juma, atributTerpilih=atributTerpilih, atributTotal=atributTotal, dataTraining=dataTraining, dataTesting=dataTesting, dataMissing=dataMissing)

app.run(debug=True)