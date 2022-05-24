from inspect import ArgInfo
from flask import Flask, render_template, request
import shutil

app = Flask(__name__)

@app.route('/')
def principal():
        #uploaded = files.upload()

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    #%matplotlib inline
    import math

    data_heart= pd.read_csv("heart_change.csv")
    data_heart.head(10)
    x = data_heart.drop("HeartDisease",axis=1)
    y= data_heart["HeartDisease"]

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    test_tamano = 0.2
    train_tamano = 1 - test_tamano
    trainX, testX, trainY, testY = train_test_split(x, y, test_size = test_tamano, random_state=42)
    logmodel=LogisticRegression()
    #logmodel.fit(trainX, trainY)
    sc=StandardScaler()

    scaler = sc.fit(trainX)
    trainX_scaled = scaler.transform(trainX)
    testX_scaled = scaler.transform(testX)
    logmodel.fit(trainX_scaled, trainY)
    predictions = logmodel.predict(testX_scaled)
    classification_report(testY,predictions)
    confusion_matrix(testY,predictions)
    Precision  = accuracy_score(testY, predictions)
    logmodel.score(testX_scaled,testY)
    Precision = limited_float = round(Precision, 3)
    Precision = Precision *100 
    return render_template('principal.html', PrecisionP=Precision, test = test_tamano, train = train_tamano)

@app.route('/inicio')
def inicio():
    return render_template('index.html')

@app.route('/inicio2')
def inicio2():
    return render_template('estadisticas.html')

@app.route('/datos', methods=['POST', 'GET'])
def datos():
    vector = [1,2,3,4,5,6,7,8,9,10,11]
    i = 0
    if request.method == "POST":
        Nombre = request.form['Nombre']

        Edad = request.form['Edad']
        vector[i]=int(Edad)
        i = i + 1
        
        Sexo = request.form['Sexo']
        vector[i]=int(Sexo)
        i = i + 1
        
        DolorPecho = request.form['DolorPecho']
        vector[i]=int(DolorPecho)
        i = i + 1
        
        PAreposo = request.form['PA']
        vector[i]=int(PAreposo)
        i = i + 1
        
        Colesterol = request.form['Colesterol']
        vector[i]=int(Colesterol)
        i = i + 1
        
        Glucemia = request.form['Glucemia']
        vector[i]=int(Glucemia)
        i = i + 1
        
        Electrocardiograma = request.form['Electrocardiograma']
        vector[i]=int(Electrocardiograma)
        i = i + 1
        
        FrecuenciaCardiaca = request.form['FrecuenciaCardiaca']
        vector[i]=int(FrecuenciaCardiaca)
        i = i + 1
        
        Angina = request.form['Angina']
        vector[i]=int(Angina)
        i = i + 1
        
        Pico = request.form['Pico']
        vector[i]=float(Pico)
        i = i + 1
        
        Pendiente = request.form['Pendiente']
        vector[i]=int(Pendiente)
        i = i + 1
    i=0
    print(Nombre)
    print(Edad)
    print(Sexo)
    print(DolorPecho)
    print(PAreposo)
    print(Colesterol)
    print(Glucemia)
    print(Electrocardiograma)
    print(FrecuenciaCardiaca)
    print(Angina)
    print(Pico)
    print(Pendiente)
    print("ARREGLO")
    print(vector)

    #uploaded = files.upload()

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    #%matplotlib inline
    import math

    data_heart= pd.read_csv("heart_change.csv")
    data_heart.head(10)
    x = data_heart.drop("HeartDisease",axis=1)
    y= data_heart["HeartDisease"]

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2, random_state=42)
    logmodel=LogisticRegression()
    #logmodel.fit(trainX, trainY)
    sc=StandardScaler()

    scaler = sc.fit(trainX)
    trainX_scaled = scaler.transform(trainX)
    testX_scaled = scaler.transform(testX)
    logmodel.fit(trainX_scaled, trainY)
    predictions = logmodel.predict(testX_scaled)
    classification_report(testY,predictions)
    confusion_matrix(testY,predictions)
    accuracy_score(testY, predictions)
    logmodel.score(testX_scaled,testY)
    #Datos tienen que ser un array 2D
    array = [vector]
    print(array)

    sc=StandardScaler()
    scaler = sc.fit(trainX)
    array_scaled = scaler.transform(array)
    respuesta = logmodel.predict(array_scaled)
    Resultado = int(respuesta[0])
    print("Aqui")
    probabilidad = logmodel.predict_proba(array_scaled) 
    print(probabilidad)
    Probabilidad1 = float(probabilidad[0][0])
    Probabilidad2 = float(probabilidad[0][1])
    Probabilidad1 = limited_float = round(Probabilidad1, 2)
    Probabilidad2 = limited_float = round(Probabilidad2, 2)
    Probabilidad1 = Probabilidad1 *100 
    Probabilidad2 = Probabilidad2 *100
    print(Resultado)



    if Resultado == 1:
        mensaje = "TENER"
    else:
        mensaje = "NO TENER"
    
    # data_heart_table= pd.read_csv("heart.csv")
    # sns.countplot(x="HeartDisease", data= data_heart_table)
    # plt.savefig("grafica1.png", transparent=True)
    # shutil.copy("grafica1.png", "static/grafica1.png")
    # plt.clf()
    # sns.countplot(x="HeartDisease", hue= "Sex", data= data_heart_table)
    # plt.savefig("grafica2.png", transparent=True)
    # shutil.copy("grafica2.png", "static/grafica2.png")
    # plt.clf()
    # sns.countplot(x="HeartDisease", hue= "ChestPainType", data= data_heart_table)
    # plt.savefig("grafica3.png", transparent=True)
    # shutil.copy("grafica3.png", "static/grafica3.png")
    # plt.clf()
    # sns.countplot(x="HeartDisease", hue= "ExerciseAngina", data= data_heart_table)
    # plt.savefig("grafica4.png", transparent=True)
    # shutil.copy("grafica4.png", "static/grafica4.png")
    # plt.clf()
    # data_heart_table["Age"].plot.hist()

    
    return render_template('Respuesta.html', R=mensaje, N=Nombre, vector=vector, P1 = Probabilidad1, P2 = Probabilidad2)

if __name__ == '__main__':
    app.run(debug=True)