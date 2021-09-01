"""
Wybrałam do swojego projektu dane dotyczące cech portugalskich win z rodzaju Vinho Verde
Zawiera on 11 zmiennych pochodzących z analizy chemicznej i jedna zmienna oznaczająca jakosc z oceny sensorycznej
wybrałam zestaw danych dotyczących win czerwonych, bo takie preferuję :)
jest też dotępny zestaw danych dla win białych
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#pobranie i wczytanie danych
dataset_url ='https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
wine = pd.read_csv(dataset_url, sep=';')

#%%
#sprawdzenie, czy dane poprawinie się załadowały i jak wygląda 5 pierwszych kolumn
print(wine.head(), '\n')

#sprawdzam podstawowe informacje o danych: nazwy kolumn, czy są jakies braki, jaki jest typ danych
print(wine.info(), '\n')
#widze, że nie ma braków i wszystkie dane są liczbowe

#najbardziej interesuje mnie kolumna "jakosc" - sprawdzam wiec jak wygladaja tam dane
print(wine['quality'].describe(), '\n', wine['quality'].value_counts(), '\n')

#sprawdzam jak wygląda rozkład zmiennej jakosc
rozklad=sns.countplot(x='quality', data=wine, palette = 'PuRd')
plt.show()

#patrzę jak rozkłada się zmienna jakosc w zaleznosci od innych zmiennych, z nadzieją, że uda mi się cos wywnioskowac
for label in wine.columns[:-1]:
    plt.scatter(wine['quality'], wine[label], c='crimson')
    plt.title(label)
    plt.xlabel('quality')
    plt.ylabel(label)
    plt.show()
#próżne są moje starania, gołym okiem nie widzę żadnej zależnosci

#sprawdzam jeszcze korelacje między poszczególnymi zmiennymi
print(wine.corr(), '\n')
sns.heatmap(wine.corr())
plt.show()
"""
widać korelacje między wolnymi i całkowitymi siarczanami oraz co absulutnie logiczne
odwrotną korelację pomiędzy kwasem cytrynowym oraz kwasowoscią z pH
jedyna zmienna, z którą jakosc ma lekką odwrotną korelację jest lotna kwasowosc - volatile acidity -
za lotną kwasowosc w winie odpowiada głównie kwas octowy - oczywistym jest, że wino, które pachnie i smakuje octem
nie będzie odbierane jako dobre jakosciowo
"""
#%%
#żeby trochę ułatwić odbiór klasyfikacji jakosci wina zmieniam skalę ocen 0-10 na "niezbyt chętnie bym wypiła" - 0 (jakosc 3 - 5),
#"nawet chętnie bym wypiła" - 1 (jakosc 6-7), "bardzo chętnie bym wypiła" - 2 (jakosc 8-9)
bins = [0, 5.5, 7.5, 10] 
labels = [0, 1, 2]
wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=labels)

#sprawdzam jak wyglądają dane po transformacji
print(wine.head(), '\n')

#wszystkie dane z analizy chemicznej ustalam jako input a stopień chęci wypicia jako output
x = wine[wine.columns[:-1]]
y = wine['quality']

#dokonuje standaryzacji danych z analizy chemicznej, ponieważ były one mierzone w różnych jednostkach i skalach
sc = StandardScaler()
x = sc.fit_transform(x)

#dzielę dane na zbiór treningowy(75%) i testowy (25%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=42)

#sprawdzam ile danych jest w każdym zbiorze, czyli czy podział zaszedł prawidłowo
print(y_train.value_counts(), '\n', y_test.value_counts())

#%%
#zaczynam od algorytmu K najbliższych sąsiadów - KNeighbors, najpierw 3 sąsiadów, potem 5, do porównania używam walidacji krzyżowej
k3 = KNeighborsClassifier(n_neighbors = 3)
k3.fit(x_train, y_train)

pred_k3 = k3.predict(x_test)
print(classification_report(y_test, pred_k3))

cross_val_3 = cross_val_score(estimator=k3, X=x_train, y=y_train)
print("Średnia walidacja przy 3 sąsiadach: ", cross_val_3.mean(), '\n')

pred3_ac = accuracy_score(y_test, pred_k3)
print("% accuracy przy 5 sąsiadach: ", pred3_ac*100, '\n')


k5 = KNeighborsClassifier(n_neighbors = 5)
k5.fit(x_train, y_train)

pred_k5 = k5.predict(x_test)
print(classification_report(y_test, pred_k5))

cross_val_5 = cross_val_score(estimator=k5, X=x_train, y=y_train)
print("Średnia walidacja przy 5 sąsiadach: ", cross_val_5.mean(), '\n')
pred5_ac = accuracy_score(y_test, pred_k5)
print("% accuracy przy 5 sąsiadach: ", pred5_ac*100, '\n')
#przy trzech sąsiadach srednia jest nieznacznie nizsza - zwiększenie liczby sąsiadów nie powoduje odczuwalnego ulepszenia modelu
#a dalsze zwiększanie może prowadzić do pogorszenia

#%%
#czas na algorytmy drzewne
#następnym algorytmem jest las losowy - Random Forest
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

pred_rfc = rfc.predict(x_test)
print(classification_report(y_test, pred_rfc))

cross_val_rfc = cross_val_score(estimator=rfc, X=x_train, y=y_train)
print("Średnia walidacja dla Random Forest: ", cross_val_rfc.mean(), '\n')
rfc_ac = accuracy_score(y_test, pred_rfc)
print("% accuracy dla Random Forest: ", rfc_ac*100, '\n')
#tutaj mamy znaczący wzrost, bo o około 10%

#%%
#algorytm drzew decyzyjnych - Decision Tree
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)

pred_dtc = dtc.predict(x_test)
print(classification_report(y_test, pred_dtc))

cross_val_dtc = cross_val_score(estimator=dtc, X=x_train, y=y_train)
print("Średnia walidacja dla Decision Tree: ", cross_val_dtc.mean(), '\n')
dtc_ac = accuracy_score(y_test, pred_dtc)
print("% accuracy dla Decision Tree: ", dtc_ac*100, '\n')
#wynik podobny do algorytmu najbliższych sąsiadów

#%%
#Regresja logistyczna
lr = LogisticRegression()
lr.fit(x_train, y_train)

pred_lr = lr.predict(x_test)
print(classification_report(y_test, pred_lr))

cross_val_lr = cross_val_score(estimator=lr, X=x_train, y=y_train)
print("Średnia walidacja dla Regresji Logistycznej: ", cross_val_lr.mean(), '\n')

lr_ac = accuracy_score(y_test, pred_lr)
print("% accuracy dla Regresji Logistycznej: ", lr_ac*100, '\n')
#wynki podobne do KNeighbors i Decision Tree

#%%
#Najlepszy okazał się algorytm Random Forest - teraz trzeba go wytrenować
n_estimators = [2,30,50]
max_features = [2,4,6,8]
max_depth = [10,15,20] + [None]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True,False]

random_grid = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'bootstrap': bootstrap}

best_rfc = RandomizedSearchCV(estimator=rfc, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42)

best_rfc.fit(x_train, y_train)
pred_best_rfc = best_rfc.predict(x_test)
print(classification_report(y_test, pred_best_rfc))
print(best_rfc.best_params_)

#%%
#sprawdzam dokładnosc wytrenowanego modelu z najlepszymi parametrami uzyskanymi wczesniej
best_rfc = RandomForestClassifier(n_estimators= 50, min_samples_split= 2, min_samples_leaf = 1, max_features= 2, max_depth=  None, bootstrap= True)
best_rfc.fit(x_train, y_train)

cross_val_best_rfc = cross_val_score(estimator=best_rfc, X=x_train, y=y_train)
print("\n Średnia walidacja dla najlepszego modelu Random Forest: ", cross_val_best_rfc.mean(), '\n')

"""
Niestety po treningu srednia walidacja wcale nie jest wyzsza, wrecz przeciwnie, 
jest odrobinę niższa, może to wynikać z faktu, że algorytm Random Forest jest wrażliwy na overfitting.
Jesli kiedykolwiek będę miała tak bogatą piwniczkę na wino, że nie będę wiedziała, które z win
obecnych w niej jest dobre, to przeprowadzę analizę chemiczną tych cech, które były wykorzystane w tym zestawie danych
a do klasyfikacji użyję algorytmu Random Forest bez trenowania
"""







