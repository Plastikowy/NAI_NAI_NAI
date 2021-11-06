"""
==========================================
Program do obliczania rankingu hoteli

Twórcy:
Tomasz Samól (Plastikowy)
Sebastian Lewandowski (SxLewandowski)
==========================================
Aby uruchomić program należy zainstalować następujące paczki:
pip install numpy
pip install scikit-fuzzy
pip install matplotlib

Stworzony przez nas program umożliwia wyliczanie oceny np. hotelu na postawie
podanych w programie danych wejściowych z wykorzystaniem logiki rozmytej.
Na wejściu podajemy oceny od 0-10 dla lokalizacji, ceny i dodatków, a na
wyjściu otrzymujemy ocenę.

==========================================
* Antecednets (dane wejściowe)
    -`cena`
 	    * ocena uniwersalna( wartości 0-10)
	    * ocena dla logiki rozmytej( poor, average, good)
    - `lokalizacja`
        * ocena uniwersalna( wartości 0-10)
        * ocena dla logiki rozmytej( poor, average, good)
    -`dodatki`
        * ocena uniwersalna( wartości 0-10)
        * ocena dla logiki rozmytej( poor, average, good)
* Consequents (dane wyjściowe)
    - `ocena`
        * ocena dla logiki rozmytej( verylow, low, medium, high, veryhigh)
* Zasady
    - Jeśli cena to  dodatki i lokalizacja będzie „poor” to ocena będzie „verylow”
    - Jeśli lokalizacja będzie "poor" & cena będzie "poor" & dodatki będą "poor" to  rating będzie "verylow"
    - Jeśli lokalizacja będzie "poor" & cena będzie "average" & dodatki będą "poor" to  rating będzie "verylow"
    - Jeśli lokalizacja będzie "poor" & cena będzie "poor" & dodatki będą "average" to  rating będzie "verylow"
    - Jeśli lokalizacja będzie "poor" & cena będzie "average" & dodatki będą "average" to  rating będzie "low"
    - Jeśli lokalizacja będzie "poor" & cena będzie "good" & dodatki będą "average" to  rating będzie "medium"
    - Jeśli lokalizacja będzie "poor" & cena będzie "average" & dodatki będą "good" to  rating będzie "medium"
    - Jeśli lokalizacja będzie "poor" & cena będzie "good" & dodatki będą "good" to  rating będzie "medium"
    - Jeśli lokalizacja będzie "poor" & cena będzie "poor" & dodatki będą "good" to  rating będzie "low"
    - Jeśli lokalizacja będzie "poor" & cena będzie "good" & dodatki będą "poor" to  rating będzie "low"
    - Jeśli lokalizacja będzie "average" & cena będzie "poor" & dodatki będą "poor" to  rating będzie "verylow"
    - Jeśli lokalizacja będzie "average" & cena będzie "average" & dodatki będą "poor" to  rating będzie "low"
    - Jeśli lokalizacja będzie "average" & cena będzie "poor" & dodatki będą "average" to  rating będzie "low"
    - Jeśli lokalizacja będzie "average" & cena będzie "average" & dodatki będą "average" to  rating będzie "medium"
    - Jeśli lokalizacja będzie "average" & cena będzie "good" & dodatki będą "average" to  rating będzie "high"
    - Jeśli lokalizacja będzie "average" & cena będzie "average" & dodatki będą "good" to  rating będzie "high"
    - Jeśli lokalizacja będzie "average" & cena będzie "good" & dodatki będą "good" to  rating będzie "veryhigh"
    - Jeśli lokalizacja będzie "average" & cena będzie "poor" & dodatki będą "good" to  rating będzie "medium"
    - Jeśli lokalizacja będzie "average" & cena będzie "good" & dodatki będą "poor" to  rating będzie "medium"
    - Jeśli lokalizacja będzie "good" & cena będzie "poor" & dodatki będą "poor" to  rating będzie "low"
    - Jeśli lokalizacja będzie "good" & cena będzie "average" & dodatki będą "poor" to  rating będzie "medium"
    - Jeśli lokalizacja będzie "good" & cena będzie "poor" & dodatki będą "average" to  rating będzie "medium"
    - Jeśli lokalizacja będzie "good" & cena będzie "average" & dodatki będą "average" to  rating będzie "high"
    - Jeśli lokalizacja będzie "good" & cena będzie "good" & dodatki będą "average" to  rating będzie "veryhigh"
    - Jeśli lokalizacja będzie "good" & cena będzie "average" & dodatki będą "good" to  rating będzie "veryhigh"
    - Jeśli lokalizacja będzie "good" & cena będzie "good" & dodatki będą "good" to  rating będzie "medium"
    - Jeśli lokalizacja będzie "good" & cena będzie "poor" & dodatki będą "good" to  rating będzie "medium"
    - Jeśli lokalizacja będzie "good" & cena będzie "good" & dodatki będą "poor" to  rating będzie "medium"

"""

# importujemy potrzebne paczki
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# deklarujemy potrzebne zmienne dla logiki rozmytej
# 3 na wejsciu, jedna na wyjściu
localization = ctrl.Antecedent(np.arange(0, 11, 1), 'localization')
price = ctrl.Antecedent(np.arange(0, 11, 1), 'price')
additions = ctrl.Antecedent(np.arange(0, 11, 1), 'additions')
rating = ctrl.Consequent(np.arange(0, 11, 1), 'rating')

# dodajemy automatycznie 3 poziomy dla naszych zmiennych ( poor, average, good)
localization.automf(3)
price.automf(3)
additions.automf(3)

# przyjmujemy 5 wartosci lignwistycznych dla oceny: verylow, low, miedium, high,
# veryhigh
rating['verylow'] = fuzz.trimf(rating.universe, [0, 2, 2])
rating['low'] = fuzz.trimf(rating.universe, [2, 4, 4])
rating['medium'] = fuzz.trimf(rating.universe, [4, 6, 6])
rating['high'] = fuzz.trimf(rating.universe, [6, 8, 8])
rating['veryhigh'] = fuzz.trimf(rating.universe, [8, 10, 10])

# wyswietlamy zasady dla lokalizacji, ceny i oceny za pomocą wykresów
localization['average'].view()
price.view()
rating.view()

# deklarujemy nasze zasady
rule1 = ctrl.Rule(localization['poor'] & price['poor'] & additions['poor'], rating['verylow'])
rule2 = ctrl.Rule(localization['poor'] & price['average'] & additions['poor'], rating['verylow'])
rule3 = ctrl.Rule(localization['poor'] & price['poor'] & additions['average'], rating['verylow'])
rule4 = ctrl.Rule(localization['poor'] & price['average'] & additions['average'], rating['low'])
rule5 = ctrl.Rule(localization['poor'] & price['good'] & additions['average'], rating['medium'])
rule6 = ctrl.Rule(localization['poor'] & price['average'] & additions['good'], rating['medium'])
rule7 = ctrl.Rule(localization['poor'] & price['good'] & additions['good'], rating['medium'])
rule8 = ctrl.Rule(localization['poor'] & price['poor'] & additions['good'], rating['low'])
rule9 = ctrl.Rule(localization['poor'] & price['good'] & additions['poor'], rating['low'])

rule10 = ctrl.Rule(localization['average'] & price['poor'] & additions['poor'], rating['verylow'])
rule11 = ctrl.Rule(localization['average'] & price['average'] & additions['poor'], rating['low'])
rule12 = ctrl.Rule(localization['average'] & price['poor'] & additions['average'], rating['low'])
rule13 = ctrl.Rule(localization['average'] & price['average'] & additions['average'], rating['medium'])
rule14 = ctrl.Rule(localization['average'] & price['good'] & additions['average'], rating['high'])
rule15 = ctrl.Rule(localization['average'] & price['average'] & additions['good'], rating['high'])
rule16 = ctrl.Rule(localization['average'] & price['good'] & additions['good'], rating['veryhigh'])
rule17 = ctrl.Rule(localization['average'] & price['poor'] & additions['good'], rating['medium'])
rule18 = ctrl.Rule(localization['average'] & price['good'] & additions['poor'], rating['medium'])

rule19 = ctrl.Rule(localization['good'] & price['poor'] & additions['poor'], rating['low'])
rule20 = ctrl.Rule(localization['good'] & price['average'] & additions['poor'], rating['medium'])
rule21 = ctrl.Rule(localization['good'] & price['poor'] & additions['average'], rating['medium'])
rule22 = ctrl.Rule(localization['good'] & price['average'] & additions['average'], rating['high'])
rule23 = ctrl.Rule(localization['good'] & price['good'] & additions['average'], rating['veryhigh'])
rule24 = ctrl.Rule(localization['good'] & price['average'] & additions['good'], rating['veryhigh'])
rule25 = ctrl.Rule(localization['good'] & price['good'] & additions['good'], rating['medium'])
rule26 = ctrl.Rule(localization['good'] & price['poor'] & additions['good'], rating['medium'])
rule27 = ctrl.Rule(localization['good'] & price['good'] & additions['poor'], rating['medium'])

# ControlSystem jest to klasa bazowa zawierajaca nasze zasady
ratinging_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
                                     rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19,
                                     rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27])

# ControlSystemSimulation tworzy kontroler na podstawie naszych zasad
ratinging = ctrl.ControlSystemSimulation(ratinging_ctrl)

# tutaj podajemy nasze wartości wejściowe
ratinging.input['localization'] = 5.7
ratinging.input['price'] = 7.8
ratinging.input['additions'] = 6.2

# obliczamy rating
ratinging.compute()

# wyświetlenie wyniku ratingu
print(ratinging.output['rating'])
rating.view(sim=ratinging)

# wyswietlamy wykres za pomocą matplotlib
plt.show()
