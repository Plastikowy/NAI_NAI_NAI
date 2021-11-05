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
    - Jeśli cena lub lokalizacja są "good" to ocena będzie „high”
    - Jeśli cena lub lokalizacja jest „average”, to ocena będzie „medium”
    - Jeśli cena będzie „poor” i lokalizacja będzie „poor” to ocena będzie „low”
    - Jeśli cena, dodatki i lokalizacja będzie „poor” to ocena będzie „verylow”

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

#
localization.automf(3)
price.automf(3)
additions.automf(5)

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
rule1 = ctrl.Rule(localization['poor'] | price['poor'] | additions['poor'], rating['verylow'])
rule2 = ctrl.Rule(price['average'], rating['medium'])
rule3 = ctrl.Rule(price['good'] | localization['good'], rating['high'])
rule4 = ctrl.Rule(localization['average'] | price['poor'], rating['medium'])
rule5 = ctrl.Rule(localization['poor'] | price['poor'], rating['low'])

# ControlSystem jest to klasa bazowa zawierajaca nasze zasady
ratinging_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])

#
ratinging = ctrl.ControlSystemSimulation(ratinging_ctrl)

# tutaj podajemy nasze wartości wejściowe
ratinging.input['localization'] = 5.7
ratinging.input['price'] = 7.8
ratinging.input['additions'] = 6.2

# obliczamy rating
ratinging.compute()

#
print (ratinging.output['rating'])
rating.view(sim=ratinging)

# wyswietlamy wykres za pomocą matplotlib
plt.show() 