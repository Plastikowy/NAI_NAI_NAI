"""
==========================================
Fuzzy Control Systems: The ratingping Problem
==========================================
To run program install
pip install scikit-fuzzy
pip install matplotlib

The 'ratingping problem' is commonly used to illustrate the power of fuzzy logic
principles to generate complex behavior from a compact, intuitive set of
expert rules.

If you're new to the world of fuzzy control systems, you might want
to check out the `Fuzzy Control Primer
<../userguide/fuzzy_control_primer.html>`_
before reading through this worked example.

The ratingping Problem
-------------------

Let's create a fuzzy control system which models how you might choose to rating
at a restaurant.  When ratingping, you consider the price and food localization,
rated between 0 and 10.  You use this to leave a rating of between 0 and 25%.

We would formulate this problem as:

* Antecednets (Inputs)
   - `price`
      * Universe (ie, crisp value range): How good was the price of the wait
        staff, on a scale of 0 to 10?
      * Fuzzy set (ie, fuzzy value range): poor, acceptable, amazing
   - `food localization`
      * Universe: How tasty was the food, on a scale of 0 to 10?
      * Fuzzy set: bad, decent, great
* Consequents (Outputs)
   - `rating`
      * Universe: How much should we rating, on a scale of 0% to 25%
      * Fuzzy set: low, medium, high
* Rules
   - IF the *price* was good  *or* the *food localization* was good,
     THEN the rating will be high.
   - IF the *price* was average, THEN the rating will be medium.
   - IF the *price* was poor *and* the *food localization* was poor
     THEN the rating will be low.
* Usage
   - If I tell this controller that I rated:
      * the price as 9.8, and
      * the localization as 6.5,
   - it would recommend I leave:
      * a 20.2% rating.


Creating the ratingping Controller Using the skfuzzy control API
-------------------------------------------------------------

We can use the `skfuzzy` control system API to model this.  First, let's
define fuzzy variables
"""
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl



# New Antecedent/Consequent objects hold universe variables and membership
# functions
localization = ctrl.Antecedent(np.arange(0, 11, 1), 'localization')
price = ctrl.Antecedent(np.arange(0, 11, 1), 'price')
additions = ctrl.Antecedent(np.arange(0, 11, 1), 'additions')
rating = ctrl.Consequent(np.arange(0, 11, 1), 'rating')

# Auto-membership function population is possible with .automf(3, 5, or 7)
localization.automf(3)
price.automf(3)
additions.automf(7)

# Custom membership functions can be built interactively with a familiar,
# Pythonic API
rating['verylow'] = fuzz.trimf(rating.universe, [0, 2, 2])
rating['low'] = fuzz.trimf(rating.universe, [2, 4, 4])
rating['medium'] = fuzz.trimf(rating.universe, [4, 6, 6])
rating['high'] = fuzz.trimf(rating.universe, [6, 8, 8])
rating['veryhigh'] = fuzz.trimf(rating.universe, [8, 10, 10])

"""
To help understand what the membership looks like, use the ``view`` methods.
"""

# You can see how these look with .view()
localization['average'].view()
"""
.. image:: PLOT2RST.current_figure
"""
price.view()
"""
.. image:: PLOT2RST.current_figure
"""
rating.view()
"""
.. image:: PLOT2RST.current_figure


Fuzzy rules
-----------

Now, to make these triangles useful, we define the *fuzzy relationship*
between input and output variables. For the purposes of our example, consider
three simple rules:

1. If the food is poor OR the price is poor, then the rating will be low
2. If the price is average, then the rating will be medium
3. If the food is good OR the price is good, then the rating will be high.

Most people would agree on these rules, but the rules are fuzzy. Mapping the
imprecise rules into a defined, actionable rating is a challenge. This is the
kind of task at which fuzzy logic excels.
"""

rule1 = ctrl.Rule(localization['poor'] | price['poor'] | additions['poor'], rating['verylow'])
rule2 = ctrl.Rule(price['average'], rating['medium'])
rule3 = ctrl.Rule(price['good'] | localization['good'], rating['high'])
rule4 = ctrl.Rule(localization['average'] | price['poor'], rating['medium'])



"""
.. image:: PLOT2RST.current_figure

Control System Creation and Simulation
---------------------------------------

Now that we have our rules defined, we can simply create a control system
via:
"""

ratinging_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])

"""
In order to simulate this control system, we will create a
``ControlSystemSimulation``.  Think of this object representing our controller
applied to a specific set of cirucmstances.  For ratingping, this might be ratingping
Sharon at the local brew-pub.  We would create another
``ControlSystemSimulation`` when we're trying to apply our ``ratingping_ctrl``
for Travis at the cafe because the inputs would be different.
"""

ratinging = ctrl.ControlSystemSimulation(ratinging_ctrl)

"""
We can now simulate our control system by simply specifying the inputs
and calling the ``compute`` method.  Suppose we rated the localization 6.5 out of 10
and the price 9.8 of 10.
"""
# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
ratinging.input['localization'] = 3.7
ratinging.input['price'] = 9.3
ratinging.input['additions'] = 8.8

# Crunch the numbers
ratinging.compute()

"""
Once computed, we can view the result as well as visualize it.
"""
print (ratinging.output['rating'])
rating.view(sim=ratinging)

"""
.. image:: PLOT2RST.current_figure

The resulting suggested rating is **20.24%**.

Final thoughts
--------------

The power of fuzzy systems is allowing complicated, intuitive behavior based
on a sparse system of rules with minimal overhead. Note our membership
function universes were coarse, only defined at the integers, but
``fuzz.interp_membership`` allowed the effective resolution to increase on
demand. This system can respond to arbitrarily small changes in inputs,
and the processing burden is minimal.
"""
plt.show() 