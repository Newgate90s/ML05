#Machine Learning Classification
from sklearn import tree

# Training data
# Features = Weight in ounces, width in millimeters
features = [[4, 149], [5, 160], [11, 200], [12, 230]]

# Labeled training data
# Labels = ["Smartphone"(0), "Smartphone"(0), "Tablet"(1), "Tablet"(1)] used as integers
labels = [0, 0, 1, 1]

# The classifier is the decision tree classifier from tree within sklearn
classifier = tree.DecisionTreeClassifier()

# Machine Learning is done here by the classifier using features and labels
# Fit is finding the patterns in the data
classifier = classifier.fit(features, labels)

print("###################################################")
print("Welcome to is it a smartphone or tablet prediction!")
print("###################################################")
print("\nWe will give you one random weight(ounces) and one random width(millimeters)")
print("You must use the weight and width to guess if it's a smartphone or table!")
print("\nCan you outguess the machine?! \nGet ready!")

# Waits for user input to continue
wait = input("Press enter to receive weight and width.")
print("\nWeight: 4 ounces \nWidth: 152 millimeters")
print("\nSmartphone or tablet?")

# Waits for user input to continue
wait = input("Press enter to see machine prediction")
print("\nMachine guess(0 = Smartphone, Tablet = 1): ")
# Random unknown data to be predicted using features
# Weight in ounces = 4, Width in millimeters = 152
print(classifier.predict([[4, 152]]))

# Waits for user input to continue
wait = input("Press enter for round two!")
print("\nWeight: 13 ounces \nWidth: 204 millimeters")
print("\nSmartphone or tablet?")

# Waits for user input to continue
wait = input("Press enter to see machine prediction")
print("\nMachine guess(0 = Smartphone, Tablet = 1): ")

# Random unknown data to be predicted using features
# Weight in ounces = 13, Width in millimeters = 204
print(classifier.predict([[13, 204]]))
