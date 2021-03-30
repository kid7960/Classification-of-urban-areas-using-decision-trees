import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# load the data
path = '/Users/kid79/Desktop/Coursework/CourseWork 2020-2021/ECM2423 Artificial Intelligence (Term 2)/data.csv'

data = pd.read_csv(path, sep=',')

# splitting dataset in features and target variable
feature_columns = ['Roads:number_intersections', 'Roads:diversity', 'Roads:total', 'Buildings:diversity',
                   'Buildings:total', 'LandUse:Mix', 'TrafficPoints:crossing', 'poisAreas:area_park',
                   'poisAreas:area_pitch', 'pois:diversity', 'pois:total', 'ThirdPlaces:oa_count',
                   'ThirdPlaces:edt_count', 'ThirdPlaces:out_count', 'ThirdPlaces:cv_count', 'ThirdPlaces:diversity',
                   'ThirdPlaces:total', 'vertical_density', 'buildings_age', 'buildings_age:diversity']

X = data[feature_columns]  # Features
y = data.most_present_age  # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

# accuracy score on train and test set
print("Accuracy of training set")
print(clf.score(X_train, y_train))
print("Accuracy of test set")
print(clf.score(X_test, y_test))

# calculate accuracy for varying depth of the tree and see how it changes
dtc = DecisionTreeClassifier()
scores = []

for dep in range(1, 50):

    dtc = DecisionTreeClassifier(max_depth = dep)
    clf = dtc.fit(X_train,y_train)
    scores.append([dep, clf.score(X_test, y_test)])

print("Accuracy for varying depth of tree")
for i in scores:
    print(i)

