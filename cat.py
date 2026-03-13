import pandas as pd
import numpy as np
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# df = pd.read_csv("tennis.csv")

df = pd.DataFrame({
    "Outlook": ["Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast","Sunny",
                "Sunny","Rain","Sunny","Overcast","Overcast","Rain"],
    "Temperature": ["Hot","Hot","Hot","Mild","Cool","Cool","Cool","Mild",
                    "Cool","Mild","Mild","Mild","Hot","Mild"],
    "Humidity": ["High","High","High","High","Normal","Normal","Normal","High",
                 "Normal","Normal","Normal","High","Normal","High"],
    "Wind": ["Weak","Strong","Weak","Weak","Weak","Strong","Strong","Weak",
             "Weak","Weak","Strong","Strong","Weak","Strong"],
    "Play": ["No","No","Yes","Yes","Yes","No","Yes","No",
             "Yes","Yes","Yes","Yes","Yes","No"]
})

def entropy(data, target):
    counts = data[target].value_counts()
    probs = counts / len(data)
    return -sum(probs * np.log2(probs))

def information_gain(data, feature, target):
    total_entropy = entropy(data, target)
    weighted_entropy = 0

    for value in data[feature].unique():
        subset = data[data[feature] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset, target)

    return total_entropy - weighted_entropy

def id3_steps(data, features, target, level=0):
    indent = "    " * level

    print("\n" + indent + "-" * 50)
    print(indent + f"Current Dataset at Level {level}")
    print(data)

    classes = data[target].unique()

    if len(classes) == 1:
        print(indent + f"All rows belong to one class -> Leaf = {classes[0]}")
        return classes[0]

    if len(features) == 0:
        majority = data[target].mode()[0]
        print(indent + f"No features left -> Leaf = {majority}")
        return majority

    total_entropy = entropy(data, target)
    print(indent + f"Entropy({target}) = {round(total_entropy, 4)}")

    gains = {}
    for feature in features:
        gains[feature] = information_gain(data, feature, target)
        print(indent + f"Information Gain({feature}) = {round(gains[feature], 4)}")

    best = max(gains, key=gains.get)
    print(indent + f"Best Feature Selected = {best}")

    tree = {best: {}}
    remaining_features = [f for f in features if f != best]

    for value in data[best].unique():
        print("\n" + indent + f"Split where {best} = {value}")
        subset = data[data[best] == value].drop(columns=[best])

        if subset.empty:
            majority = data[target].mode()[0]
            print(indent + f"Empty subset -> Leaf = {majority}")
            tree[best][value] = majority
        else:
            tree[best][value] = id3_steps(subset, remaining_features, target, level + 1)

    return tree

def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "Class =", tree)
        return

    for feature, branches in tree.items():
        print(indent + feature)
        for value, subtree in branches.items():
            print(indent + "├── " + str(value))
            print_tree(subtree, indent + "    ")

print("Dataset")
print(df)

target = "Play"
features = [c for c in df.columns if c != target]

print("\nBuilding ID3 Tree Step by Step")
tree = id3_steps(df, features, target)

print("\nFinal ID3 Tree")
print(tree)

print("\nReadable Tree")
print_tree(tree)

X = pd.get_dummies(df[features])
y = df[target]

model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X, y)

tree_graph = graphviz.Source(
    export_graphviz(
        model,
        out_file=None,
        feature_names=X.columns,
        class_names=model.classes_,
        filled=True,
        rounded=True
    )
)

tree_graph.render("id3_tennis_tree", format="png", cleanup=True)
print("\nVisual tree saved as id3_tennis_tree.png")
