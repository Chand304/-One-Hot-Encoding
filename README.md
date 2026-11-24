One-Hot Encoding

A beginner-friendly practice project on One-Hot Encoding: converting text (categorical) columns into 0/1 numeric columns so a machine learning model can use them. Practiced with small toy examples and the Titanic dataset.

What is One-Hot Encoding?

Machine learning models only understand numbers. One-Hot Encoding gives every unique value of a column its own new column. A 1 means "this row has this value" and a 0 means "it does not".

Before:                After:
city                   city_Karachi  city_Lahore  city_Multan
Lahore            ->         0            1            0
Karachi           ->         1            0            0
Multan            ->         0            0            1
Lahore            ->         0            1            0

Each row has exactly one 1, which is where the name "one-hot" comes from.

Why not just use numbers like 0, 1, 2?

If Karachi = 0, Lahore = 1, Multan = 2, the model assumes Multan is "bigger" than Lahore, and Lahore is "bigger" than Karachi. Cities have no such order. One-Hot Encoding avoids this fake order because every category gets its own column and all are treated equally.

When to use it
Situation	Use One-Hot?
Categories have no order (city, color, brand, port)	Yes
Categories have a real order (Small < Medium < Large)	No, use Ordinal Encoding
Target column (Yes/No)	No, use Label Encoding
Hundreds of unique values (like zip codes)	No, it creates too many columns
Method 1: pandas get_dummies

The simplest way. It works directly on the DataFrame and returns a table.

python
import pandas as pd

df = pd.DataFrame({
    "brand": ["Samsung", "Apple", "Xiaomi", "Apple"],
    "price": [30000, 150000, 45000, 140000]
})

df = pd.get_dummies(df, columns=["brand"], dtype=int)
print(df)

Output:

    price  brand_Apple  brand_Samsung  brand_Xiaomi
0   30000            0              1             0
1  150000            1              0             0
2   45000            0              0             1
3  140000            1              0             0
columns=["brand"] tells pandas which column to encode.
The original brand column is replaced by the new columns, named column_value.
dtype=int gives 1/0. Without it, newer pandas versions return True/False, which means the same thing.
New columns are created in alphabetical order of the category names.
Save the result with df = ..., otherwise the DataFrame does not change.
Method 2: scikit-learn OneHotEncoder

The scikit-learn way. It returns a NumPy array instead of a table.

python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame({
    "city": ["Lahore", "Karachi", "Multan", "Lahore"]
})

ohe = OneHotEncoder(sparse_output=False)
result = ohe.fit_transform(df[["city"]])

print(result)
print(ohe.get_feature_names_out())

Output:

[[0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]
 [0. 1. 0.]]
['city_Karachi' 'city_Lahore' 'city_Multan']
OneHotEncoder(sparse_output=False) creates the encoder. sparse_output=False returns a normal array that is easy to read.
fit_transform(df[["city"]]) learns the categories and converts them. Double brackets are needed because the encoder expects a table (2D), not a single list.
get_feature_names_out() shows which column belongs to which category. The () is required, without it Python only prints the method itself.
Titanic example

Applied to the Embarked column (ports S, C and Q) of the Titanic dataset.

python
import pandas as pd

df = pd.read_csv("titanic.csv")

# Fill the 2 missing values first
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df = pd.get_dummies(df, columns=["Embarked"], dtype=int)
print(df.columns)

Embarked is replaced by Embarked_C, Embarked_Q and Embarked_S. A passenger who boarded at S has Embarked_S = 1 and the other two are 0.

Mistakes I ran into
Mistake	What happened	Fix
Encoding before filling missing values	sklearn treated NaN as its own category and created an extra 4th column	Use fillna first, then encode
ohe.get_feature_names_out without ()	Printed the method instead of the column names	Add ()
Running get_dummies twice	KeyError, because the original column was already removed	Reload the data and run it once
df["city"] instead of df[["city"]] in OneHotEncoder	Error, the encoder needs a 2D input	Use double brackets
Key takeaways
One-Hot Encoding is for categories with no order.
Each unique value becomes its own 0/1 column, and each row has a single 1.
pd.get_dummies is the quick way, OneHotEncoder is the scikit-learn way.
Always handle missing values before encoding.
Tech stack

Python, pandas, scikit-learn, Jupyter Notebook
