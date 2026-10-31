# Ex.No-3b-DATA PREPROCESSING
## Aim
To perform data preprocessing on a dataset using Python and Scikit-learn by handling missing values, encoding categorical data, splitting the dataset, and applying feature scaling. 
## Procedure
1.	Import the required Python libraries. 
2.	Mount Google Drive and load the dataset using Pandas.
3.	Display the first few records of the dataset. 
4.	Inspect the dataset using df.info() and df.shape. 
5.	Separate the independent variables (X) and dependent variable (Y). 
6.	Convert the independent variables into an array. 
7.	Identify and handle missing values using SimpleImputer with the mean strategy. 
8.	Encode the categorical Country column using LabelEncoder. 
9.	Apply One-Hot Encoding to convert categorical country values into dummy variables. 
10.	Encode the dependent variable Purchased using LabelEncoder. 
11.	Split the dataset into training and testing sets using train_test_split. 
12.	Apply StandardScaler for feature scaling. 
13.	Display the preprocessed training and testing datasets.
### Program
    from google.colab import drive
    drive.mount('/content/drive')
    import pandas as pd
    df = pd.read_csv('/content/drive/My Drive/Data.csv')
    df.head()


    print("Dataset Information:")
    df.info()

    print("\nDataset Shape:")
    print(df.shape)


    x = df[['Country', 'Age', 'Salary']]
    y = df[['Purchased']].values

    x = df[['Country', 'Age', 'Salary']].values

    print("Independent Variable X:")
    print(x)

    print("\nDependent Variable Y:")
    print(y)


    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(
    missing_values=np.nan,
    strategy='mean'
    )

    imputer.fit(x[:, 1:3])

    x[:, 1:3] = imputer.transform(x[:, 1:3])

    print("After Handling Missing Values:")
    print(x)


    from sklearn.preprocessing import LabelEncoder

    label_encoder_x = LabelEncoder()

    x[:, 0] = label_encoder_x.fit_transform(x[:, 0])

    print("After Label Encoding:")
    print(x)


    from sklearn.preprocessing import OneHotEncoder

    onehotencoder = OneHotEncoder()

    x_country = onehotencoder.fit_transform(
    df.Country.values.reshape(-1, 1)
    ).toarray()

    print("One-Hot Encoded Country:")
    print(x_country)
