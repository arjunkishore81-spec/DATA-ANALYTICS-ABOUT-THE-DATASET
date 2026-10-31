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

    labelencoder_y = LabelEncoder()

    y = labelencoder_y.fit_transform(y)

    print("\nEncoded Dependent Variable:")
    print(y)


    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=0
    )

    print("X Training Data:")
    print(x_train)

    print("\nX Testing Data:")
    print(x_test)

    print("\nY Training Data:")
    print(y_train)

    print("\nY Testing Data:")
    print(y_test)

    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)

    x_test = sc_x.transform(x_test)

    print("Scaled X Training Data:")
    print(x_train)

    print("\nScaled X Testing Data:")
    print(x_test)

## OUTPUT
<img width="357" height="246" alt="image" src="https://github.com/user-attachments/assets/e0b92d03-f9d8-45bf-98e9-a17fa88dfa92" />
<img width="422" height="327" alt="image" src="https://github.com/user-attachments/assets/0394e2d0-c63d-4195-96bd-aad1382b2bc1" />
<img width="280" height="522" alt="image" src="https://github.com/user-attachments/assets/ad2e456a-1ad7-4c8f-89e3-28db89af9f5d" />
<img width="412" height="251" alt="image" src="https://github.com/user-attachments/assets/f39d9a18-ffd1-46f1-932e-bc8292923d2c" />
<img width="281" height="311" alt="image" src="https://github.com/user-attachments/assets/44a9e458-a798-4591-9003-2645260d2254" />
<img width="310" height="422" alt="image" src="https://github.com/user-attachments/assets/f2fa933c-d740-4c14-ad02-d5e6931bad17" />
<img width="440" height="291" alt="image" src="https://github.com/user-attachments/assets/fcec91ff-197e-437e-a494-63dfabfde215" />

## Conclusion
Thus, the given dataset was successfully preprocessed by handling missing values, encoding categorical variables, splitting the data into training and testing sets, and performing feature scaling.
