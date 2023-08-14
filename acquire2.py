import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from env import get_connection

def get_db_connection(database):
    return get_connection(database)

# Titanic Data

def new_titanic_data():
    sql_query = 'SELECT * FROM passengers'
    df = pd.read_sql(sql_query, get_db_connection('titanic_db'))
    return df

def get_titanic_data():
    if os.path.isfile('titanic_df.csv'):
        df = pd.read_csv('titanic_df.csv', index_col=0)
    else:  
        df = new_titanic_data()
        df.to_csv('titanic_df.csv')
    return df

# Iris Data

def new_iris_data():
    sql_query = "SELECT species_id, species_name, sepal_length, sepal_width, petal_length, petal_width FROM measurements JOIN species USING(species_id)"
    df = pd.read_sql(sql_query, get_db_connection('iris_db'))
    return df

def get_iris_data():
    if os.path.isfile('iris_df.csv'):
        df = pd.read_csv('iris_df.csv', index_col=0)
    else:   
        df = new_iris_data()
        df.to_csv('iris_df.csv')
    return df

# Telco Data

def new_telco_data():
    sql_query = "SELECT * FROM customers JOIN contract_types USING (contract_type_id) JOIN internet_service_types USING (internet_service_type_id) JOIN payment_types USING (payment_type_id)"
    df = pd.read_sql(sql_query, get_db_connection('telco_churn'))
    return df

def get_telco_data():
    if os.path.isfile('telco.csv'):
        df = pd.read_csv('telco.csv', index_col=0)
    else:
        df = new_telco_data()
        df.to_csv('telco.csv')
    return df

# Seed for random operations
seed = 42

# Load and split Titanic data
titanic_df = get_titanic_data()
train_titanic, val_test_titanic = train_test_split(titanic_df, train_size=0.7, random_state=seed, stratify=titanic_df['survived'])
val_titanic, test_titanic = train_test_split(val_test_titanic, train_size=0.5, random_state=seed, stratify=val_test_titanic['survived'])

# Load and split Iris data
iris_df = get_iris_data()
train_iris, val_test_iris = train_test_split(iris_df, train_size=0.7, random_state=seed, stratify=iris_df['species'])
val_iris, test_iris = train_test_split(val_test_iris, train_size=0.5, random_state=seed, stratify=val_test_iris['species'])

# Load and split Telco data
telco_df = get_telco_data()
train_telco, val_test_telco = train_test_split(telco_df, train_size=0.7, random_state=seed, stratify=telco_df['churn'])
val_telco, test_telco = train_test_split(val_test_telco, train_size=0.5, random_state=seed, stratify=val_test_telco['churn'])

# Calculate value counts for each target variable
train_titanic['survived'].value_counts(normalize=True)
val_titanic['survived'].value_counts(normalize=True)
test_titanic['survived'].value_counts(normalize=True)

train_iris['species'].value_counts(normalize=True)
val_iris['species'].value_counts(normalize=True)
test_iris['species'].value_counts(normalize=True)

train_telco['churn'].value_counts(normalize=True)
val_telco['churn'].value_counts(normalize=True)
test_telco['churn'].value_counts(normalize=True)
