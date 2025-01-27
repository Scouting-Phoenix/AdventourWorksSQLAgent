# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 23:18:22 2025

@author: d_sch
"""

import pyodbc
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer




conn_str_mssql = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER=localhost\\SQLEXPRESS;DATABASE=AdventureWorks2016;Trusted_Connection=yes;'

# Your connection params here



# Set up the database connection


def calculate_embedding(df, col):
    embedding_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

    texts=df[col]
    embeddings=[]
    
    df['embedding'] = df[col].apply(lambda x: embedding_model.encode([x])[0].tolist())

    # for text in texts:
    #     embedding = embedding_model.encode([text])[0].tolist()
    #     #embeddings = pd.concat([embeddings,pd.DataFrame(embedding)],axis=0)    
    #     embeddings.append(embedding)
    
    return df


def save_embedding(target_table, df):
    
    MY_DB_HOST = 'localhost'
    MY_DB_PORT = 5433
    MY_DB_NAME = 'AdventureWorks_SQLAgent'
    MY_DB_USER = 'postgres'
    MY_DB_PASSWORD =os.getenv('PostGresAdminPW')
    
    # conn = psycopg2.connect(
    #     host=MY_DB_HOST,
    #     port=MY_DB_PORT,
    #     dbname=MY_DB_NAME,
    #     user=MY_DB_USER,
    #     password=os.getenv('PostGresAdminPW')
    # )
    
    # conn.autocommit = True
    
    # #df.to_sql(target_table, conn, if_exists= 'replace') - not yet supported`!
    #conn1.commit() 
    #conn.close()    
    
    
    engine = create_engine(f'postgresql+psycopg2://{MY_DB_USER}:{MY_DB_PASSWORD}@{MY_DB_HOST}:{MY_DB_PORT}/{MY_DB_NAME}')
    df.to_sql(target_table, engine, if_exists='replace', index=False)
    
    
    
    #Todo: Create Index to optimize similarity search

    print("Embeddings saved to table " +target_table)     
    return True
        
        
        
        
        
###test
data = {
    'ID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Description': [
        'Alice is a data scientist.',
        'Bob is a software engineer.',
        'Charlie is a machine learning expert.',
        'David is a software developer.'
    ]
}

# Create the DataFrame
df = pd.DataFrame(data)        

result=calculate_embedding(df,'Description')



save_embedding('Test',df[['Description','embedding']])
