


import pyodbc
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.types import UserDefinedType

from sentence_transformers import SentenceTransformer

conn_str_mssql = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER=localhost\\SQLEXPRESS;DATABASE=AdventureWorks2016;Trusted_Connection=yes;'


MY_DB_HOST = 'localhost'
MY_DB_PORT = 5433
MY_DB_NAME = 'AdventureWorks_SQLAgent'
MY_DB_USER = 'postgres'
MY_DB_PASSWORD =os.getenv('PostGresAdminPW')

conn_str_postgres=f'postgresql+psycopg2://{MY_DB_USER}:{MY_DB_PASSWORD}@{MY_DB_HOST}:{MY_DB_PORT}/{MY_DB_NAME}'

def calculate_embedding(df, col):
    embedding_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    
    df['embedding'] = df[col].apply(lambda x: embedding_model.encode([x])[0].tolist())

    # for text in texts:
    #     embedding = embedding_model.encode([text])[0].tolist()
    #     #embeddings = pd.concat([embeddings,pd.DataFrame(embedding)],axis=0)    
    #     embeddings.append(embedding)
    
    return df


class Vector(UserDefinedType):
    def get_col_spec(self):
        return "VECTOR(768)" 
    

def save_embedding(target_table, df):

    
    engine = create_engine(conn_str_postgres)
    df.to_sql(target_table, engine, if_exists='append', index=False, dtype={'embedding':  Vector() })  # Explicitly define the vector column type if necessary

    
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









def extended_table_property():
    #returns extendet property information on a table level
    
    
    
    sql = '''
    
    with results as (
    SELECT 
        ep.name AS PropertyName,
        ep.value AS PropertyValue,
        CASE ep.class
            WHEN 0 THEN 'DATABASE'
            WHEN 1 THEN 
                CASE 
                    WHEN o.type_desc = 'USER_TABLE' THEN 'TABLE'
                    WHEN o.type_desc = 'VIEW' THEN 'VIEW'
                    WHEN c.name IS NOT NULL THEN 'COLUMN'
                    WHEN i.object_id IS NOT NULL THEN 'INDEX'
                    WHEN fk.object_id IS NOT NULL THEN 'FOREIGN KEY'
                    ELSE 'OTHER'
                END
            WHEN 2 THEN 'PARAMETER'
            ELSE 'OTHER'
        END AS LevelType,
        SCHEMA_NAME(o.schema_id) AS SchemaName,
        o.name AS ObjectName,
        c.name AS ColumnName,
        COALESCE(o.type_desc, 'N/A') AS ObjectType
    FROM sys.extended_properties ep
    LEFT JOIN sys.objects o 
        ON ep.major_id = o.object_id
    LEFT JOIN sys.columns c 
        ON ep.minor_id = c.column_id AND c.object_id = o.object_id
    LEFT JOIN sys.indexes i 
        ON ep.major_id = i.object_id
    LEFT JOIN sys.foreign_keys fk 
        ON ep.major_id = fk.parent_object_id
    where
    	  SCHEMA_NAME(o.schema_id) !='dbo' 
    	and   SCHEMA_NAME(o.schema_id) IS NOT NULL
    ) 
    SELECT DISTINCT SchemaName, ObjectName AS TableName, CONVERT(NVARCHAR(MAX), SchemaName )+' '+ CONVERT(NVARCHAR(MAX), ObjectName) +' ' +CONVERT(NVARCHAR(MAX), PropertyValue ) as [Description] FROM results where LevelType='TABLE' and ColumnName is null
    '''
    
    
    conn = pyodbc.connect(conn_str_mssql)
    
    result=pd.read_sql(sql, conn)
    
    conn.close()
    
    return result


def extended_colume_properties():
    #returns extendet property information on a column level
    
    
    sql='''
    with results as (
    SELECT 
        ep.name AS PropertyName,
        ep.value AS PropertyValue,
        CASE ep.class
            WHEN 0 THEN 'DATABASE'
            WHEN 1 THEN 
                CASE 
                    WHEN o.type_desc = 'USER_TABLE' THEN 'TABLE'
                    WHEN o.type_desc = 'VIEW' THEN 'VIEW'
                    WHEN c.name IS NOT NULL THEN 'COLUMN'
                    WHEN i.object_id IS NOT NULL THEN 'INDEX'
                    WHEN fk.object_id IS NOT NULL THEN 'FOREIGN KEY'
                    ELSE 'OTHER'
                END
            WHEN 2 THEN 'PARAMETER'
            ELSE 'OTHER'
        END AS LevelType,
        SCHEMA_NAME(o.schema_id) AS SchemaName,
        o.name AS ObjectName,
        c.name AS ColumnName,
        COALESCE(o.type_desc, 'N/A') AS ObjectType
    FROM sys.extended_properties ep
    LEFT JOIN sys.objects o 
        ON ep.major_id = o.object_id
    LEFT JOIN sys.columns c 
        ON ep.minor_id = c.column_id AND c.object_id = o.object_id
    LEFT JOIN sys.indexes i 
        ON ep.major_id = i.object_id
    LEFT JOIN sys.foreign_keys fk 
        ON ep.major_id = fk.parent_object_id
    where
    	  SCHEMA_NAME(o.schema_id) !='dbo' 
    	and   SCHEMA_NAME(o.schema_id) IS NOT NULL
		and c.name  IS NOT NULL
    )
    SELECT DISTINCT SchemaName, ObjectName AS TableName, ColumnName, CONVERT(NVARCHAR(MAX), ColumnName )+' ' +CONVERT(NVARCHAR(MAX), PropertyValue ) as [Description] FROM results where LevelType='TABLE' 
    
    '''

    conn = pyodbc.connect(conn_str_mssql)
    
    result=pd.read_sql(sql, conn)
    
    conn.close()
    
    return result
        
        
        
def main():
    
    tables=extended_table_property()
    df=calculate_embedding(tables,'Description')
        
    df['id'] =df['SchemaName'] + df['TableName']
    
    save_embedding('tables', df[['id','SchemaName','TableName','Description','embedding']])
 
    
 
    columns=extended_colume_properties()
    df=calculate_embedding(columns,'Description')
    
    
    df['id'] =df['SchemaName'] + df['TableName']+ df['ColumnName']
    
    save_embedding('columns', df[['id','SchemaName','TableName','ColumnName','Description','embedding']])
    
main()