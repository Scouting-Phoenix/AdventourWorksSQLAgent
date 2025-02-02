
import os
import pyodbc
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
#from transformers import AutoModelForCausalLM, AutoTokenizer
#from huggingface_hub import login
import openai

conn_str_mssql = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER=localhost\\SQLEXPRESS;DATABASE=AdventureWorks2016;Trusted_Connection=yes;'


MY_DB_HOST = 'localhost'
MY_DB_PORT = 5433
MY_DB_NAME = 'AdventureWorks_SQLAgent'
MY_DB_USER = 'postgres'
#MY_DB_PASSWORD =os.getenv('PostGresAdminPW')


embedding_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')


def table_similarity(embedding):
    
    conn = psycopg2.connect(host=MY_DB_HOST,port=MY_DB_PORT,dbname=MY_DB_NAME,user=MY_DB_USER,password=os.getenv('PostGresAdminPW'))
    cur = conn.cursor()
    
    similarity_limit=0.95
    top_n_limit=4
    
    
    query = '''
    WITH top_results AS (
        SELECT 
            "SchemaName",
            "TableName",
            "Description",
            1 - (embedding <=> %s::vector(768)) AS cosine_similarity
        FROM 
            tables
        ORDER BY 
            cosine_similarity DESC 
        LIMIT %s
    ), filtered_results AS (
        SELECT 
            "SchemaName",
            "TableName",
            "Description",
            1 - (embedding <=> %s::vector(768)) AS cosine_similarity
        FROM 
            tables
        WHERE 
            1 - (embedding <=> %s::vector(768)) > %s
    )
        
    SELECT DISTINCT * FROM top_results
    UNION 
    SELECT DISTINCT * FROM filtered_results;   
    '''

    
    cur.execute(query,(embedding,top_n_limit,embedding,embedding,similarity_limit ))
    
    result=cur.fetchall()
    cur.close()
    conn.close()    
    df = pd.DataFrame(result, columns=['SchemaName', 'TableName', 'Description', 'cosine_similarity'])
    print('Similarity search for tables finished')
    
    return df


def colume_similarity(embedding):
    
    conn = psycopg2.connect(host=MY_DB_HOST,port=MY_DB_PORT,dbname=MY_DB_NAME,user=MY_DB_USER,password=os.getenv('PostGresAdminPW'))
    cur = conn.cursor()
    
    similarity_limit=0.95
    top_n_limit=4
    
    
    query = '''
    WITH top_results AS (
        SELECT 
            "SchemaName",
            "TableName",
            "ColumnName",
            "Description",
            1 - (embedding <=> %s) AS cosine_similarity
        FROM 
            columns
        ORDER BY 
            cosine_similarity DESC 
        LIMIT %s
    ), filtered_results AS (
        SELECT 
            "SchemaName",
            "TableName",
            "ColumnName",            
            "Description",
            1 - (embedding <=> %s) AS cosine_similarity
        FROM 
            columns
        WHERE 
            1 - (embedding <=> %s) > %s
    )
        
    SELECT DISTINCT * FROM top_results
    UNION 
    SELECT DISTINCT * FROM filtered_results;   
    '''

    
    cur.execute(query,(embedding,top_n_limit,embedding,embedding,similarity_limit) )
    
    result=cur.fetchall()
    cur.close()
    conn.close()    
    df = pd.DataFrame(result, columns=['SchemaName', 'TableName','ColumnName', 'Description', 'cosine_similarity'])
    print('Similarity search for tables finished')    
    
    return df



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
    WITH results AS (
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
            COALESCE(o.type_desc, 'N/A') AS ObjectType,
            t.name AS DataType  -- Adding DataType
        FROM sys.extended_properties ep
        LEFT JOIN sys.objects o 
            ON ep.major_id = o.object_id
        LEFT JOIN sys.columns c 
            ON ep.minor_id = c.column_id AND c.object_id = o.object_id
        LEFT JOIN sys.indexes i 
            ON ep.major_id = i.object_id
        LEFT JOIN sys.foreign_keys fk 
            ON ep.major_id = fk.parent_object_id
        LEFT JOIN sys.types t   -- Join to get data type
            ON c.user_type_id = t.user_type_id
        WHERE
            SCHEMA_NAME(o.schema_id) != 'dbo' 
            AND SCHEMA_NAME(o.schema_id) IS NOT NULL
            AND c.name IS NOT NULL
    )
    SELECT DISTINCT 
        SchemaName, 
        ObjectName AS TableName, 
        ColumnName, 
        CONVERT(NVARCHAR(MAX), ColumnName) + ' ' + CONVERT(NVARCHAR(MAX), PropertyValue) AS [Description],
        DataType  -- Add DataType to the SELECT statement
    FROM results 
    WHERE LevelType = 'TABLE';
    '''

    conn = pyodbc.connect(conn_str_mssql)
    
    result=pd.read_sql(sql, conn)
    
    conn.close()
    
    return result
        
      

def find_paths(network,all_tables):
    # here we calculate the minimal spanning tree in the networt
    #ToDo: Fix output, currently not output is created, something is wrong....
    print("---------------------------------------------------------------------------------------")
    print("debugg Network search")
    import networkx as nx
    network['Weight']=1
    
    print("debugg network")
    print(network[['ParentTableObjectId','ReferencedTableObjectId']])
    
    edges = list(network[['ParentTableObjectId','ReferencedTableObjectId','Weight']].itertuples(index=False, name=None))
    
    print("edges")
    print(edges)

    # Create a graph (undirected)
    G = nx.Graph()
    
    points = all_tables['ObjectId'].tolist()
    print('Points')
    print(points)
    mst = nx.minimum_spanning_tree(G)
    subnetwork_edges = list(mst.edges())
    unique_values = set([item for sublist in subnetwork_edges for item in sublist])
    neded_ids=pd.DataFrame(unique_values, columns=['ObjectId'])
    
    sql = """
    SELECT 
        s.name AS SchemaName,
        t.name AS TableName,
        t.object_id AS ObjectId
    FROM 
        sys.schemas AS s
    JOIN 
        sys.tables AS t ON t.schema_id = s.schema_id

    WHERE 
        s.name <> 'dbo'
    ORDER BY 
        s.name, t.name;
    """    
    
    conn = pyodbc.connect(conn_str_mssql)
    
    # Read query result into a DataFrame
    df_sql = pd.read_sql(sql, conn)    
    conn.close()    
    
    results=pd.merge(df_sql, neded_ids, on='ObjectId', how='inner')[['SchemaName','TableName']].drop_duplicates()
    print("debugg Network search finisehd")    
    print("---------------------------------------------------------------------------------------")
    return results
    
    
    
    
    
#ToDo: Optimize to find all dependend tables
def get_dependent_tables(tables,columns):

    sql='''
    SELECT 
        fk.name AS ForeignKeyName,
        fk.object_id AS ForeignKeyObjectId,
        tp.name AS ParentTable,
        tp.object_id AS ParentTableObjectId,
        cp.name AS ParentColumn,
        cp.column_id AS ParentColumnObjectId,
        tr.name AS ReferencedTable,
        tr.object_id AS ReferencedTableObjectId,
        cr.name AS ReferencedColumn,
        cr.column_id AS ReferencedColumnObjectId
    FROM 
        sys.foreign_keys AS fk
    INNER JOIN 
        sys.foreign_key_columns AS fkc 
        ON fk.object_id = fkc.constraint_object_id
    INNER JOIN 
        sys.tables AS tp 
        ON fk.parent_object_id = tp.object_id
    INNER JOIN 
        sys.columns AS cp 
        ON fkc.parent_column_id = cp.column_id 
        AND fkc.parent_object_id = cp.object_id
    INNER JOIN 
        sys.tables AS tr 
        ON fk.referenced_object_id = tr.object_id
    INNER JOIN 
        sys.columns AS cr 
        ON fkc.referenced_column_id = cr.column_id 
        AND fkc.referenced_object_id = cr.object_id
    ORDER BY 
        ParentTable, ForeignKeyName;
    '''    
    conn = pyodbc.connect(conn_str_mssql)
    
    # Execute SQL query and store result in DataFrame
    network = pd.read_sql(sql,conn)
    
    conn.close()
    
    
    sql = """
    SELECT 
        s.name AS SchemaName,
        t.name AS TableName,
        t.object_id AS ObjectId,
        c.name AS ColumnName
    FROM 
        sys.schemas AS s
    JOIN 
        sys.tables AS t ON t.schema_id = s.schema_id
    JOIN 
        sys.columns AS c ON c.object_id = t.object_id
    WHERE 
        s.name <> 'dbo'
    ORDER BY 
        s.name, t.name, c.name;
    """    
    
    conn = pyodbc.connect(conn_str_mssql)
    
    # Read query result into a DataFrame
    df_sql = pd.read_sql(sql, conn)    
    conn.close()

    tables_from_columns=columns.merge(df_sql, on=['SchemaName','TableName'], how='inner')
    

   
    tables=tables.merge(df_sql,  on=['SchemaName','TableName'], how='inner')
    tables=tables[['SchemaName','TableName','ObjectId']]
    tables=tables.drop_duplicates()

    
    all_tables = pd.concat([tables_from_columns[['SchemaName','TableName','ObjectId']],tables[['SchemaName','TableName','ObjectId']] ],ignore_index=True).drop_duplicates()
    
    print("debugg network")
    print(network)
    print("debugg all_tables")

    print(all_tables)
    result=find_paths(network,all_tables)
    
    
    return result



#ToDo: Optimising formating??
def format_table_promt(tables):
    
    column_extension = extended_colume_properties() 
    table_extension = extended_table_property()
    
    
    #ToDo: Join extended property with input tables
    table_prompt=''
    for index, row in table_extension.iterrows():
        table_prompt = table_prompt + "Tablename: " + row['SchemaName'] +"." +    row['TableName'] + row['Description'] +  " with the following columns \n"
        for colindex, colrow in column_extension.loc[ (column_extension['SchemaName'] == row['SchemaName']) & (column_extension['TableName'] == row['TableName']) ].iterrows(): 
            table_prompt =table_prompt+  colrow['ColumnName']+ ", "+ colrow['DataType']+ ", "+ colrow['Description'] +"; \n"
        
    
    return table_prompt
        
    
    
    
    
    

def build_promt_question(question):
    embedding=embedding_model.encode([question])[0].tolist()
    
    # Convert the embedding to a PostgreSQL vector-compatible format
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
    
    tables=table_similarity(embedding_str)
    print('Similar tables')
    pd.set_option('display.max_columns', None)
    print(tables)
    columns=colume_similarity(embedding_str)
    print('Similar columns')
    print(columns)
    
    tables=get_dependent_tables(tables,columns)

    
    print("Relevant tables")
    print(tables)
    
    formated_tables=format_table_promt(tables)
    #format_columne_promt()
    
    promt='''
            Please translate the following question seperated by " into an SQL Query.
            The Database is an MS SQL Databaes Version 2022.
            "
        '''+question+'''"
        The following tables are relevant
        '''+formated_tables
        
    
    return promt




#ToDO: Refactor
# def extract_SQL(LLM_response): 


def call_LLM(promt):
    # login(HF_token)

    # model_name = "meta-llama/Llama-2-7b-chat-hf"   # Use appropriate LLaMA 3 model size
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # inputs = tokenizer(prompt, return_tensors="pt")
    #  # Generate output
    # #output = model.generate(**inputs, max_length=200)
    # output = model.generate(**inputs, max_length=200)

    
    # # Decode and print the result
    # response = tokenizer.decode(output[0], skip_special_tokens=True)   
    client = openai.OpenAI(api_key=os.getenv('OpenAPIKey'))

    response = client.chat.completions.create(
         model='gpt-4o-mini',
         messages=[
             {'role': 'system', 'content': promt}
         ],
         temperature=0,
    )
    return response.choices[0].message.content 



# def execute_SQL():
    
#     conn = pyodbc.connect(conn_str_mssql)
    
#     result=pd.read_sql(sql, conn)
    
#     conn.close()
    
#     return result

def RAG(question):
    
    promt=build_promt_question(question)
    #print(promt)
    
    result=call_LLM(promt)
    print("iniial response")
    print(result)
    
    
    print(promt[0:1000])
    promt='''
        From the following text seperated by " only return the sql code which can be directly executed on MSSQL "
    ''' + result +'"'
    
    print('SQL-Query')
    sql=call_LLM(promt)
    sql=sql.replace("```", "").replace("sql", "")
    print('SQL-Query')
    print(sql)
    print('SQL Result')
    try:
        # Establish connection
        conn = pyodbc.connect(conn_str_mssql)
    
        # Execute SQL query and store result in DataFrame
        result = pd.read_sql(sql, conn)
    
        # Display results
        print("Data retrieved successfully.")
        print(result)
    except pyodbc.DatabaseError as e:
        print(f"Database error occurred: {e}")
    finally:
        conn.close()
        print("Database connection closed.")    
    
    
    return True



question ="Which are the top 5 customers who have spent the most on orders in the last year?"
RAG(question)


question="Which employees have generated the most sales revenue, and how much have they contributed?"
RAG(question)

question = "For every region show me the employes with an order fullfillment time which is one standard deviation worse then the regions average"
RAG(question)

question="How much revenue has each employee generated in the North America territory?"
RAG(question)

question="Calculate month over month change of sales and inventory value. In whitch month was the inventory change relative to the sales change maximal and minimal?"
RAG(question)
