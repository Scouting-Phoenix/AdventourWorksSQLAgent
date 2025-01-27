

def main():
    '''
        Credits to https://www.computerwoche.de/article/3803224/postgresql-als-rag-vektordatenbank-nutzen.html
        Set environment variable in Windows in cmd: setx PostGresAdminPW "Your_PW"
        Call ENV Variable with os.getenv('PostGresAdminPW')
        pip install psycopg2
    
    
    '''
    import psycopg2
    import os
    
    
    
    
    # Your connection params here
    
    MY_DB_HOST = 'localhost'
    MY_DB_PORT = 5433 # I use this port for Postgres V16
    MY_DB_NAME = 'AdventureWorks_SQLAgent'
    MY_DB_USER = 'postgres'
    
    # Set up the database connection
    
    conn = psycopg2.connect(
        host=MY_DB_HOST,
        port=MY_DB_PORT,
        dbname=MY_DB_NAME,
        user=MY_DB_USER,
        password=os.getenv('PostGresAdminPW')
    )
    cur = conn.cursor()
    
    
    
    
    
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
    conn.commit()

    
    cur.execute('''
        DROP TABLE IF EXISTS tables;
        	CREATE TABLE  tables (
        	id TEXT,
        "SchemaName" TEXT,
        	"TableName" TEXT,
        "Description" TEXT,
        	embedding VECTOR(768)
    	);
        ''')
        
        
    
    cur.execute('''
        DROP TABLE IF EXISTS columns;
    	CREATE TABLE  columns (
        	id TeXT,
        	"SchemaName" TEXT,
        "TableName" TEXT,
        "ColumnName" TEXT,
        "Description" TEXT,
        	embedding VECTOR(768)
    	);
        ''')
        

    cur.execute('''
        DROP TABLE IF EXISTS test;
    	CREATE TABLE Test (
        	id TEXT,
        	embedding VECTOR(768)
    	);
        ''')
        
    conn.commit()
    conn.close()
    
main()