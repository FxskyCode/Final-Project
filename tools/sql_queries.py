from config.sql_connection import engine
import pandas as pd

def get_everything ():
    query = """SELECT * FROM dummy;"""
    df = pd.read_sql_query(query, engine)
    return df.to_dict(orient="records")

def get_top_valuation ():
    query = f"""SELECT * FROM dummy.dummy
	ORDER BY Valuation DESC
    LIMIT 5;"""
    df = pd.read_sql_query(query, engine)
    return df.to_dict(orient="records")

