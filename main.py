from  solutionAgent import *
from docSearchAgent import MySQLDocSearchAgent;
from dBAnalysisAgent import MySQLDBAnalysisAgent;
from solutionAgent import MySQLSolutionAgent;
from assistantWorkflow import MySQLAssistantWorkflow;

from prompt import SYSTEM_INSTRUCTIONS

def run_test_cases():
    
    workflow = MySQLAssistantWorkflow(
        model="mistral", 
        embedding_model="nomic-embed-text",
        docs_folder="sql_docs"
    )
    
    connection_params = {
        "host": "localhost",
        "user": "root",
        "password": "root",
        "database": "multiAgent",
        "port": 3306
    }
    
    # # Test case 1: Max Connections Error
    # print("\n========== TEST CASE 1: Max Connections Error ==========")
    # query1 = "Why are my max connections getting an error?"
    # response1 = workflow.process_query(query1, connection_params)
    # print("\nFINAL RESPONSE:")
    # print(response1)
    
    # # Test case 2: Slow Query
    # print("\n========== TEST CASE 2: Slow Query ==========")
    # query2 = "Why is my query running slowly?"
    # response2 = workflow.process_query(query2, connection_params)
    # print("\nFINAL RESPONSE:")
    # print(response2)
    
    # # Test case 3: Database Connection Error
    # print("\n========== TEST CASE 3: Disk Space ==========")
    # query3 = "Deleting data from mysql tables, disk space is still not freed up?"
    # response3 = workflow.process_query(query3, connection_params)
    # print("\nFINAL RESPONSE:")
    # print(response3)

    # Test case 4: Data Consistency
    print("\n========== TEST CASE 4: InnoDB Online DDL Error ==========")
    query4 = "Online DDL error: innodb_online_alter_log_max_size exceeded?"
    response4 = workflow.process_query(query4, connection_params)
    print("\nFINAL RESPONSE:")
    print(response4)
    
    # Test case 6: Replication Lag
    print("\n========== TEST CASE 6: Replication Lag ==========")
    query6 = "Why is my MySQL replication showing high lag time?"
    response6 = workflow.process_query(query6, connection_params)
    print("\nFINAL RESPONSE:")
    print(response6)


if __name__ == "__main__":
    try:
        run_test_cases()
    except Exception as e:
        print(f"Error running test cases: {str(e)}")
        print(traceback.format_exc())