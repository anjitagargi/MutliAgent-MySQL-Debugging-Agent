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
    
    # Test case 1: Max Connections Error
    print("\n========== TEST CASE 1: Max Connections Error ==========")
    query1 = "Why are my max connections getting an error?"
    response1 = workflow.process_query(query1, connection_params)
    print("\nFINAL RESPONSE:")
    print(response1)
    
    # Test case 2: Slow Query
    print("\n========== TEST CASE 2: Slow Query ==========")
    query2 = "Why is my query running slowly?"
    response2 = workflow.process_query(query2, connection_params)
    print("\nFINAL RESPONSE:")
    print(response2)
    

if __name__ == "__main__":
    try:
        run_test_cases()
    except Exception as e:
        print(f"Error running test cases: {str(e)}")
        print(traceback.format_exc())