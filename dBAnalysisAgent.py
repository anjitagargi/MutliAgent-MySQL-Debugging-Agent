import pymysql
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import MultiPromptChain, LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.agents import AgentExecutor, AgentType, Tool, initialize_agent
from langchain.schema.runnable import RunnablePassthrough

from prompt import SYSTEM_INSTRUCTIONS


class MySQLDBAnalysisAgent:
    """
    Agent that connects to MySQL database and performs diagnostic queries
    """
    def __init__(self, model="mistral"):
        """
        Initialize the MySQL DB Analysis Agent.
        
        Args:
            model: LLM model name to use with Ollama
        """
        self.llm = Ollama(model=model)
        self._setup_prompts()

        self.system_instructions = SYSTEM_INSTRUCTIONS
        
    def _setup_prompts(self):
        """Set up the prompt templates for different analysis types."""
        self.analysis_prompt = ChatPromptTemplate.from_template("""
        You are a MySQL database analyst. Based on the following diagnostic information,
        provide an analysis of the database system:

        User query: {query}
        
        Database diagnostic information:
        {context}
        
        DocSearch context: {doc_context}

        Analyze the current database state and identify potential issues or optimizations.
        Focus on performance, configuration, and resource utilization.
        
        If the diagnostics suggest a problem related to the user query, explain the connection.
        If information is asked, directly reply with the information.
                                                                                                         
        Response:
        """)
        
    def connect_to_database(self, host="localhost", user="root", password="root", database="multiAgent", port=3306) -> Optional[pymysql.connections.Connection]:
        try:
            connection_params = {
                "host": host,
                "user": user,
                "password": password,
                "database": database,
                "port": port,
                "connect_timeout": 5,
                "cursorclass": pymysql.cursors.DictCursor
            }
            
            if database:
                connection_params["database"] = database
                
            connection = pymysql.connect(**connection_params)
            return connection
        except Exception as e:
            print(f"Error connecting to MySQL: {str(e)}")
            return None
            
    def execute_query(self, connection, query) -> Tuple[bool, List[Dict[str, Any]], str]:
        try:
            with connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall()
                return True, result, ""
        except Exception as e:
            error_message = f"Error executing query: {str(e)}"
            return False, [], error_message
            
    def collect_diagnostics(self, connection) -> Dict[str, Any]:
        diagnostics = {}
        queries = {
            "database_info": f"""
                SELECT 
                    TABLE_SCHEMA as 'Database',
                    SUM(DATA_LENGTH + INDEX_LENGTH) as 'Size',
                    COUNT(TABLE_NAME) as 'Tables',
                    SUM(TABLE_ROWS) as 'Total Rows'
                FROM information_schema.TABLES 
                WHERE TABLE_SCHEMA = 'multiAgent'
                GROUP BY TABLE_SCHEMA
            """,
            "table_details": """
                SELECT 
                    TABLE_NAME,
                    ENGINE,
                    TABLE_ROWS,
                    DATA_LENGTH + INDEX_LENGTH as 'Size',
                    CREATE_TIME,
                    UPDATE_TIME
                FROM information_schema.TABLES 
                WHERE TABLE_SCHEMA = 'multiAgent'
            """,
            "database_processes": """
                SELECT ID, USER, HOST, DB, COMMAND, TIME, STATE, INFO 
                FROM INFORMATION_SCHEMA.PROCESSLIST 
                WHERE DB = 'multiAgent'
            """,
            "table_indexes": """
                SELECT 
                    TABLE_NAME,
                    INDEX_NAME,
                    COLUMN_NAME,
                    NON_UNIQUE
                FROM information_schema.STATISTICS
                WHERE TABLE_SCHEMA = 'multiAgent'
            """
        }
        
        for name, query in queries.items():
            try:
                success, results, error = self.execute_query(connection, query)
                if success:
                    diagnostics[name] = results
                else:
                    diagnostics[f"{name}_error"] = error
            except Exception as e:
                diagnostics[f"{name}_error"] = str(e)
                
        return diagnostics

    def collect_targeted_diagnostics(self, connection, query_text) -> Dict[str, Any]:
        diagnostics = {}
        # Modified queries to focus on multiAgent database
        if any(word in query_text.lower() for word in ["slow", "performance", "optimization", "tuning"]):
            queries = {
                "slow_queries": """
                    SELECT * FROM mysql.slow_log 
                    WHERE db = 'multiAgent' 
                    ORDER BY start_time DESC LIMIT 10
                """,
                "table_stats": """
                    SELECT * FROM information_schema.TABLE_STATISTICS 
                    WHERE TABLE_SCHEMA = 'multiAgent'
                """,
                "index_stats": """
                    SELECT * FROM information_schema.INDEX_STATISTICS 
                    WHERE TABLE_SCHEMA = 'multiAgent'
                """
            }
        elif any(word in query_text.lower() for word in ["connection", "cannot connect", "access denied"]):
            queries = {
                "user_privileges": """
                    SELECT * FROM information_schema.USER_PRIVILEGES 
                    WHERE GRANTEE LIKE '%multiAgent%'
                """,
                "schema_privileges": """
                    SELECT * FROM information_schema.SCHEMA_PRIVILEGES 
                    WHERE TABLE_SCHEMA = 'multiAgent'
                """
            }
        elif any(word in query_text.lower() for word in ["disk", "space", "storage", "full"]):
            queries = {
                "table_sizes": """
                    SELECT 
                        TABLE_NAME,
                        ROUND((DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024, 2) as 'Size_MB'
                    FROM information_schema.TABLES
                    WHERE TABLE_SCHEMA = 'multiAgent'
                    ORDER BY (DATA_LENGTH + INDEX_LENGTH) DESC
                """
            }
        else:
            queries = {
                "table_summary": """
                    SELECT 
                        TABLE_NAME,
                        ENGINE,
                        TABLE_ROWS,
                        UPDATE_TIME
                    FROM information_schema.TABLES
                    WHERE TABLE_SCHEMA = 'multiAgent'
                """
            }
            
        for name, query in queries.items():
            try:
                success, results, error = self.execute_query(connection, query)
                if success:
                    diagnostics[name] = results
                else:
                    diagnostics[f"{name}_error"] = error
            except Exception as e:
                diagnostics[f"{name}_error"] = str(e)
                
        return diagnostics
    
    def format_diagnostics(self, diagnostics: Dict[str, Any]) -> str:
        formatted_diagnostics = ""
        for key, value in diagnostics.items():
            formatted_diagnostics += f"\n--- {key} ---\n"
            
            if isinstance(value, list):
                if len(value) > 10:
                    # For large result sets, only include a summary
                    formatted_diagnostics += f"(showing first 10 of {len(value)} results)\n"
                    for item in value[:10]:
                        formatted_diagnostics += f"{item}\n"
                else:
                    for item in value:
                        formatted_diagnostics += f"{item}\n"
            else:
                formatted_diagnostics += f"{value}\n"
                
        return formatted_diagnostics
    
    def analyze_diagnostics(self, query: str, diagnostics: Dict[str, Any], doc_context: str = "") -> str:
        # Format diagnostic info for the prompt
        formatted_diagnostics = self.format_diagnostics(diagnostics)
        
        # Prepare doc_context with a limit to avoid exceeding token limits
        trimmed_doc_context = str(doc_context)[:2000] if doc_context else "No documentation context available."
        
        # Create a runnable chain using the new pipe syntax
        chain = self.analysis_prompt | self.llm
        
        # Run the chain with input variables
        response = chain.invoke({
            "query": query,
            "context": formatted_diagnostics,
            "doc_context": trimmed_doc_context
        })
        
        return response.content if hasattr(response, 'content') else str(response)
    
    def process_query(self, query: str, doc_search_result: Dict[str, Any] = None, 
                 connection_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a database query and return analysis results
        """
        try:
            # Connect to database
            connection = self.connect_to_database(**connection_params)
            if not connection:
                return {
                    "success": False,
                    "response": "Failed to connect to database",
                    "diagnostics": {}
                }

            # For information queries, directly query the database
            if any(keyword in query.lower() for keyword in ["what is", "tell me", "show", "how many", "maximum"]):
                if "maximum" in query.lower() and "connections" in query.lower():
                    success, results, error = self.execute_query(
                        connection, 
                        "SHOW VARIABLES LIKE 'max_connections'"
                    )
                    if success:
                        return {
                            "success": True,
                            "response": f"The maximum number of connections allowed is {results[0]['Value']}",
                            "diagnostics": {"max_connections": results}
                        }

            # Collect diagnostic information
            diagnostics = self.collect_diagnostics(connection)
            
            # Get targeted diagnostics based on query
            targeted_diagnostics = self.collect_targeted_diagnostics(connection, query)
            diagnostics.update(targeted_diagnostics)
            
            # Format diagnostics for analysis
            formatted_diagnostics = self.format_diagnostics(diagnostics)
            
            # Analyze diagnostics
            doc_context = doc_search_result.get("response", "") if doc_search_result else ""
            analysis = self.analyze_diagnostics(query, diagnostics, doc_context)
            
            connection.close()
            
            return {
                "success": True,
                "response": analysis,
                "diagnostics": diagnostics
            }
            
        except Exception as e:
            return {
                "success": False,
                "response": f"Error processing query: {str(e)}",
                "diagnostics": {}
            }

    # def print_analysis_result(self, result):
    #     print("\n" + "="*50)
    #     print("MULTIAGENT DATABASE ANALYSIS")
    #     print("="*50)
        
    #     if result["success"]:
    #         print("\nDatabase Summary:")
    #         print("-"*50)
    #         if "database_info" in result["diagnostics"]:
    #             for info in result["diagnostics"]["database_info"]:
    #                 print(f"Database Size: {info.get('Size', 'N/A')} bytes")
    #                 print(f"Total Tables: {info.get('Tables', 'N/A')}")
    #                 print(f"Total Rows: {info.get('Total Rows', 'N/A')}")
            
    #         print("\nTable Details:")
    #         print("-"*50)
    #         if "table_details" in result["diagnostics"]:
    #             for table in result["diagnostics"]["table_details"]:
    #                 print(f"\nTable: {table.get('TABLE_NAME')}")
    #                 print(f"Engine: {table.get('ENGINE')}")
    #                 print(f"Rows: {table.get('TABLE_ROWS')}")
    #                 print(f"Size: {table.get('Size')} bytes")
                    
    #         print("\nAnalysis Response:")
    #         print("-"*50)
    #         print(result["response"])
    #     else:
    #         print("\nAnalysis Failed:")
    #         print(result["response"])
        
    #     print("\n" + "="*50 + "\n")

    def print_analysis_result(self, result):
        print("\n" + "="*50)
        print("MULTIAGENT DATABASE ANALYSIS")
        print("="*50)
        
        if result["success"]:
            # Add additional diagnostic queries for system variables and status
            connection = self.connect_to_database()
            if connection:
                # Get important system variables
                success, variables, _ = self.execute_query(connection, """
                    SHOW VARIABLES WHERE Variable_name IN (
                        'max_connections',
                        'innodb_buffer_pool_size',
                        'innodb_log_file_size',
                        'key_buffer_size',
                        'max_allowed_packet',
                        'wait_timeout',
                        'interactive_timeout',
                        'thread_cache_size',
                        'query_cache_size',
                        'innodb_flush_method',
                        'innodb_flush_log_at_trx_commit'
                    )
                """)
                
                # Get InnoDB status
                success_innodb, innodb_status, _ = self.execute_query(connection, "SHOW ENGINE INNODB STATUS")
                
                # Get global status
                success_status, global_status, _ = self.execute_query(connection, """
                    SHOW GLOBAL STATUS WHERE Variable_name IN (
                        'Threads_connected',
                        'Threads_running',
                        'Queries',
                        'Slow_queries',
                        'Uptime',
                        'Bytes_received',
                        'Bytes_sent'
                    )
                """)

                print("\nDatabase Summary:")
                print("-"*50)
                if "database_info" in result["diagnostics"]:
                    for info in result["diagnostics"]["database_info"]:
                        print(f"Database Size: {info.get('Size', 'N/A')} bytes")
                        print(f"Total Tables: {info.get('Tables', 'N/A')}")
                        print(f"Total Rows: {info.get('Total Rows', 'N/A')}")
                
                print("\nSystem Variables:")
                print("-"*50)
                if success and variables:
                    for var in variables:
                        print(f"{var['Variable_name']}: {var['Value']}")
                
                print("\nGlobal Status:")
                print("-"*50)
                if success_status and global_status:
                    for status in global_status:
                        print(f"{status['Variable_name']}: {status['Value']}")
                
                print("\nTable Details:")
                print("-"*50)
                if "table_details" in result["diagnostics"]:
                    for table in result["diagnostics"]["table_details"]:
                        print(f"\nTable: {table.get('TABLE_NAME')}")
                        print(f"Engine: {table.get('ENGINE')}")
                        print(f"Rows: {table.get('TABLE_ROWS')}")
                        print(f"Size: {table.get('Size')} bytes")
                        print(f"Created: {table.get('CREATE_TIME')}")
                        print(f"Updated: {table.get('UPDATE_TIME')}")
                
                print("\nIndex Information:")
                print("-"*50)
                if "table_indexes" in result["diagnostics"]:
                    current_table = None
                    for index in result["diagnostics"]["table_indexes"]:
                        if current_table != index.get('TABLE_NAME'):
                            current_table = index.get('TABLE_NAME')
                            print(f"\nTable: {current_table}")
                        print(f"Index: {index.get('INDEX_NAME')} "
                            f"(Column: {index.get('COLUMN_NAME')}, "
                            f"Non-Unique: {index.get('NON_UNIQUE')})")
                
                print("\nActive Processes:")
                print("-"*50)
                if "database_processes" in result["diagnostics"]:
                    for process in result["diagnostics"]["database_processes"]:
                        print(f"ID: {process.get('ID')}, "
                            f"User: {process.get('USER')}, "
                            f"State: {process.get('STATE')}, "
                            f"Time: {process.get('TIME')}s")
                
                print("\nAnalysis Response:")
                print("-"*50)
                print(result["response"])
                
                connection.close()
            else:
                print("Could not connect to database to fetch additional information")
        else:
            print("\nAnalysis Failed:")
            print(result["response"])
        
        print("\n" + "="*50 + "\n")

# if __name__ == "__main__":
#     # Run the test function if this file is executed directly
#     test_result = test_mysql_db_analysis_agent()
#     print(f"\nOverall test result: {'PASSED' if test_result else 'FAILED'}")

