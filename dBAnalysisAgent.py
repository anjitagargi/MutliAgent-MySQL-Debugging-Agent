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
    def __init__(self, model="mistral"):
    
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
        Process a database query and return structured analysis results
        """
        try:
            connection = self.connect_to_database(**connection_params)
            if not connection:
                return {
                    "success": False,
                    "response": "Failed to connect to database",
                    "diagnostics": {}
                }

            # Collect all relevant database metrics
            diagnostics = {}
            
            # System variables
            success, variables, _ = self.execute_query(connection, """
                SHOW VARIABLES WHERE Variable_name IN (
                    'max_connections', 'innodb_buffer_pool_size', 
                    'query_cache_size', 'query_cache_type', 
                    'max_heap_table_size', 'tmp_table_size'
                )
            """)
            if success:
                diagnostics['system_variables'] = variables

            # Performance metrics
            success, performance, _ = self.execute_query(connection, """
                SHOW GLOBAL STATUS WHERE Variable_name IN (
                    'Threads_connected', 'Max_used_connections',
                    'Aborted_connects', 'Bytes_received',
                    'Bytes_sent', 'Questions', 'Slow_queries'
                )
            """)
            if success:
                diagnostics['performance_metrics'] = performance

            # Table statistics
            success, tables, _ = self.execute_query(connection, """
                SELECT 
                    TABLE_NAME,
                    TABLE_ROWS,
                    ROUND(DATA_LENGTH/1024/1024, 2) AS data_size_mb,
                    ROUND(INDEX_LENGTH/1024/1024, 2) AS index_size_mb,
                    ENGINE
                FROM information_schema.TABLES 
                WHERE TABLE_SCHEMA = 'multiAgent'
            """)
            if success:
                diagnostics['table_stats'] = tables

            # Process information
            success, processes, _ = self.execute_query(connection, """
                SELECT COUNT(*) as active_processes, 
                       SUM(TIME) as total_process_time 
                FROM INFORMATION_SCHEMA.PROCESSLIST 
                WHERE COMMAND != 'Sleep'
            """)
            if success:
                diagnostics['process_info'] = processes

            # Format all collected data into a structured response
            analysis_result = {
                "system_config": {
                    var['Variable_name']: var['Value'] 
                    for var in diagnostics.get('system_variables', [])
                },
                "performance_stats": {
                    stat['Variable_name']: stat['Value'] 
                    for stat in diagnostics.get('performance_metrics', [])
                },
                "table_metrics": [
                    {
                        "name": table['TABLE_NAME'],
                        "rows": table['TABLE_ROWS'],
                        "size_mb": table['data_size_mb'],
                        "index_mb": table['index_size_mb'],
                        "engine": table['ENGINE']
                    }
                    for table in diagnostics.get('table_stats', [])
                ],
                "process_metrics": diagnostics.get('process_info', [{}])[0]
            }

            connection.close()

            # Analyze the collected data
            analysis = self.analyze_diagnostics(query, analysis_result, 
                                             doc_search_result.get("response", "") if doc_search_result else "")

            return {
                "success": True,
                "response": analysis,
                "diagnostics": analysis_result  # Structured metrics for SolutionAgent
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
    #         connection = self.connect_to_database()
    #         if connection:
    #             # Get transaction analysis
    #             success_trans, transactions, _ = self.execute_query(connection, """
    #                 SELECT 
    #                     t.transaction_id, 
    #                     t.transaction_type, 
    #                     t.amount, 
    #                     t.transaction_date, 
    #                     u.username
    #                 FROM 
    #                     transactions t
    #                 JOIN 
    #                     users u ON t.user_id = u.user_id
    #                 WHERE 
    #                     t.transaction_date BETWEEN 
    #                         DATE_SUB(NOW(), INTERVAL 15 DAY) AND 
    #                         DATE_SUB(NOW(), INTERVAL 10 DAY)
    #                 ORDER BY 
    #                     t.amount DESC
    #                 LIMIT 5
    #             """)

    #             # Get connection statistics
    #             success_conn, conn_errors, _ = self.execute_query(connection, 
    #                 "SHOW GLOBAL STATUS LIKE 'Connection_errors%'")

    #             # Get system variables
    #             success_vars, variables, _ = self.execute_query(connection, """
    #                 SHOW VARIABLES WHERE Variable_name IN (
    #                     'max_connections',
    #                     'innodb_buffer_pool_size',
    #                     'query_cache_size',
    #                     'query_cache_type'
    #                 )
    #             """)

    #             # Get long running queries
    #             success_long, long_queries, _ = self.execute_query(connection, """
    #                 SELECT 
    #                     ID, USER, HOST, DB, COMMAND, TIME, STATE, INFO 
    #                 FROM 
    #                     INFORMATION_SCHEMA.PROCESSLIST 
    #                 WHERE 
    #                     COMMAND != 'Sleep' 
    #                     AND TIME > 5
    #                 ORDER BY 
    #                     TIME DESC
    #             """)

    #             # Get table statistics
    #             success_stats, table_stats, _ = self.execute_query(connection, """
    #                 SELECT 
    #                     TABLE_SCHEMA, 
    #                     TABLE_NAME, 
    #                     TABLE_ROWS,
    #                     ROUND(DATA_LENGTH/1024/1024, 2) AS data_size_mb,
    #                     ROUND(INDEX_LENGTH/1024/1024, 2) AS index_size_mb
    #                 FROM 
    #                     information_schema.TABLES 
    #                 WHERE 
    #                     TABLE_SCHEMA = 'multiAgent'
    #                 ORDER BY 
    #                     data_size_mb DESC
    #             """)

    #             # Get InnoDB transactions and locks
    #             success_trx, innodb_trx, _ = self.execute_query(connection, 
    #                 "SELECT * FROM information_schema.INNODB_TRX")
    #             success_locks, data_locks, _ = self.execute_query(connection, 
    #                 "SELECT * FROM performance_schema.data_locks")

    #             print("\nRecent Transactions (Last 15-10 days):")
    #             print("-"*50)
    #             if success_trans and transactions:
    #                 for tx in transactions:
    #                     print(f"ID: {tx.get('transaction_id')}, "
    #                         f"Type: {tx.get('transaction_type')}, "
    #                         f"Amount: {tx.get('amount')}, "
    #                         f"Date: {tx.get('transaction_date')}, "
    #                         f"User: {tx.get('username')}")

    #             print("\nConnection Statistics:")
    #             print("-"*50)
    #             if success_conn and conn_errors:
    #                 for err in conn_errors:
    #                     print(f"{err.get('Variable_name')}: {err.get('Value')}")

    #             print("\nSystem Variables:")
    #             print("-"*50)
    #             if success_vars and variables:
    #                 for var in variables:
    #                     print(f"{var.get('Variable_name')}: {var.get('Value')}")

    #             print("\nLong Running Queries (>5 seconds):")
    #             print("-"*50)
    #             if success_long and long_queries:
    #                 for query in long_queries:
    #                     print(f"ID: {query.get('ID')}, "
    #                         f"User: {query.get('USER')}, "
    #                         f"Time: {query.get('TIME')}s, "
    #                         f"State: {query.get('STATE')}")

    #             print("\nTable Statistics:")
    #             print("-"*50)
    #             if success_stats and table_stats:
    #                 for stat in table_stats:
    #                     print(f"Table: {stat.get('TABLE_NAME')}, "
    #                         f"Rows: {stat.get('TABLE_ROWS')}, "
    #                         f"Data Size: {stat.get('data_size_mb')}MB, "
    #                         f"Index Size: {stat.get('index_size_mb')}MB")

    #             print("\nActive InnoDB Transactions:")
    #             print("-"*50)
    #             if success_trx and innodb_trx:
    #                 for trx in innodb_trx:
    #                     print(f"TRX ID: {trx.get('trx_id')}, "
    #                         f"State: {trx.get('trx_state')}, "
    #                         f"Started: {trx.get('trx_started')}")

    #             print("\nData Locks:")
    #             print("-"*50)
    #             if success_locks and data_locks:
    #                 for lock in data_locks:
    #                     print(f"Lock ID: {lock.get('LOCK_ID')}, "
    #                         f"Type: {lock.get('LOCK_TYPE')}, "
    #                         f"Mode: {lock.get('LOCK_MODE')}")

    #             print("\nAnalysis Response:")
    #             print("-"*50)
    #             print(result["response"])
                
    #             connection.close()
    #         else:
    #             print("Could not connect to database to fetch additional information")
    #     else:
    #         print("\nAnalysis Failed:")
    #         print(result["response"])
        
    # print("\n" + "="*50 + "\n")

# if __name__ == "__main__":
#     # Run the test function if this file is executed directly
#     test_result = test_mysql_db_analysis_agent()
#     print(f"\nOverall test result: {'PASSED' if test_result else 'FAILED'}")

