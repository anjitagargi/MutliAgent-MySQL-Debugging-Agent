from typing import Dict, Any, Tuple, List
from pymysql.connections import Connection
import pymysql

def execute_query(connection: Connection, query: str) -> Tuple[bool, List[Dict], str]:
    """
    Execute a MySQL query and return results
    
    Args:
        connection: MySQL connection object
        query: SQL query string
        
    Returns:
        Tuple of (success, results, error_message)
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
            return True, result, ""
    except Exception as e:
        error_message = f"Error executing query: {str(e)}"
        return False, [], error_message

def collect_system_parameters(connection: Connection) -> Dict[str, Any]:
    """
    Collect important MySQL system parameters and configuration settings
    
    Args:
        connection: MySQL database connection object
        
    Returns:
        Dict containing system parameters and their values
    """
    system_params = {}
    param_queries = {
        "buffer_pools": """
            SHOW GLOBAL VARIABLES WHERE Variable_name IN (
                'innodb_buffer_pool_size',
                'innodb_buffer_pool_instances',
                'innodb_buffer_pool_chunk_size'
            )
        """,
        "connection_limits": """
            SHOW GLOBAL VARIABLES WHERE Variable_name IN (
                'max_connections',
                'max_user_connections',
                'thread_cache_size',
                'wait_timeout'
            )
        """,
        "memory_settings": """
            SHOW GLOBAL VARIABLES WHERE Variable_name IN (
                'key_buffer_size',
                'query_cache_size',
                'tmp_table_size',
                'max_heap_table_size'
            )
        """,
        "innodb_settings": """
            SHOW GLOBAL VARIABLES WHERE Variable_name IN (
                'innodb_log_file_size',
                'innodb_log_buffer_size',
                'innodb_flush_log_at_trx_commit',
                'innodb_flush_method'
            )
        """,
        "current_usage": """
            SHOW GLOBAL STATUS WHERE Variable_name IN (
                'Max_used_connections',
                'Threads_connected',
                'Threads_running',
                'Queries',
                'Slow_queries'
            )
        """
    }

    for name, query in param_queries.items():
        try:
            success, results, error = execute_query(connection, query)
            if success:
                system_params[name] = {row['Variable_name']: row['Value'] for row in results}
            else:
                system_params[f"{name}_error"] = error
        except Exception as e:
            system_params[f"{name}_error"] = str(e)

    return system_params