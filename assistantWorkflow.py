import os
import glob
import pymysql
import traceback
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

from dBAnalysisAgent import MySQLDBAnalysisAgent
from docSearchAgent import MySQLDocSearchAgent
from solutionAgent import MySQLSolutionAgent



class MySQLAssistantWorkflow:
    """
    Main workflow class that orchestrates the sequential flow of agents
    """
    def __init__(self, model="mistral", embedding_model="nomic-embed-text", docs_folder="sql_docs"):
        """
        Initialize the MySQL Assistant Workflow.
        
        Args:
            model: LLM model name to use with Ollama
            embedding_model: Embedding model name to use with Ollama
            docs_folder: Folder containing MySQL documentation PDFs
        """
        self.doc_search_agent = MySQLDocSearchAgent(model=model, embedding_model=embedding_model, docs_folder=docs_folder)
        self.db_analysis_agent = MySQLDBAnalysisAgent(model=model)
        self.solution_agent = MySQLSolutionAgent(model=model)
        
    def process_query(self, query: str, connection_params: Dict[str, Any]) -> str:
        """
        Process a user query through the complete workflow.
        
        Args:
            query: User's original query
            connection_params: Database connection parameters
            
        Returns:
            Final response to the user
        """
        print("Step 1: Executing DocSearch Agent...")
        doc_search_result = self.doc_search_agent.process_query(query)
        
        if not doc_search_result.get("success", False):
            return f"Error in DocSearch Agent: {doc_search_result.get('response')}"
        
        print("Step 2: Executing DB Analysis Agent...")
        db_analysis_result = self.db_analysis_agent.process_query(query, doc_search_result, connection_params)
        
           
        print("\n" + "="*50)
        print("DB ANALYSIS RESULTS:")
        print("="*50)
        if db_analysis_result.get("success", False):
            print("\n Analysis Response:")
            print("-"*50)
            print(db_analysis_result.get("response"))
            
            # print("\n Key Diagnostics:")
            # print("-"*50)
            # for key, value in db_analysis_result.get("diagnostics", {}).items():
            #     if isinstance(value, list) and len(value) > 0:
            #         print(f"\n{key}:")
            #         print(f"Sample: {value[0]}")
        else:
            print("\n Analysis Failed:")
            print(db_analysis_result.get("response"))
        print("\n" + "="*50)


        if not db_analysis_result.get("success", False):
            return f"Error in DB Analysis Agent: {db_analysis_result.get('response')}\n\nDocumentation results:\n{doc_search_result.get('response')}"
        

        print("Step 3: Executing Solution Agent...")
        solution_result = self.solution_agent.process_query(query, doc_search_result, db_analysis_result)
        
        if not solution_result.get("success", False):
            return f"Error in Solution Agent: {solution_result.get('response')}\n\nPartial results:\n{doc_search_result.get('response')}\n\n{db_analysis_result.get('response')}"
        
        return solution_result.get("response")
    


    ## host, troubleshooting, anaylze, particular host, alot of invalid queries, and tell number of hosts and then suggest increase it 