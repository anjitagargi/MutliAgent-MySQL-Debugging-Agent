import traceback
from typing import Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.chains.combine_documents.stuff import StuffDocumentsChain, create_stuff_documents_chain
from langchain.agents import AgentType, initialize_agent, Tool

from prompt import SYSTEM_INSTRUCTIONS


class MySQLSolutionAgent:
    def __init__(self, model="mistral"):
        
        self.llm = Ollama(model=model)
        self._setup_prompts()
        self.system_instructions = SYSTEM_INSTRUCTIONS
        
        
    def _setup_prompts(self):
        """Set up the prompt templates"""
        self.solution_prompt = ChatPromptTemplate.from_template("""
        You are a MySQL database solution architect. Based on the following information,
        provide a detailed solution:

        User query: {query}
        
        Documentation search results: {doc_search_result}
        
        Database analysis: {db_analysis_result}
        
        Additional context: {context}

        Incase of a information is asked, give solution of that information directly only. Give concise answer.
                                                                                                                                                                               
        Important:
        - Extract and use EXACT values from the database analysis results
        - For max_connections changes, use the specific value recommended in db_analysis_result
        - Always reference actual values found in the analysis
        
        Follow these guidelines for your response:
        1. For configuration changes (like max_connections):
        - Extract the exact suggested value from db_analysis_result
        - Show current value and recommended value in the overview
        - Provide commands with the specific extracted value
        
        2. For general solutions:
        - List concrete steps with actual values from analysis
        - Keep steps concise and actionable
        - Include verification steps with expected values

        Format your response as:
        SOLUTION OVERVIEW:
        [Brief summary with actual current and target values]

        STEPS TO IMPLEMENT:
        1. [Step 1 with exact values]
        2. [Step 2 with exact values]
        ...

    VERIFICATION:
    - [Specific checks with expected values]

    ADDITIONAL RECOMMENDATIONS:
    - [If applicable, with specific metrics]
        Please provide a comprehensive solution that addresses the user's query,
        taking into account both the documentation and database analysis results.
        Include specific recommendations and steps to implement the solution.
                                                                
        Solution:
        """)

    def generate_solution(self, query: str, doc_search_result: Dict[str, Any], 
                        db_analysis_result: Dict[str, Any]) -> str:
        
        # Create a runnable chain
        chain = self.solution_prompt | self.llm
        
        # Run the chain with all required variables
        response = chain.invoke({
            "query": query,
            "doc_search_result": doc_search_result.get("response", "No documentation results available"),
            "db_analysis_result": db_analysis_result.get("response", "No database analysis available"),
            "context": "Additional context for the solution"  # Required context variable
        })
        
        return response.content if hasattr(response, 'content') else str(response)
    
    def process_query(self, query: str, doc_search_result: Dict[str, Any], 
                     db_analysis_result: Dict[str, Any]) -> Dict[str, Any]:

        try:
            solution = self.generate_solution(query, doc_search_result, db_analysis_result)
            
            return {
                "response": solution,
                "success": True
            }
            
        except Exception as e:
            error_msg = f"An error occurred in Solution Agent: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            return {
                "response": error_msg,
                "success": False
            }
        