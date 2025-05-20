import os
import glob
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
import traceback

from prompt import SYSTEM_INSTRUCTIONS

class MySQLDocSearchAgent:
    """
    A LangChain agent that specializes in MySQL documentation assistance.
    """
    def __init__(self, model="mistral", embedding_model="nomic-embed-text", docs_folder="sql_docs"):
        """
        Initialize the MySQL Documentation Search Agent.
        
        Args:
            model: LLM model name to use with Ollama
            embedding_model: Embedding model name to use with Ollama
            docs_folder: Folder containing MySQL documentation PDFs
        """

        self.llm = Ollama(model=model)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.docs_folder = docs_folder
        self.vectorstores = {}
        self.last_used_docs = []
    

        self.system_instructions = SYSTEM_INSTRUCTIONS

        # Initialize prompt templates
        self._setup_prompts()
        
        # Load vector stores
        self.load_vectorstores()
        
        # Setup routing chain
        self.setup_routing_chain()
        
        # Create tools
        self._create_tools()
        
        # Initialize agent
        self._initialize_agent()
    
    def _setup_prompts(self):
        """Set up the prompt templates for different query types."""
        self.compare_prompt = ChatPromptTemplate.from_template("""
        You are Doc Search, an AI assistant specializing in MySQL.
        Use ONLY the following documentation to answer the query.

        Source version and target version: {versions}
        User query: {query}

        Relevant documentation:
        {context}

        Provide a detailed explanation related to the query. If comparing versions, highlight the differences.
        If troubleshooting an error, explain the cause and solution.
        
        If applicable, provide example fixes or workarounds with code examples.

        If the documentation doesn't provide enough information, say so explicitly.
        Response:
        """)

        self.troubleshoot_prompt = ChatPromptTemplate.from_template("""
        You are Doc Search, an AI assistant specializing in MySQL troubleshooting.
        Use ONLY the following documentation to answer the query.

        User query: {query}

        Relevant documentation:
        {context}

        Provide a detailed explanation of the error or issue.
        Explain the cause and recommend solutions with code examples when possible.
        
        Format your response with clear headings for:
        - Problem Analysis
        - Likely Causes
        - Recommended Solutions
        - Preventive Measures (if applicable)

        If the documentation doesn't provide enough information, say so explicitly.
        Response:
        """)

        self.general_prompt = ChatPromptTemplate.from_template("""
        You are Doc Search, an AI assistant specializing in MySQL.
        Use ONLY the following documentation to answer the query.

        User query: {query}

        Relevant documentation:
        {context}

        Provide a comprehensive answer to the query based on the documentation.
        Include code examples when relevant.

        If the documentation doesn't provide enough information, say so explicitly.
        Response:
        """)
    
    def load_vectorstores(self):
        """Load and prepare vector stores from MySQL PDF documentation."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        pdf_files = glob.glob(f"{self.docs_folder}/*.pdf")

        if not pdf_files:
            print(f"Warning: No PDF files found in {self.docs_folder}.")
            return

        for pdf_file in pdf_files:
            try:
                filename = os.path.basename(pdf_file)
                if "mysql" in filename.lower():
                    version = filename.lower().replace("reference.pdf", "").replace("mysql", "").strip().replace("_", ".")
                    if not version:
                        version = "default"
                else:
                    version = "troubleshooting"

                print(f"Loading {filename} as version {version}...")
                docs = PyPDFLoader(pdf_file).load()
                for doc in docs:
                    doc.metadata["filename"] = filename
                chunks = splitter.split_documents(docs)

                for chunk in chunks:
                    chunk.metadata["version"] = version
                    chunk.metadata.setdefault("page", "unknown")

                self.vectorstores[version] = FAISS.from_documents(chunks, self.embeddings)
                print(f"Successfully loaded {filename} with {len(chunks)} chunks")
            except Exception as e:
                print(f"Error loading {pdf_file}: {e}")
    
    def retrieve_docs(self, query: str, versions: str = None, k: int = 5) -> List[Document]:
        
        if not self.vectorstores:
            print("No vector stores available. Loading them now...")
            self.load_vectorstores()
            if not self.vectorstores:
                return []

        docs = []

        # If versions are specified, search in those specific vectorstores
        if versions:
            versions_list = [v.strip() for v in versions.split(",")]
            for v in versions_list:
                if v in self.vectorstores:
                    docs.extend(self.vectorstores[v].similarity_search(query, k=k))

        # If no docs found and it's an error query, search in troubleshooting
        if not docs:
            error_keywords = ["error", "fail", "exception", "cannot", "invalid", "denied", "troubleshoot", "problem"]
            if any(word in query.lower() for word in error_keywords) and "troubleshooting" in self.vectorstores:
                docs.extend(self.vectorstores["troubleshooting"].similarity_search(query, k=k))

        # If still no docs, search in all vectorstores
        if not docs:
            for store in self.vectorstores.values():
                docs.extend(store.similarity_search(query, k=3))

        self.last_used_docs = docs[:k]
        return self.last_used_docs

    def setup_routing_chain(self):
        """Set up the routing chain for classifying queries."""
        prompt_infos = [
            {
                "name": "Version Migration",
                "description": "Use this for version migration or comparison queries.",
                "prompt_template": """You are a MySQL migration specialist. Determine if this query is about migrating between MySQL versions or comparing version features.
If so, identify the source and target versions mentioned or implied.
Return format: source_version,target_version
If not, respond with "NOT_MIGRATION"

Query:
{input}
"""
            },
            {
                "name": "Error Troubleshooting",
                "description": "Use this for troubleshooting MySQL errors.",
                "prompt_template": """You are a MySQL troubleshooting specialist. Determine if this query is about an error or troubleshooting.
If so, respond with "TROUBLESHOOT"
If not, respond with "NOT_TROUBLESHOOT"

Query:
{input}
"""
            },
            {
                "name": "General MySQL",
                "description": "Use this for general MySQL usage questions.",
                "prompt_template": """You are a MySQL knowledge expert. Handle general MySQL queries.
Respond with "GENERAL"

Query:
{input}
"""
            }
        ]

        self.multi_chain = MultiPromptChain.from_prompts(
            llm=self.llm,
            prompt_infos=prompt_infos,
            verbose=True
        )

    def _format_references(self, docs: List[Document]) -> str:
        """Format document references for inclusion in responses."""
        if not docs:
            return ""

        refs = "\n\nReferences:\n"
        seen = set()
        for doc in docs:
            ref = f"{doc.metadata.get('filename')}, Page: {doc.metadata.get('page')}, Version: {doc.metadata.get('version')}"
            if ref not in seen:
                refs += f"- {ref}\n"
                seen.add(ref)
        return refs

    def handle_version_migration(self, query: str, versions: str) -> str:
        """Handle version migration or comparison queries."""
        docs = self.retrieve_docs(query, versions)
        doc_chain = create_stuff_documents_chain(self.llm, self.compare_prompt)
        response = doc_chain.invoke({
            "query": query,
            "versions": versions,
            "context": docs  
        })
        references = self._format_references(docs)
        if references:
            return response + references
        return response
    
    def handle_troubleshooting(self, query: str) -> str:
        """Handle troubleshooting queries."""
        docs = self.retrieve_docs(query, "troubleshooting")
        if not docs:
            docs = self.retrieve_docs(query)
        doc_chain = create_stuff_documents_chain(self.llm, self.troubleshoot_prompt)
        response = doc_chain.invoke({
            "query": query,
            "context": docs  
        })
        references = self._format_references(docs)
        if references:
            return response + references
        return response
    
    def handle_general_query(self, query: str) -> str:
        """Handle general MySQL queries."""
        docs = self.retrieve_docs(query)
        doc_chain = create_stuff_documents_chain(self.llm, self.general_prompt)
        response = doc_chain.invoke({
            "query": query,
            "context": docs  
        })
        references = self._format_references(docs)
        if references:
            return response + references
        return response

    def _route_and_process_query(self, query: str) -> str:
        """Route and process a query using the routing chain."""
        try:
            route_result = self.multi_chain.invoke({"input": query})
            destination = route_result.get("destination", "")
            routed_text = route_result.get("text", "").strip()

            if "Version Migration" in destination:
                if routed_text == "NOT_MIGRATION":
                    return self.handle_general_query(query)
                return self.handle_version_migration(query, routed_text)

            if "Error Troubleshooting" in destination:
                if routed_text == "TROUBLESHOOT":
                    return self.handle_troubleshooting(query)
                return self.handle_general_query(query)

            return self.handle_general_query(query)

        except Exception as e:
            return f"An error occurred: {str(e)}\n\n{traceback.format_exc()}"
    
    def _create_tools(self):
        """Create tools for the agent."""
        self.tools = [
            Tool(
                name="MySQL Version Migration Tool",
                func=lambda q: self.handle_version_migration(q, self._extract_versions(q)),
                description="Useful for comparing features between different MySQL versions or migration questions."
            ),
            Tool(
                name="MySQL Troubleshooting Tool",
                func=self.handle_troubleshooting,
                description="Useful for diagnosing and fixing MySQL errors and issues."
            ),
            Tool(
                name="MySQL General Query Tool",
                func=self.handle_general_query,
                description="Useful for general MySQL questions about syntax, features, or usage."
            )
        ]
    
    def _extract_versions(self, query: str) -> str:
        """Try to extract MySQL versions from a query."""
        common_versions = ["5.6", "5.7", "8.0", "8.1"]
        mentioned_versions = []
        
        for version in common_versions:
            if version in query:
                mentioned_versions.append(version)
        
        return ",".join(mentioned_versions) if mentioned_versions else ""
    
    def _initialize_agent(self):
        """Initialize the agent with tools."""
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query using the appropriate method.
        
        Returns:
            Dictionary containing response text and metadata
        """
        try:
            # First try with the agent
            try:
                response = self.agent.invoke({"input": query})
                return {
                    "response": response,
                    "docs": self.last_used_docs,
                    "success": True
                }
            except Exception as agent_error:
                print(f"Agent error: {agent_error}. Falling back to routing chain.")
                # If agent fails, fall back to routing chain
                response = self._route_and_process_query(query)
                return {
                    "response": response,
                    "docs": self.last_used_docs,
                    "success": True
                }
        except Exception as e:
            error_msg = f"An error occurred in DocSearch Agent: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            return {
                "response": error_msg,
                "docs": [],
                "success": False
            }

