import os
import sys
import json
import time
import boto3
import streamlit as st
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Graph Database Components
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Vector Store Components
from langchain_community.vectorstores import FAISS
from langchain_aws.embeddings import BedrockEmbeddings

# Document Processing
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# LLM and Chain Components
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Initialize AWS Bedrock clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Document Loading and Processing
def load_documents(directory="data"):
    """Load and process PDFs and Markdown (.md) files."""
    docs = []
    
    # Process PDF files
    pdf_files = [f for f in os.listdir(directory) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(directory, pdf_file))
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(pages)
        for doc in split_docs:
            docs.append(Document(page_content=doc.page_content.replace("\n", ""), 
                                metadata={'source': pdf_file, 'type': 'pdf'}))

    # Process Markdown files
    md_files = [f for f in os.listdir(directory) if f.endswith(".md")]
    for md_file in md_files:
        with open(os.path.join(directory, md_file), "r", encoding="utf-8") as file:
            content = file.read()
        
        # Extract project name from filename
        project_name = os.path.splitext(md_file)[0]
        
        # Parse markdown content
        sections = parse_markdown_sections(content)
        
        # Create a document for each section with rich metadata
        for section_name, section_content in sections.items():
            docs.append(Document(
                page_content=section_content.strip(),
                metadata={
                    'source': md_file,
                    'type': 'markdown',
                    'section': section_name,
                    'project_name': project_name
                }
            ))

    return docs

def parse_markdown_sections(md_content):
    """Parse markdown content into structured sections."""
    sections = {}
    
    # Extract sections based on headers
    current_section = "General"
    current_content = []
    
    for line in md_content.split('\n'):
        if line.startswith('#'):
            # Save previous section
            if current_content:
                sections[current_section] = '\n'.join(current_content)
                current_content = []
            
            # Extract new section name
            current_section = line.strip('# \n')
        else:
            current_content.append(line)
    
    # Save the last section
    if current_content:
        sections[current_section] = '\n'.join(current_content)
    
    return sections

# Neo4j Graph Database Functions
def initialize_graph(neo4j_url, neo4j_username, neo4j_password):
    """Connect to Neo4j without loading data."""
    try:
        graph = Neo4jGraph(url=neo4j_url, username=neo4j_username, password=neo4j_password)
        st.sidebar.success("Connected to Neo4j database.")
        return graph
    except Exception as e:
        st.sidebar.error(f"Neo4j Connection Failed: {e}")
        return None

def load_data_into_neo4j(graph, directory="dtata"):
    """Load documents and build a rich knowledge graph with direct relationship creation."""
    with st.spinner("Loading documents..."):
        all_docs = load_documents(directory)
        if not all_docs:
            st.sidebar.warning("No documents found in the 'data' folder.")
            return None
        st.sidebar.success(f"Loaded {len(all_docs)} document chunks.")

    # Clear existing data to avoid duplication
    try:
        graph.query("MATCH (n) DETACH DELETE n")
        st.sidebar.info("Cleared existing graph data.")
    except Exception as e:
        st.sidebar.warning(f"Could not clear existing data: {e}")

    # First, extract all projects and build a project index
    projects = {}
    technologies = set()
    problems = set()
    needs = set()
    solutions = set()
    
    # Process all documents to extract key entities
    for doc in all_docs:
        # Extract project name
        project_name = None
        if 'project_name' in doc.metadata:
            project_name = doc.metadata['project_name']
        elif 'Brief:' in doc.page_content:
            lines = doc.page_content.split('\n')
            for line in lines:
                if line.startswith('Brief:'):
                    project_name = line.replace('Brief:', '').strip()
                    break
        
        if not project_name:
            continue
            
        # Initialize project data structure if new
        if project_name not in projects:
            projects[project_name] = {
                'technologies': set(),
                'problems': set(),
                'needs': set(),
                'solutions': set(),
                'sections': {}
            }
        
        # Extract technologies, problems, needs and solutions based on section names
        if 'section' in doc.metadata:
            section_name = doc.metadata['section']
            content = doc.page_content
            
            # Store section content
            projects[project_name]['sections'][section_name] = content
            
            # Extract entities based on section
            if section_name == "Technologies Involved" or "Technologies" in section_name:
                # Extract technologies
                tech_items = [t.strip() for t in re.split(r'[,\n]', content) if t.strip()]
                for tech in tech_items:
                    if len(tech) > 3:  # Avoid very short names
                        projects[project_name]['technologies'].add(tech)
                        technologies.add(tech)
                        
            elif section_name == "Customer Need" or "Need" in section_name:
                # Store need content
                projects[project_name]['needs'].add(content)
                needs.add(content)
                
            elif "Problem" in section_name or "Challenge" in section_name:
                # Extract problems
                problem_items = [p.strip() for p in re.split(r'[.\n]', content) if p.strip() and len(p.strip()) > 10]
                for problem in problem_items:
                    projects[project_name]['problems'].add(problem)
                    problems.add(problem)
                    
            elif "Solution" in section_name:
                # Store solution content
                projects[project_name]['solutions'].add(content)
                solutions.add(content)
    
    # Now let's build the graph with explicit relationships
    with st.spinner("Building knowledge graph with explicit relationships..."):
        try:
            # 1. Create all Project nodes
            for project_name in projects:
                # Sanitize string for Cypher
                safe_name = project_name.replace("'", "''")
                cypher = f"""
                CREATE (p:Project {{name: '{safe_name}'}})
                RETURN p
                """
                graph.query(cypher)
                
            # 2. Create Technology nodes and relationships
            for tech in technologies:
                # Sanitize string for Cypher
                safe_tech = tech.replace("'", "''")
                # Create Technology node
                cypher = f"""
                MERGE (t:Technology {{name: '{safe_tech}'}})
                RETURN t
                """
                graph.query(cypher)
                
                # Create relationships to Projects
                for project_name, project_data in projects.items():
                    if tech in project_data['technologies']:
                        safe_project = project_name.replace("'", "''")
                        cypher = f"""
                        MATCH (p:Project {{name: '{safe_project}'}}), (t:Technology {{name: '{safe_tech}'}})
                        MERGE (p)-[:USES_TECHNOLOGY]->(t)
                        """
                        graph.query(cypher)
            
            # 3. Create Problem nodes and relationships
            for problem in problems:
                if len(problem) > 10:  # Ensure meaningful problems
                    # Sanitize string for Cypher
                    safe_problem = problem.replace("'", "''")
                    # Create Problem node
                    cypher = f"""
                    MERGE (pr:Problem {{description: '{safe_problem}'}})
                    RETURN pr
                    """
                    graph.query(cypher)
                    
                    # Create relationships to Projects
                    for project_name, project_data in projects.items():
                        if problem in project_data['problems']:
                            safe_project = project_name.replace("'", "''")
                            cypher = f"""
                            MATCH (p:Project {{name: '{safe_project}'}}), (pr:Problem {{description: '{safe_problem}'}})
                            MERGE (p)-[:FACES_PROBLEM]->(pr)
                            """
                            graph.query(cypher)
            
            # 4. Create Need nodes and relationships
            for need in needs:
                if len(need) > 10:  # Ensure meaningful needs
                    # Sanitize string for Cypher
                    safe_need = need.replace("'", "''")
                    # Create Need node
                    cypher = f"""
                    MERGE (n:Need {{description: '{safe_need}'}})
                    RETURN n
                    """
                    graph.query(cypher)
                    
                    # Create relationships to Projects
                    for project_name, project_data in projects.items():
                        if need in project_data['needs']:
                            safe_project = project_name.replace("'", "''")
                            cypher = f"""
                            MATCH (p:Project {{name: '{safe_project}'}}), (n:Need {{description: '{safe_need}'}})
                            MERGE (p)-[:HAS_NEED]->(n)
                            """
                            graph.query(cypher)
            
            # 5. Create Solution nodes and relationships
            for solution in solutions:
                if len(solution) > 10:  # Ensure meaningful solutions
                    # Sanitize string for Cypher
                    safe_solution = solution.replace("'", "''")
                    # Create Solution node
                    cypher = f"""
                    MERGE (s:Solution {{description: '{safe_solution}'}})
                    RETURN s
                    """
                    graph.query(cypher)
                    
                    # Create relationships to Projects
                    for project_name, project_data in projects.items():
                        if solution in project_data['solutions']:
                            safe_project = project_name.replace("'", "''")
                            cypher = f"""
                            MATCH (p:Project {{name: '{safe_project}'}}), (s:Solution {{description: '{safe_solution}'}})
                            MERGE (p)-[:PROVIDES_SOLUTION]->(s)
                            """
                            graph.query(cypher)
            
            # 6. Create relationships between entities
            # Connect Problems to Solutions (if they belong to the same project)
            for project_name, project_data in projects.items():
                for problem in project_data['problems']:
                    for solution in project_data['solutions']:
                        safe_problem = problem.replace("'", "''")
                        safe_solution = solution.replace("'", "''")
                        cypher = f"""
                        MATCH (pr:Problem {{description: '{safe_problem}'}}), 
                              (s:Solution {{description: '{safe_solution}'}})
                        MERGE (s)-[:SOLVES_PROBLEM]->(pr)
                        """
                        graph.query(cypher)
            
            # Connect Technologies to Solutions (if they belong to the same project)
            for project_name, project_data in projects.items():
                for tech in project_data['technologies']:
                    for solution in project_data['solutions']:
                        safe_tech = tech.replace("'", "''")
                        safe_solution = solution.replace("'", "''")
                        cypher = f"""
                        MATCH (t:Technology {{name: '{safe_tech}'}}), 
                              (s:Solution {{description: '{safe_solution}'}})
                        MERGE (s)-[:USES_TECHNOLOGY]->(t)
                        """
                        graph.query(cypher)
            
            # 7. Create relationships between similar entities across projects
            # Connect similar technologies
            cypher = """
            MATCH (t:Technology)
            WITH t.name AS tech, COUNT(*) AS occurrences
            WHERE occurrences > 1
            MATCH (p1:Project)-[:USES_TECHNOLOGY]->(t1:Technology), 
                  (p2:Project)-[:USES_TECHNOLOGY]->(t2:Technology)
            WHERE t1.name = t2.name AND p1 <> p2
            MERGE (p1)-[:SIMILAR_TECHNOLOGY {technology: t1.name}]->(p2)
            """
            graph.query(cypher)
            
            # Create indices for better performance
            cypher_indices = [
                "CREATE INDEX IF NOT EXISTS FOR (p:Project) ON (p.name)",
                "CREATE INDEX IF NOT EXISTS FOR (t:Technology) ON (t.name)",
                "CREATE INDEX IF NOT EXISTS FOR (pr:Problem) ON (pr.description)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Need) ON (n.description)",
                "CREATE INDEX IF NOT EXISTS FOR (s:Solution) ON (s.description)"
            ]
            
            for index_query in cypher_indices:
                try:
                    graph.query(index_query)
                except:
                    pass  # Ignore index creation errors
                
            st.sidebar.success("Knowledge graph built with rich relationships!")
            return all_docs
            
        except Exception as e:
            st.sidebar.error(f"Error building knowledge graph: {e}")
            st.error(str(e))
            return all_docs

# Vector Store Functions
def create_vector_store(docs, store_name="faiss_index"):
    """Create and save a FAISS vector store from documents."""
    with st.spinner("Creating vector embeddings..."):
        try:
            vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
            vectorstore_faiss.save_local(store_name)
            st.sidebar.success(f"Vector store created with {len(docs)} documents.")
            return vectorstore_faiss
        except Exception as e:
            st.sidebar.error(f"Error creating vector store: {e}")
            return None

def load_vector_store(store_name="faiss_index"):
    """Load a saved FAISS vector store."""
    try:
        if os.path.exists(store_name):
            vectorstore_faiss = FAISS.load_local(store_name, bedrock_embeddings, allow_dangerous_deserialization=True)
            return vectorstore_faiss
        else:
            st.warning(f"Vector store '{store_name}' not found.")
            return None
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

# Query Processing Functions
def query_refinement(question, conversation_history=None):
    """Enhance user queries while preserving their meaning."""
    llm = Bedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",  # or "anthropic.claude-3-haiku-20240307-v1:0"
        client=bedrock,
        model_kwargs={
            "max_tokens": 512,
            "temperature": 0.2,
            "anthropic_version": "bedrock-2023-05-31"
        }
    )
    
    history_context = ""
    if conversation_history and len(conversation_history) > 0:
        history_context = "Previous conversation:\n" + "\n".join(
            [f"User: {q}\nSystem: {a[:100]}..." for q, a in conversation_history[-3:]]
        )
    
    refinement_prompt = PromptTemplate(
        template="""
        You are an expert in enhancing questions about technology and business projects.
        
        The knowledge system helps people understand and answer questions about various technology implementation projects with:
        - Project details and client solutions
        - Customer needs and business challenges
        - Technical solutions and technologies used
        - Project outcomes and business impacts
        
        Original user question: "{question}"
        
        Your task:
        1. Correct any spelling or grammatical errors
        2. Expand abbreviations or technical terms that would help clarify the question
        3. Make minor enhancements to wording for clarity
        4. Keep the original meaning and scope of the question unchanged
        5. DO NOT make general questions more specific
        6. DO NOT add constraints not in the original question
        
        Enhanced question:
        """,
        input_variables=["question"]
    )
    
    refined_question = llm.invoke(
        refinement_prompt.format(
            question=question
        )
    ).strip()
    
    return refined_question

def setup_graph_qa_chain(graph):
    """Set up an improved chain for querying the knowledge graph with better Cypher generation."""
    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={
        'max_gen_len': 1024,
        'temperature': 0.2
    })

    # Get graph schema
    try:
        schema = graph.get_schema
    except:
        try:
            schema = graph.schema
        except:
            # Generate a basic schema with manual query
            try:
                node_types = graph.query("CALL db.labels()")
                rel_types = graph.query("CALL db.relationshipTypes()")
                sample_props = graph.query("""
                MATCH (n)
                WITH LABELS(n) AS labels, KEYS(n) AS props, n
                RETURN DISTINCT labels, props
                LIMIT 5
                """)
                
                schema = f"""
                Node types: {[row['label'] for row in node_types]}
                Relationship types: {[row['relationshipType'] for row in rel_types]}
                Sample properties: {sample_props}
                """
            except:
                schema = "Schema information not available"
    
    # Create a function to generate better Cypher queries
    def generate_cypher(question):
        cypher_prompt = PromptTemplate(
            template="""
            Task: Generate a precise Cypher statement to query a Neo4j graph database to answer the given question.

            Database Schema:
            {schema}

            Available Node Labels:
            - Project: Central node for each project (properties: name)
            - Technology: Technologies used in projects (properties: name)
            - Problem: Issues/challenges faced in projects (properties: description)
            - Need: Customer requirements for projects (properties: description)
            - Solution: Implementations that address needs (properties: description)
            
            Available Relationship Types:
            - USES_TECHNOLOGY: (Project)-[:USES_TECHNOLOGY]->(Technology)
            - FACES_PROBLEM: (Project)-[:FACES_PROBLEM]->(Problem)
            - HAS_NEED: (Project)-[:HAS_NEED]->(Need)
            - PROVIDES_SOLUTION: (Project)-[:PROVIDES_SOLUTION]->(Solution)
            - SOLVES_PROBLEM: (Solution)-[:SOLVES_PROBLEM]->(Problem)
            - SIMILAR_TECHNOLOGY: (Project)-[:SIMILAR_TECHNOLOGY]->(Project)

            Best practices for Neo4j Cypher queries:
            1. For "common" questions: Look for nodes connected to multiple projects
            2. For specific project questions: Start with MATCH (p:Project {{name: 'ProjectName'}})
            3. For relationship questions: Use pattern matching with multiple relationships
            4. Always include RETURN statements with meaningful aliases
            5. For aggregations, use COUNT, COLLECT, etc.
            6. For sorting, use ORDER BY
            7. Limit large result sets with LIMIT clause

            Question: {question}
            
            Generate a correct, executable Cypher query that will answer this question:
            """,
            input_variables=["schema", "question"]
        )
        
        cypher_query = llm.invoke(
            cypher_prompt.format(
                schema=schema,
                question=question
            )
        ).strip()
        
        # Clean up the query to ensure it's valid Cypher
        if not cypher_query.upper().startswith("MATCH") and not cypher_query.upper().startswith("CALL"):
            # If response includes explanation text, try to extract just the query
            query_lines = []
            capture = False
            for line in cypher_query.split('\n'):
                if line.upper().strip().startswith("MATCH") or line.upper().strip().startswith("CALL"):
                    capture = True
                if capture:
                    query_lines.append(line)
            
            if query_lines:
                cypher_query = '\n'.join(query_lines)
            else:
                # Fallback to a safe query if we can't parse a valid Cypher query
                cypher_query = """
                MATCH (p:Project)
                OPTIONAL MATCH (p)-[:USES_TECHNOLOGY]->(t:Technology)
                OPTIONAL MATCH (p)-[:FACES_PROBLEM]->(pr:Problem)
                OPTIONAL MATCH (p)-[:HAS_NEED]->(n:Need)
                OPTIONAL MATCH (p)-[:PROVIDES_SOLUTION]->(s:Solution)
                RETURN p.name AS Project, 
                       COLLECT(DISTINCT t.name) AS Technologies,
                       COLLECT(DISTINCT pr.description) AS Problems,
                       COLLECT(DISTINCT n.description) AS Needs,
                       COLLECT(DISTINCT s.description) AS Solutions
                """
        
        return cypher_query
    
    # Create a function to execute Cypher and generate a final answer
    def execute_and_answer(cypher_query, question):
        # Execute the query
        try:
            result = graph.query(cypher_query)
            
            if not result or len(result) == 0:
                # If no results, try a more general query
                fallback_query = """
                MATCH (p:Project)
                OPTIONAL MATCH (p)-[:USES_TECHNOLOGY]->(t:Technology)
                OPTIONAL MATCH (p)-[:FACES_PROBLEM]->(pr:Problem)
                RETURN p.name AS Project, 
                       COLLECT(DISTINCT t.name) AS Technologies,
                       COLLECT(DISTINCT pr.description) AS Problems
                LIMIT 5
                """
                
                try:
                    result = graph.query(fallback_query)
                    cypher_query = fallback_query
                except:
                    result = [{"response": "No data found in the knowledge graph."}]
            
            # Generate an answer based on the results
            answer_prompt = PromptTemplate(
                template="""
                You are an expert in analyzing graph database results about technology projects.
                
                Question: {question}
                
                Cypher Query Used:
                {cypher_query}
                
                Query Results:
                {results}
                
                Your task:
                1. Analyze these results from the knowledge graph
                2. Provide a comprehensive, well-structured answer to the question
                3. Highlight patterns, connections, and insights from the graph structure
                4. If the results are limited, acknowledge this but provide the best possible answer
                5. Focus on relationships between entities - this is the strength of a graph database
                6. Format your answer with appropriate headings and bullet points
                
                Comprehensive Answer:
                """,
                input_variables=["question", "cypher_query", "results"]
            )
            
            answer = llm.invoke(
                answer_prompt.format(
                    question=question,
                    cypher_query=cypher_query,
                    results=str(result)
                )
            )
            
            return {
                "query": cypher_query,
                "result": result,
                "answer": answer
            }
        except Exception as e:
            error_msg = str(e)
            
            # Try to provide a helpful error message
            if "syntax error" in error_msg.lower():
                fallback_query = """
                MATCH (p:Project)
                RETURN p.name AS Project 
                LIMIT 5
                """
                
                try:
                    result = graph.query(fallback_query)
                    return {
                        "query": cypher_query,
                        "error": error_msg,
                        "answer": f"There was a syntax error in the query. Some projects in the database include: {[r['Project'] for r in result if 'Project' in r]}"
                    }
                except:
                    pass
            
            return {
                "query": cypher_query,
                "error": error_msg,
                "answer": f"There was an error executing the query: {error_msg}"
            }
    
    # Create a custom QA chain with these functions
    def qa_chain(query):
        cypher = generate_cypher(query)
        response = execute_and_answer(cypher, query)
        return {
            "query": query,
            "result": response["answer"],
            "intermediate_steps": [{"query": cypher, "result": response.get("result", [])}]
        }
    
    return qa_chain

def setup_vector_qa_chain(vectorstore):
    """Set up a RetrievalQA chain for the vector store."""
    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})
    
    prompt_template = """
    Human: Use the following pieces of context to provide a 
    concise answer to the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa

def hybrid_query(question, graph_qa, vector_qa, graph_weight=0.5):
    """Execute a true hybrid query that properly leverages both graph and vector approaches."""
    # Try the vector approach first
    try:
        vector_response = vector_qa({"query": question})
        vector_answer = vector_response["result"]
        vector_success = True
    except Exception as e:
        vector_success = False
        vector_answer = f"Vector search failed: {str(e)}"
    
    # Try the graph approach
    try:
        graph_response = graph_qa(question)
        graph_answer = graph_response["result"]
        graph_query = graph_response.get("intermediate_steps", [{}])[0].get("query", "No query available")
        graph_results = graph_response.get("intermediate_steps", [{}])[0].get("result", [])
        
        # Check if there was an error or empty result in the graph query
        if "error executing the query" in graph_answer.lower() or "there was an error" in graph_answer.lower():
            graph_success = False
        else:
            # Check if we got actual data
            if isinstance(graph_results, list) and len(graph_results) > 0:
                graph_success = True
            else:
                graph_success = False
    except Exception as e:
        graph_success = False
        graph_answer = f"Graph query failed: {str(e)}"
        graph_query = "Query failed"
        graph_results = []
    
    # Prepare the hybrid response based on what succeeded
    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={
        'max_gen_len': 1024,
        'temperature': 0.3
    })
    
    # Different scenarios based on what worked
    if graph_success and vector_success:
        # Both approaches worked - true hybrid
        hybrid_prompt = PromptTemplate(
            template="""
            You are an expert in synthesizing structured and unstructured information about technology projects.
            
            Question: "{question}"
            
            Graph Database Information (shows relationships and connections):
            {graph_answer}
            
            Document Search Information (provides detailed context):
            {vector_answer}
            
            Your task:
            1. Create a comprehensive answer that leverages BOTH information sources:
               - Use the graph information for relationships, patterns, and connections between entities
               - Use the document search for detailed descriptions and context
            2. Structure your answer with clear headings and sections
            3. Highlight technologies, challenges, solutions, and outcomes
            4. Draw conclusions that combine insights from both sources
            5. NEVER mention the sources of your information in your answer
            
            Comprehensive Answer:
            """,
            input_variables=["question", "graph_answer", "vector_answer"]
        )
        
        synthesized_answer = llm.invoke(
            hybrid_prompt.format(
                question=question,
                graph_answer=graph_answer,
                vector_answer=vector_answer
            )
        )
    elif vector_success and not graph_success:
        # Only vector worked - enhance vector results
        enhancement_prompt = PromptTemplate(
            template="""
            You are an expert in presenting information about technology projects in a structured and insightful way.
            
            Question: "{question}"
            
            Available information:
            {vector_answer}
            
            Your task:
            1. Create a comprehensive, well-structured answer based on the available information
            2. Organize the answer with clear headings, bullet points, and paragraphs
            3. Highlight key technologies, challenges, solutions, and outcomes
            4. Draw connections between different aspects (e.g., how technologies solve specific problems)
            5. Ensure the answer is complete and addresses all aspects of the question
            
            Enhanced Comprehensive Answer:
            """,
            input_variables=["question", "vector_answer"]
        )
        
        synthesized_answer = llm.invoke(
            enhancement_prompt.format(
                question=question,
                vector_answer=vector_answer
            )
        )
    elif graph_success and not vector_success:
        # Only graph worked - enhance graph results
        enhancement_prompt = PromptTemplate(
            template="""
            You are an expert in interpreting and explaining graph database results about technology projects.
            
            Question: "{question}"
            
            Graph Database Results:
            {graph_answer}
            
            Your task:
            1. Create a comprehensive explanation of the relationships and patterns found in the graph
            2. Organize your answer with clear headings and structure
            3. Highlight connections between projects, technologies, problems, and solutions
            4. Expand on the structured information to provide a complete answer
            5. Explain the significance of the relationships identified
            
            Enhanced Graph Analysis:
            """,
            input_variables=["question", "graph_answer"]
        )
        
        synthesized_answer = llm.invoke(
            enhancement_prompt.format(
                question=question,
                graph_answer=graph_answer
            )
        )
    else:
        # Neither worked - provide a helpful failure message
        synthesized_answer = f"""
        I couldn't find specific information to answer your question about "{question}". 
        
        This might be because:
        - The knowledge base doesn't contain information on this specific topic
        - The question might need to be rephrased to match available information
        - There might be technical limitations in accessing the information
        
        Please try:
        - Rephrasing your question with different terminology
        - Asking about a related but more general topic
        - Breaking your question into smaller, more specific parts
        """
    
    return {
        "question": question,
        "graph_answer": graph_answer,
        "vector_answer": vector_answer,
        "synthesized_answer": synthesized_answer,
        "graph_query": graph_query if 'graph_query' in locals() else "No query available"
    }

def main():
    st.set_page_config(layout="wide", page_title="Hybrid RAG System", page_icon=":brain:")
    st.title("Hybrid RAG System: Graph + Vector Search")

    # Initialize session state variables
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
    
    # Sidebar for connections and data loading
    with st.sidebar:
        st.subheader("1. Connect to Neo4j Database")
        neo4j_url = st.text_input("Neo4j URL:", value="neo4j+s://0fd8b8ba.databases.neo4j.io")
        neo4j_username = st.text_input("Neo4j Username:", value="neo4j")
        neo4j_password = st.text_input("Neo4j Password:", type='password',value="MIPXDxjuN-MupmAxLy0tHxyMTD62DFRPTUzVWzz-AqU")
        connect_button = st.button("Connect to Neo4j")

        if connect_button and neo4j_password:
            graph = initialize_graph(neo4j_url, neo4j_username, neo4j_password)
            if graph:
                st.session_state['graph'] = graph
                st.success("Connected to Neo4j!")
        
        st.markdown("---")
        
        st.subheader("2. Load Data")
        data_directory = st.text_input("Data Directory:", value="data")
        
        if st.button("Load Data and Create Indexes"):
            if 'graph' in st.session_state:
                with st.spinner("Loading data into Neo4j..."):
                    docs = load_data_into_neo4j(st.session_state['graph'], data_directory)
                    
                    if docs:
                        # Also create vector store from the same documents
                        vectorstore = create_vector_store(docs)
                        if vectorstore:
                            st.session_state['vectorstore'] = vectorstore
            else:
                st.error("Please connect to Neo4j first.")
        
        st.markdown("---")
        
        st.subheader("3. Load Existing Vector Store")
        if st.button("Load Existing Vector Store"):
            vectorstore = load_vector_store()
            if vectorstore:
                st.session_state['vectorstore'] = vectorstore
                st.success("Vector store loaded successfully!")

    # Main area for queries
    st.subheader("Ask a Question")
    
    # RAG method selection
    rag_method = st.radio(
        "Choose RAG Method:",
        ["Hybrid (Graph + Vector)", "Graph Only", "Vector Only"],
        horizontal=True
    )
    
    # Query input
    with st.form("query_form"):
        user_question = st.text_input("Enter your question:")
        enable_refinement = st.checkbox("Enable query refinement", value=True)
        submit_button = st.form_submit_button("Submit")
    
    # Process the query
    if submit_button and user_question:
        # Store the original question
        original_question = user_question
        
        # Optionally refine the query
        if enable_refinement:
            with st.spinner("Refining your query..."):
                refined_question = query_refinement(user_question, st.session_state.get('conversation_history', []))
                if refined_question != user_question:
                    st.info(f"Refined query: {refined_question}")
                    user_question = refined_question
        
        # Check if we have the necessary components
        if rag_method == "Graph Only" and 'graph' not in st.session_state:
            st.error("Graph database not connected. Please connect to Neo4j first.")
        elif rag_method == "Vector Only" and 'vectorstore' not in st.session_state:
            st.error("Vector store not loaded. Please load data or existing vector store first.")
        elif rag_method == "Hybrid (Graph + Vector)" and ('graph' not in st.session_state or 'vectorstore' not in st.session_state):
            st.error("Both graph database and vector store are required for hybrid queries. Please set up both.")
        else:
            # Process based on selected method
            with st.spinner("Generating answer..."):
                try:
                    start_time = time.time()
                    
                    # Initialize QA chains if needed
                    if 'graph' in st.session_state and 'graph_qa' not in st.session_state:
                        st.session_state['graph_qa'] = setup_graph_qa_chain(st.session_state['graph'])
                    
                    if 'vectorstore' in st.session_state and 'vector_qa' not in st.session_state:
                        st.session_state['vector_qa'] = setup_vector_qa_chain(st.session_state['vectorstore'])
                    
                    # Execute the appropriate query
                    if rag_method == "Graph Only":
                        response = st.session_state['graph_qa'](user_question)
                        answer = response["result"]
                        debug_info = {
                            "Cypher Query": response.get("intermediate_steps", [{}])[0].get("query", "No query available"),
                            "Process": "Graph Database Query"
                        }
                    
                    elif rag_method == "Vector Only":
                        response = st.session_state['vector_qa']({"query": user_question})
                        answer = response["result"]
                        debug_info = {
                            "Source Documents": [doc.page_content[:100] + "..." for doc in response.get("source_documents", [])],
                            "Process": "Vector Similarity Search"
                        }
                    
                    else:  # Hybrid
                        response = hybrid_query(
                            user_question, 
                            st.session_state['graph_qa'], 
                            st.session_state['vector_qa']
                        )
                        answer = response["synthesized_answer"]
                        debug_info = {
                            "Graph Answer": response["graph_answer"],
                            "Vector Answer": response["vector_answer"],
                            "Cypher Query": response["graph_query"],
                            "Process": "Hybrid Query (Graph + Vector)"
                        }
                    
                    processing_time = time.time() - start_time
                    
                    # Display the answer
                    st.markdown("### Answer:")
                    st.markdown(answer)
                    
                    # Add debug expander
                    with st.expander("Debug Information"):
                        st.markdown(f"**Original Question:** {original_question}")
                        st.markdown(f"**Processed Question:** {user_question}")
                        st.markdown(f"**Processing Time:** {processing_time:.2f} seconds")
                        st.markdown(f"**Query Process:** {debug_info['Process']}")
                        
                        if "Cypher Query" in debug_info:
                            st.markdown("**Cypher Query:**")
                            st.code(debug_info["Cypher Query"], language="cypher")
                        
                        if "Source Documents" in debug_info:
                            st.markdown("**Source Documents:**")
                            for i, doc in enumerate(debug_info["Source Documents"]):
                                st.markdown(f"Document {i+1}: {doc}")
                        
                        if "Graph Answer" in debug_info and "Vector Answer" in debug_info:
                            st.markdown("**Graph Answer:**")
                            st.markdown(debug_info["Graph Answer"])
                            st.markdown("**Vector Answer:**")
                            st.markdown(debug_info["Vector Answer"])
                    
                    # Update conversation history
                    st.session_state['conversation_history'].append((original_question, answer))
                
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
    
    # Display conversation history
    if 'conversation_history' in st.session_state and st.session_state['conversation_history']:
        with st.expander("Conversation History", expanded=False):
            for i, (q, a) in enumerate(st.session_state['conversation_history']):
                st.markdown(f"**Question {i+1}:** {q}")
                st.markdown(f"**Answer {i+1}:** {a}")
                st.markdown("---")

if __name__ == "__main__":
    main()
