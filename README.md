
# Hybrid RAG with AWS and Langchain
[![Hybrid RAG with AWS and Langchain - YouTube](https://img.youtube.com/vi/lc2oy-W-eoY/0.jpg)](https://youtu.be/lc2oy-W-eoY)

## Inspiration
My inspiration and encouragement to initiate and finish this project was when I lost a hackathon and could not finish creating a RAG system that was going to assist rural children in availing themselves of personalized education. The initial idea was to produce quizzes and learning materials based on the strengths and weaknesses of individual students. That experience compelled me to come up with a more comprehensive Hybrid RAG system that could be implemented in educational as well as business settings.

## What it does
This project deploys a Hybrid Retrieval Augmented Generation (RAG) system with AWS services and Langchain. It uses both vector-based and graph-based retrieval approaches to develop a more holistic information retrieval system. The system:

1. Processes company data and project data  
2. Stores information in vector and graph databases  
3. Retrieves information using parallel execution of both retrieval approaches  
4. Synthesizes responses based on the combined retrieved information  
5. Assists people in an organization to enhance their performance by personalized information retrieval  

## How we built it
We developed the Hybrid RAG system with the following components:

### 1. **Graph Database Implementation**:
- Employed Neo4j to store data as nodes and relations  
- Utilized AWS Bedrock for mapping text data into computer-readable format  
- First tried to utilize OpenAI but switched to AWS Bedrock because of rate limitations and improved batch processing functionality  

### 2. **Implementation of Vector Database**:
- Utilized AWS Bedrock's Titan Embeddings model to convert text to dense numerical vectors  
- Saved embeddings in vector databases (Amazon OpenSearch, Pinecone, or FAISS)  
- Applied semantic search functionality for efficient similarity matching  

### 3. **Hybrid RAG Architecture**:
- Built parallel execution of vector and graph queries to minimize latency  
- Incorporated advanced error handling to identify which components returned successful results  
- Built conditional response synthesis based on the success of each component  
- Employed strategic prompt engineering for successful answer synthesis  

## Challenges we encountered
During the development process, we faced a number of challenges:

1. **Limitations in Data Processing**: Originally experienced rate limitations with OpenAI when processing large data batches, prompting the use of AWS Bedrock.  
2. **Complexity of Integration**: Integrating the vector-based and graph-based methods needed to be done with careful architecture planning and error management.  
3. **Synthesis of Response**: Developing a coherent response that properly integrates information from both retrieval methods needed advanced prompt engineering.  
4. **Optimization of Performance**: Tying the recall-precision trade-off between the two retrieval approaches to achieve both accuracy and comprehensiveness.  
5. **Scalability Issues**: Making the system scalable enough to support large corporate datasets without compromising response speed and quality.  

## Achievements that we're proud of
1. Successfully created a Hybrid RAG system that performs better than conventional single-method RAG systems.  
2. Designed a dual-retrieval pipeline that preserves both implicit contextual relevance and explicit relational dependencies.  
3. Exemplified better performance in key evaluation measures like faithfulness, entity recall, and answer completeness over baseline RAG deployments.  
4. Developed a system that can handle ambiguity, resolve coreferences, and infer implicit relations more reliably.  
5. Used AWS services to build a scalable and maintainable solution effectively.  

## What we learned
By working on this project, we learned:

1. The shortcomings of conventional RAG architectures and how hybrid methods can overcome these limitations.  
2. The complementarity of vector-based and graph-based retrieval systems.  
3. Efficient utilization of AWS services for big data processing and retrieval.  
4. Strategic prompt engineering methods for answer synthesis.  
5. The significance of parallel execution and advanced error handling in hybrid systems.  
6. How to compare and assess various RAG models using different performance metrics.  

## Agentic RAG's next steps with AWS and Langchain
The following are our future areas for this project:

1. Hybrid RAG extension to deal with multi-modal input retrieval (tabular, image, temporal).  
2. Building application-specific graph pipeline construction methods.  
3. Adding context-perceptive reasoning units for live decision-making support.  
4. Adding streaming input sources for even more dynamic information retrieval.  
5. Enhancing underlying prompt orchestrations for more effective response generation by LLMs.  
6. Turning Hybrid RAG into a fully fledged adaptable system for application in complex enterprise scenarios.  
7. Going back to the original learning application in order to build student-specific quiz generators based on detecting and covering their weak points.  

These extensions, we hope, will help Hybrid RAG evolve into an adaptive platform capable of high-fidelity information generation and recall for different domains of industry.
