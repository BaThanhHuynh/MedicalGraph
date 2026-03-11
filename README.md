MedicalGraph: Symptom Extraction and ICD-10 Knowledge Graph
MedicalGraph is an end-to-end data pipeline designed to extract medical symptoms from unstructured text, map them to the ICD-10 (International Classification of Diseases) standard, and transform the results into a structured Knowledge Graph using Neo4j.

Getting Started
Follow these steps to set up the environment and execute the data processing pipeline.

Prerequisites
Python 3.8+

Neo4j Desktop or a running Neo4j Aura instance.

Raw medical data (e.g., JSON/CSV extracted from sources like Vinmec).

Step 1: Installation
Clone this repository and install the required dependencies:

Bash

pip install -r requirements.txt
Step 2: Symptom Extraction
Run the NLP engine to identify and extract relevant symptoms from your raw data:

Bash

python extract_symptoms.py
Step 3: Preliminary ICD-10 Mapping
Map the extracted entities to the global ICD-10 medical coding standard:

Bash

python map_icd10.py
Step 4: Final Graph Integration
Execute the final script to clean the mappings and push the structured data into your Neo4j database:

Bash

python map_icd.py
Usage and Querying (Neo4j)
Once the data is imported, you can explore the relationships using Cypher queries in the Neo4j Browser.

1. Visualize Disease-Symptom Relationships
To see the connections between diseases and their identified symptoms:

Cypher

MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
RETURN d, r, s
LIMIT 50
2. Search by ICD-10 Code
To find a specific disease and its symptoms using a standard code:

Cypher

MATCH (d:Disease {icd10_code: 'A15'})-[:HAS_SYMPTOM]->(s:Symptom)
RETURN d.name, s.name
Features and Technologies
NLP Pipeline: Custom extraction of medical entities from text.

Standardization: Automated mapping to ICD-10-CM.

Graph Database: High-performance storage and visualization using Neo4j.

Scalability: Designed to handle large datasets from medical web scraping.

Configuration
Before running Step 4, ensure your Neo4j connection settings are configured:

NEO4J_URI: neo4j://localhost:7687

NEO4J_USER: neo4j

NEO4J_PASSWORD: your_secure_password

Future Roadmap
Topic Modeling: Implementing LDA to identify emerging medical trends.

Predictive Analytics: Developing a model to predict potential disease risks based on symptom clusters.

Dashboard Integration: Building a web interface to visualize the graph in real-time.

Author
Your Name - Data Mining & Software Development

GitHub: [github.com/yourusername]

LinkedIn: [linkedin.com/in/yourprofile]

License
This project is licensed under the MIT License - see the LICENSE file for details.