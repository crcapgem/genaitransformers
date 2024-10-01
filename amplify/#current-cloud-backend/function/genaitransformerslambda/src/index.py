import json
import boto3
import pandas as pd
import re
import os
from requests_aws4auth import AWS4Auth
from langchain_aws import BedrockEmbeddings
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain_aws import ChatBedrock
import logging

OPENSEARCH_HOST = 'iellhhrn6kean028im78.us-east-1.aoss.amazonaws.com'
OPENSEARCH_PORT = 443
OPENSEARCH_REGION = 'us-east-1'
OPENSEARCH_SERVICE = 'aoss'
BEDROCK_MODEL_ID = 'amazon.titan-embed-text-v2:0'
INDEX_CONFIG ={'Hackathon_index': 'testing_index3'}
INDEX_NAME = INDEX_CONFIG['Hackathon_index']

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def load_model():
    # bedrock client
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )
    # Update to use amazon.titan-text-premier-v1:0
    model_id = "amazon.titan-text-premier-v1:0"
    model_kwargs = {
        "temperature": 0.0,
    }
    # Bedrock chat model
    model = ChatBedrock(
        client=bedrock_runtime,
        model_id=model_id,
        model_kwargs=model_kwargs,
    ).with_retry()
    return model

def query_router(query):
    prompt =  f"""
    Your task is to Classify the following texts into the appropriate categories. 
    The categories to classify are: 
    -alert
    -trasnformer
    
    Examples: 
    User: provide comparative analysis of transformers - 2217, 1235, 1685 and 2484 based on their parameters
    Bot: transformer
    
    User: How is my transformer fleet - 2217, 1235, 1685 and 2484 doing?
    Bot: transformer
    
    User: which transformer needs inspection currently and for what ? ignore missing data
    Bot: alert
    
    User: When will a transformer fail? which one first
    Bot: alert
    Classify the following request: {query}"""
    
    model = load_model()
    response = model.invoke(prompt)
    content = response.content
    
    return content

def parse_keywords(content):
    txid_keywords = []
    must_keywords = []
    should_keywords = []

    txid_line = re.search(r'TXIDs: \[(.*?)\]', content)
    if txid_line:
        txid_keywords = [keyword.strip() for keyword in txid_line.group(1).split(',')]
    must_line = re.search(r'Numerical Keywords: \[(.*?)\]', content)
    if must_line:
        must_keywords = [keyword.strip() for keyword in must_line.group(1).split(',')]
    should_line = re.search(r'Text Keywords: \[(.*?)\]', content)
    if should_line:
        should_keywords = [keyword.strip() for keyword in should_line.group(1).split(',')]

    return txid_keywords, must_keywords, should_keywords

def extract_keywords(query):
    prompt = f"""
    Extract the keywords from the following query: "{query}"
    Keywords should include specific identifiers like transformer numbers, date, or other entities relevant to the query. 
    All text keywords should be noun. Transformer should never be a keyword.
    If the keyword is clearly a transformer ID and it doesn't start with 'TXID_', add this to its prefix
    The output should have format of: TXIDs:[...]; Numerical Keywords: [...]; Text Keywords: [...]
    
    Example
    Input: provide comparative analysis of transformers - 2217, 1235, 1685 and 2484 based on their parameters?
    Output: TXIDs: [TXID_2217, TXID_1235, TXID_1685, TXID_2484]; Numerical Keywords: [...]; Text Keyword: …
    
    Input: what is the Precipitation value for transformer 2484
    Output: TXIDs: [TXID_2484]; Numerical Keywords: [...]; Text Keyword: [Precipitation]

    Input: which transformer that was maintenaned on 2023-12-20
    Ouput: TXIDs: []; Numerical Keywords: [2023-12-20]; Text Keyword: [maintenaned]
    """
    
    model = load_model()
    response = model.invoke(prompt)
    
    content = response.content
    print(content)
    
    return parse_keywords(content)

def keyword_search(opensearch_client, index_name, query):
    txid_keywords, must_keywords, should_keywords = extract_keywords(query)

    # print(txid_keywords)
    msearch_body = []
    if txid_keywords and any(txid_keywords):
        for keyword in txid_keywords:
            msearch_body.append({'index': index_name})
            
            msearch_body.append({
                'size': 1, 
                'query': {
                    'bool': {
                        'must': [
                            {
                                'match_phrase': {
                                    'content': keyword
                                }
                            }
                        ]
                    }
                }
            })
    else: 
        should_queries = [
            {'match': {'content': keyword}} for keyword in should_keywords 
        ]
        must_queries = [
            {'match_phrase': {'content': keyword}} for keyword in must_keywords 
        ]
        msearch_body.append({'index': index_name})
        msearch_body.append({
                    'size': 5, 
                    'query': {
                        'bool': {
                            'must': must_queries,
                            'should': should_queries,
                            'minimum_should_match': 0
                        }
                    }
                })
    print(msearch_body)
    response = opensearch_client.msearch(body=msearch_body)

    all_results = []
    for res in response['responses']:
        if 'hits' in res and 'hits' in res['hits']:
            all_results.extend(res['hits']['hits'])  

    return all_results

def vector_search(opensearch_client, index_name, embedding_model, query, k=2):
    query_vector = embedding_model.embed_documents([query])[0]
    response = opensearch_client.search(
        index=index_name,
        body={
            'size': k,
            'query': {
                'knn': {
                    'embedding': {
                        'vector': query_vector,
                        'k': k
                    }
                }
            }
        }
    )

    similarity_threshold = 0.75
    # Filter results based on similarity threshold
    filtered_results = [
        result for result in response['hits']['hits'] if result['_score'] >= similarity_threshold
    ]
    
    return filtered_results

def alert_search(opensearch_client, index_name, k=10):
    response = opensearch_client.search(
        index=index_name,
        body={
            'size':k,
            'query':{
            'nested':{
              'path':"attributes",
                  'query':{
                    'bool':{
                      'should':[
                            {
                              'range':{
                                'attributes.Health index':{
                                  'gt':70
                                                          }  
                                      }
                            },
                            {
                              'range':{
                                'attributes.Power factor':{
                                  'gt':0.3
                                                          }
                                      }
                            },
                            {
                              'range':{
                                'attributes.Dielectric rigidity':{
                                   'lt':20
                                                                 }
                                      }
                            },
                            {
                              'range':{
                                'attributes.C2H2_H2_ratio':{
                                  'gt':0.5
                                                           }
                                      }
                            },
                            {
                              'range':{
                                'attributes.C2H4_H2_ratio':{
                                  'gt':0.5
                                                           }
                                      }
                            },
                            {
                              'range':{
                                'attributes.CH4_H2_ratio':{
                                  'gt':0.5
                                                          }
                                      }
                            }
          ]
        }
      }
              }  
          }
            
        }
    )
    return response['hits']['hits']

def print_query_result(query, results):
    search_results = ""
    
    if not results:
        search_results += f"# Query: {query} (search)"
        search_results += "--------------------------------"
        search_results += "No results found."
        search_results += "--------------------------------"
        print(search_results)
        return search_results

    df = pd.concat([pd.DataFrame([result['_source'] if isinstance(result, dict) else result]) for result in results], ignore_index=True)
    search_results += f"# Query: {query} (search)"
    search_results += "--------------------------------"
    for i, result in enumerate(results):
        # print(result)
        metadata = result['_source'] if isinstance(result, dict) else result
        search_results += f"# Search result {i+1} (relevant document chunk):"
        search_results += f"Source: {metadata['source']}"
        search_results += "Content:"
        row_content = json.loads(metadata['content'])
        for key, value in row_content.items():
            search_results += f"'{key}': {value}"
        search_results +=  "--------------------------------"
    return search_results

def test_queries(index_name, query, opensearch_client, embedding_model, search_type):

    results = ""
    if search_type == 'transformer': #Hybrid
        # print(f"\nTesting query {query} - Hybrid Search:")
        results = keyword_search(opensearch_client, index_name, query) + vector_search(opensearch_client, index_name, embedding_model, query)
    elif search_type == 'alert':
        # print('Hit alerting logic')
        results = alert_search(opensearch_client, index_name)
    search_result = print_query_result(query, results)


    formatted_output = format_output(query, [search_result])
    # print(formatted_output)
    return formatted_output

def format_output(query, results):
    formatted_results = []
    for result in results:
        formatted_results.append(f"<documents>\n{result}\n</documents>")
    return f"<query> # Query: {query} </query>\n" + "\n--------------------------------\n".join(formatted_results)

# Initialize OpenSearch client
def init_opensearch_client(host, port, region, service):
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)
    return OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30000
    )

def generate_response(model, user_prompt, system_prompt):
    input_prompt = f"System: {system_prompt}\n\nHuman: {user_prompt}\n\nAI:"
    
    response = model.invoke(
        input=input_prompt
    )
    
    return response

def clean_response(raw_response):
    raw_text = raw_response.content
    cleaned_response = raw_text.strip()
    
    return cleaned_response

# Initialize OpenSearch client outside the handler for reuse
opensearch_client = init_opensearch_client(OPENSEARCH_HOST, OPENSEARCH_PORT, OPENSEARCH_REGION, OPENSEARCH_SERVICE)
embedding_model = BedrockEmbeddings(client=boto3.client("bedrock-runtime", region_name=OPENSEARCH_REGION), model_id=BEDROCK_MODEL_ID)
model = load_model()

def handler(event, context):
    print('received event:')
    print(event)
    logger.info("Lambda function invoked with event: %s", event)  # Log event details

    query = event.get('query', '') # Get the query from the event
    logger.info("Received query: %s", query)  # Log the query

    try:
        search_type = query_router(query)
        logger.info("Query classified as: %s", search_type)

        search_result = test_queries(INDEX_NAME, query, opensearch_client, embedding_model, search_type)
        logger.info("Search result: %s", search_result)

        user_prompt = search_result
        system_prompt = """You are a specialized assistant trained to provide detailed information on power transformers. You have access to a comprehensive dataset containing operational, environmental, and performance data for transformers. Your role is to provide accurate, detailed, and context-specific information in response to user queries.
        Instructions:

        1. If a query contains specific transformer IDs:
        - Present details for those transformers, including health index, gas ratios, maintenance details, and temperature readings.
        - Provide comparative tabular analysis if multiple transformers are mentioned.

        2. If no transformer ID is provided:
        - Search for the most relevant transformer based on operational parameters (health index, gas ratios, oil quality, temperature, power factor).
        - Provide a reasoned conclusion or advice for transformer health based on the most similar transformers.

        3. Response Format:
        - Use a tabular format where comparison is requested.
        - Mention critical insights into dissolved gas analysis (DGA), oil quality tests, power factor, and any operational incidents.
        - If the transformer shows signs of failure, suggest actions based on the specific parameters.

        4. Maintain factual accuracy by relying strictly on the dataset provided. Do not offer unverifiable or speculative information.

        5. If the search results do not contain information that can answer the 
        question, please state that "I could not find an exact answer to the 
        question."

        6. Always include a reference at the end of your response.
        """
        response = generate_response(model, user_prompt, system_prompt)
        cleaned_response = clean_response(response)

        logger.info("Generated response: %s", cleaned_response)
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps({'response': cleaned_response})
        }
    
    except Exception as e:
        logger.error("Error processing query: %s", str(e))  # Log exceptions
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps({'error': 'An error occurred while processing the request.'})
        }
