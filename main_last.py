from flask import Flask, request, jsonify
import requests
import pandas as pd
from rich import print
from warnings import filterwarnings
from bs4 import BeautifulSoup
import emoji
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.preprocessing import normalize
import faiss

filterwarnings('ignore')

app = Flask(__name__)

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
import json

index = faiss.read_index('/content/real_estate_final_index.faiss')
properties = pd.read_csv('/content/modified_properties_new.csv')
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2', cache_dir='./cache_L')
model = AutoModel.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2', cache_dir='./cache_L')

def get_q(msg):
    llm = ChatOpenAI(temperature=0, model='gpt-4-turbo-2024-04-09',
                       api_key="sk-proj-pHkywA2mDUiHGcCWRM4TT3BlbkFJz3qlbeGgfG8C70iHX7WK")
    memory = ConversationBufferMemory(memory_key="get_bed", return_messages=True)
    sys_prompt = f"""The user says: "{msg}", Just incase the text is long and not straightforward, Look for the main 
    information in the user's input and extract it. main information is the number of bedrooms, so, anything the user type
    just take note of the number of bedrooms and return something like 3 bedroom or 2 bedroom, and map it with bedroom
    
    also extract the location and map it with location, also extract if the house is for Rent or for Sale and map it to sub_type,
    if for rent just return Rent or for sale just return Sale
    then return the whole words and map it to query
    finally return a dictionary... only return the dict.. no suffix or prefix, no additional message! the dict is all we need
    and return just empty string if any of the three keywords aren't found, and return others that are found
    if any of the key parameter is passed, just automatically pass none and map it to the key that the value is not provided
    for example the user might not mention of they're looking for a house that's for rent or for sale.. in that case, just pass the
    unknown keyword to the sub_type, and if they didn't mention number of bedroom they want, just pass unknown keyword to bedroom
    likewise for location too.
    extract the price and map it to price key, map the price key to unknown if it's not found in the sentence
        prices can be a number in full like 1000, 10,000, 1,000,000 or a short form such as 1K, 10K, 1M, 1B, etc 
        Take note of price modifiers such as less than, more than, for etc...
    extract the agent name and map it to agent key, map the agent key to unknown if it's not found in the sentence
    extract the property type and map it to property, map the property key to unknown if it's not found in the sentence,
    user might input only the agent names for examples:
        ['ACE REALTORS', 'Agnes Mukami', 'Alvin Mahindi', 'Ann Maina',
           'Anne wambui ngugi', 'Berach Dimensions', 'Beritah Nabwile',
           'Brian Pareno', 'Brian Wafula', 'Brian kiiru', 'Charity Peris',
           'Christopher Oyuech', 'Daniel Wahome', 'Daniel okemwa biyaya',
           'David Muna', 'Dennis', 'Dick Nesto',
           'Equity Lifestyle Properties',
           'Esther Njoroge REMAX Proffesional Agent', 'Felistus Gichia',
           'Fiona K', 'FoQus reality Homes kileleshwa', 'Gladys Muyanga',
           'Gloria Kioko', 'Haaften properties', 'Homeken Limited',
           'James Nyumu', 'Juliet Njeri', 'Kenneth Wanguya', 'Kevin Mugwe',
           'Kyalo Muli', 'Linda Muto', 'Lore Enkaji', 'MAGGYDALMA',
           'Majestic Homes Kenya', 'Mandela Kivondo', 'Melmay Properties',
           'Michelle Savethi Kilonzo', 'Patrick Kariuki Mwangi', 'Paul',
           'Pauline Mungai', 'Peris Gichere', 'Shadrack Mutuku',
           'Simon Njoroge', 'Smart Focus', 'Stable Merchants',
           'Victor Mbugua', 'Wabacha Mbaria', 'Winnie Chemutai',
           'chris otieno', 'jane karanja', 'joan nyathore', 'joan omanyo',
           'ken', 'pemaka limited', 'purity kathure', 'sam magiya',
           'webach properties']
           those agent names might be in small letter, or just a single word, it may be just their firstname or just their lastname, then you still need to return it as the agent name without extra white spaces.
    then you can return that as the agent.. and since it might usually be just two keywords, you can return only that as agent name, not the query, then query can be blank.
    user may mention mansion as property type, just return maisonette.
    users may just want to search by a single keyword, make sure you're not returning empty dictionary, and find patterns to what maps.
    they may input just ace, realtors, haaften, nyathore, sam, peris and so on... those are still agent names
    as for the price, user might mention something like 100K or 1m or 30K or 1b or fifty or hundred ... you should convert those to 0s... 50K = 50,000, 1m = 1,000,000 and so on.
    also we need to map the belows as well:
    if user mentions with gated community, map gated_commumity to Yes, and No otherwise, and if they specify none, just map to unknown.
    if user mentions owned or own compound , map own_compound to Yes, and No otherwise, and if they specify none, just map to unknown.
    if user searches something similar to landlord, then map submitted_by to owner, otherwise agent if they specify agent, and if they specify none, just map to unknown
    """
    prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(sys_prompt),
                                               MessagesPlaceholder(variable_name="get_bed"),
                                               HumanMessagePromptTemplate.from_template(f"{msg}")])
    conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)

    memory.chat_memory.add_user_message(msg)
    response = conversation.invoke({"text": msg})
    return json.loads(response['text'])


# Function to clean HTML tags from string
def clean_text(raw_html):
    clean_text = BeautifulSoup(raw_html, "html.parser").get_text()
    clean_text = clean_text.replace('📌', '').strip()
    clean_text = emoji.replace_emoji(clean_text, replace='')
    return clean_text
    

# Function to embed text
def embed_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=50)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def calculateInnerProduct(L2_score):
    import math
    return (2 - math.pow(L2_score, 2)) / 2

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great-circle distance between two points 
    on the Earth's surface using the Haversine formula.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    r = 6371  # Radius of Earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r
def calculate_dynamic_weight(query, data):
    location_weight = 1.0  # Default weight

    if query['location']:
        location_weight = 2.0  # Increase weight if location is specified

    # Adjust weights based on proximity
    if 'distance' in data.columns:
        data['location_weight'] = np.exp(-0.1 * data['distance']) * location_weight
    else:
        data['location_weight'] = location_weight

    return data
def expand_search_radius(query, data, max_radius=50):
    ref_lat = data.iloc[0]['lat']
    ref_lon = data.iloc[0]['lon']
    
    if 'distance' in data.columns:
        within_radius = data[data['distance'] <= max_radius]
        
        if within_radius.empty:
            # Gradually increase the search radius
            for radius in range(50, 201, 50):
                within_radius = data[data['distance'] <= radius]
                if not within_radius.empty:
                    break
    
        return within_radius
    else:
        return data
    
def optimized_faiss_search(query, index, tokenizer, model, data, topk=20, nprobe=5):
    # Set nprobe dynamically based on distance
    index.nprobe = nprobe if query['location'] else 1
    
    description_embedding = embed_text(query['query'], tokenizer, model)
    
    # Additional embeddings if required
    # Combine embeddings into a single query vector
    query_embedding = np.hstack([description_embedding])
    
    dim = query_embedding.shape[0]
    query_embedding = query_embedding.reshape(1, dim)
    faiss.normalize_L2(query_embedding)
    
    D, I = index.search(query_embedding, topk)
    
    indices = I[0]
    L2_score = D[0]
    inner_product = [calculateInnerProduct(l2) for l2 in L2_score]
    
    matching_data = data.iloc[indices]
    
    search_result = pd.DataFrame({
        'index': indices,
        'cosine_sim': inner_product,
        'L2_score': L2_score
    })
    
    dat = pd.concat([matching_data.reset_index(drop=True), search_result.reset_index(drop=True)], axis=1)
    return dat
def refined_sorting_logic(dat, query):
    sort_criteria = [
        f"is_{query['location'].lower()}", 
        f"is_{query['agent'].lower()}", 
        f"is_{query['sub_type'].lower()}", 
        f"is_{query['bedroom'].lower()}", 
        f"is_{query['property'].lower()}"
    ]
    sort_ascending = [False, False, False, False, False]

    # Apply dynamic weighting to sort by location proximity
    if 'location_weight' in dat.columns:
        sort_criteria.insert(0, 'location_weight')
        sort_ascending.insert(0, False)

    sort_criteria.extend(['agent.name', 'location', 'submission_type', 'property_type', 'bedrooms_numeric', 'distance'])
    sort_ascending.extend([True, True, True, True, True, True])

    dat = dat.sort_values(by=sort_criteria, ascending=sort_ascending)
    
    return dat