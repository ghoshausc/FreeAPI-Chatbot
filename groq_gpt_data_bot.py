import time, sys, re, csv, json, time, io, ast, requests, demjson3, os
import pandas as pd
import numpy as np
import openai, constants


from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch



###################### Place all Flask imports!
######## Flask imports
import flask
from flask import Flask, render_template, request, jsonify
import werkzeug
from werkzeug.exceptions import HTTPException
import json
import urllib.parse
from flask_cors import CORS, cross_origin



app = Flask(__name__)

CORS(app, support_credentials=True)

app.config['JSON_SORT_KEYS'] = True

app.json.sort_keys = False


###################### Create another .py file that will store the embeddings of each column. Refer to the ipynb Conversational_data.ipynb


openai.api_key = constants.OPENAI_API_KEY


########################################### Better ways of printing! :)  ###########################################

def print_flush(*args, **kwargs):
    
    print(*args, **kwargs)
    
    sys.stdout.flush()


####### Groq Imports and key stuffs!
from groq import Groq


os.environ["GROQ_API_KEY"] = constants.GROQ_API_KEY

client = Groq()

################### In place of Groq, you can use Cerebras which is free up to 30 requests per minute!

# client = Cerebras(
#     api_key=constants.CEREBRAS_API_KEY
# )

################## Or even Open Router API! Please visit link : https://openrouter.ai/docs/api-reference/limits



################################################################## STATIC STUFFS! ##################################



################################################################## Lets load the DataFrame from the CSV file!



################################################################## Loading the DataFrame from PArquet file created by data_embeddings_for_product_bot.py


df = pd.read_parquet("product_bot_data.parquet")

def load_embeddings_from_parquet(parquet_file):

    
    df_embeddings = pd.read_parquet(parquet_file)

    
    column_embeddings = {}

    
    for col in df_embeddings['column'].unique():
        
        embeddings = np.array(df_embeddings[df_embeddings['column'] == col]['embedding'].tolist())
        
        
        column_embeddings[col] = embeddings

    return column_embeddings


start_load = time.time()
loaded_embeddings = load_embeddings_from_parquet("column_embeddings.parquet")
end_load = time.time()


print_flush("Time taken to load embeddings is : ",end_load - start_load,"\n\n")

###################### Loading the model for fetching related content!

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

###################### Column meanings explained for better answers!


important_columns = list(df.columns)

column_meanings = {
    "DR Number": "The ID of the traffic accident",
    "Date Reported": "The date when the accident was reported",
    "Date Occurred": "The date when the accident occurred",
    "Time Occurred": "The time when the accident occurred",
    "Area ID": "The ID of the area where the accident occurred",
    "Area Name": "The name of the area where the accident occurred",
    "Reporting District": "The name of the district reporting the accident",
    "Crime Code": "The code of the crime committed",
    "Crime Code Description": "The description of the crime committed",
    "MO Codes": "The Modus Operandi Code",
    "Victim Age": "The age of the victim",
    "Victim Sex": "The sex of the victim",
    "Victim Descent": "The original descent of the victim",
    "Premise Code": "The code of the premise where the accident occurred",
    "Premise Description": "The description of the premise where the accident occurred",
    "Address":"The address where the accident occurred",
    "Cross Street":"The cross street where the accident occurred",
    "Location":"The exact coordinates(latitude,longitude) of the place where the accident occurred"
}

###################### Column meanings explained for better answers!

########################################### Using GPT to break a question into multiple questions! ###########################################


########################################### Defining the Answer Template for GPT!

answer_template = dict()


answer_template["Input"] = "How many accidents took place in 2019? How many of them were female? What were the crimes?"

answer_template["Output"] = ["Identify accidents that took place in 2019 year","Check the victims genders","Check the crime codes and descriptions of these accidents"]


def get_steps_to_answer_question(user_query):

    
    messages = []
    
    messages.append({"role":"system","content":f''' For the input you get, break the question into single-condition questions that when answered IN ORDER will answer the original question. Please note every small answer should contain only ONE condition, not more. See example here : for input {answer_template["Input"]}, your output is {answer_template["Output"]}. Note that every answer in {answer_template["Output"]} has only ONE condition. Do not use ambiguous words like `similar`,`previous`,`next`. Use words that DO NOT have any reference to any previous or next questions. Your output should be meaningful.  Do not give verbose answers.  '''  })

    messages.append({"role":"user","content":f'''Similarly, provide the output for input {user_query} '''})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.6
        )

        cost, output = 0.0, ""
    
        if response:
            
            cost = response.__dict__["_previous"]["usage"]["prompt_tokens"]*0.15/1000000 + response.__dict__["_previous"]["usage"]["completion_tokens"]*0.60/1000000
    
            output = response.choices[0].message['content'].replace("`", "").replace("json", "")
            
        
    except Exception as e:

        return str(e), 0.0

    
    return output, cost



def get_steps_to_answer_question_groq(user_query):

    messages = []

    start_groq = time.time()
    
    messages.append({"role":"system","content":f''' For the input you get, break the question into single-condition questions that when answered IN ORDER will answer the original question. Please note every small answer should contain only ONE condition, not more. See example here : for input {answer_template["Input"]}, your output is {answer_template["Output"]}. You will use these columns while breaking the input question. Note that every answer in {answer_template["Output"]} has only ONE condition. Do not use ambiguous words like `similar`,`previous`,`next`. Use words that DO NOT have any reference to any previous or next question. Your output should be meaningful. For every input, you will only provide the column name like {answer_template["Output"]}. Do not use any other text in your answer. Try to break the question into smaller questions optimally i.e. the smaller th enumber of sub-questions, the better. Do not give verbose answers.  '''  })

    messages.append({"role":"user","content":f'''Similarly, provide the output for input {user_query} '''})

    start = time.time()
    
    answer = ""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", ###### check the model name for Cerebras here.. https://cloud.cerebras.ai/platform/org_repdydkp566nht4ymvj4wt4d/playground
            
            messages=messages,
            top_p=1,
            stream=True,
            stop=None,
        )
        
        for chunk in completion:
        
            content = chunk.choices[0].delta.content or ""
            
            answer += content
    
        end_groq = time.time()
    
        print_flush(f"\n\n Inside get_steps_to_answer_question_groq, Groq took time : {end_groq - start_groq} \n\n")

    except Exception as e:

        print_flush(f"\n\n Groq Error inside get_steps_to_answer_question_groq is {str(e) } \n\n")
        answer = ""
        
    return answer, 0.0


    
########################################### Based on the question, which is the most related column? ###########################################


def get_me_the_column_groq(gpt_question):

    messages = []

    template = dict()

    start = time.time()

    template["Input"] = "How did the accident take place?"
    
    template["Output"] = "MO Codes"

    messages.append({"role":"system","content":f''' You will choose ONLY 1 most similar column from the list {list(loaded_embeddings.keys())} based on the question {gpt_question}. See meaning of each column here : {column_meanings}. For example, for question {template["Input"]}, your answer is {template["Output"]}. Do not give verbose answers. '''  })

    messages.append({"role":"user","content":f''' Similarly, provide output for {gpt_question}'''})

    answer = ""

    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=messages,
            top_p=1,
            stream=True,
            stop=None,
        )
         
        for chunk in completion:
        
            content = chunk.choices[0].delta.content or ""
            
            answer += content
            
    except Exception as e:
        print_flush(f"\n\n Groq had an error with get_me_the_column_groq : {str(e)} \n\n")
        answer = ""

    end = time.time()

    print_flush(f"\n\n Inside get_me_the_column_groq, Groq took time : {end - start} \n\n")
    
    return answer, 0.00



def get_me_the_column(gpt_question):

    
    messages = []

    template = dict()

    template["Input"] = "How did the accident take place?"
    
    template["Output"] = "MO Codes"

    messages.append({"role":"system","content":f''' You will choose ONLY 1 most similar column from the list {list(loaded_embeddings.keys())} based on the question {gpt_question}. See meaning of each column here : {column_meanings}. For example, for question {template["Input"]}, your answer is {template["Output"]}. Do not give verbose answers. '''  })

    messages.append({"role":"user","content":f''' Similarly, provide output for {gpt_question}'''})

    try:
        
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.5
        )
        
        cost, output = 0.0, ""
    
        if response:
    
            cost = response.__dict__["_previous"]["usage"]["prompt_tokens"]*0.15/1000000 + response.__dict__["_previous"]["usage"]["completion_tokens"]*0.60/1000000
    
            output = response.choices[0].message['content'].replace("`", "").replace("json", "").replace("'","")
                        
        
    except Exception as e:

        return str(e), 0.0

    return output, cost




########################################### Based on each small question, get the top related items ###########################################


def get_top_K_items_for_question(stored_embeddings, user_query):

    #################### Embedding the user's qiuetion first with Sentence transformer (same model : paraphrase)!

    user_query_embedding = torch.tensor(model.encode([user_query], device='cpu'))

    ################## ready to get most related top K indices!

    stored_embeddings = torch.tensor(stored_embeddings,dtype=torch.float32)

    similarities = torch.cosine_similarity(user_query_embedding, stored_embeddings, dim=1)

    ###################### Only display the maximum similarity one, but what if we want top 5?


    ####################### This one below will consider top K items!

    TOP_K = constants.TOP_K

    if len(similarities)<TOP_K:
        TOP_K = len(similarities)
        
    top_5_indices = torch.topk(similarities, TOP_K, largest=True).indices         ############# Change TOP_K!

    return top_5_indices


########################################### Get indices of the top related items for each question ###########################################


def get_top_k_indices(column_for_question_dict):

    top_k_indices = []

    cols_already_considered = set()

    new_dict_indices_for_each_question = dict()
    
    for question, value in column_for_question_dict.items():
    
        ##################### Load embeddings of the specific column!
    
        col_name = value[0]

        if col_name not in loaded_embeddings:
            continue
            
        most_similar_col_embeddings = loaded_embeddings[col_name]
    
        #################### Judging similarity with the question and returning top 10, indices!

        if "items to look for" not in question:
            
            top_k_indices = get_top_K_items_for_question(most_similar_col_embeddings, question)
            top_k_indices = [i.item() for i in top_k_indices]
            
        else:
            top_k_indices = previous_indices
        
        new_dict_indices_for_each_question[question] = top_k_indices

        previous_indices = new_dict_indices_for_each_question[question]
        
    
    return top_k_indices, cols_already_considered, new_dict_indices_for_each_question
    


########################################### Only indices will not work! We need the specifc values of the indices for each column! ###########################################



def impute_value(value):

    if value and "nan" in value:
        value = value.replace("nan","Not present")
    return value



def get_values_from_df(question, top_k_indices, column_for_question_dict):
    
    top_k_indices = [int(i) for i in top_k_indices if i is not None]

    top_k_indices = [idx for idx in top_k_indices if idx < len(df)]
    
    filtered_df = df.iloc[top_k_indices]

    cols_to_consider = [column_for_question_dict[question][0]]

    # all_columns = ["Item ID_2","Vendor Item Description"]
    all_columns = []
    
    for k, v in column_for_question_dict.items():
        all_columns.append(v[0])

    all_columns = list(set(all_columns))

    filtered_df = filtered_df[filtered_df.columns.intersection(all_columns)]

    filtered_df.fillna("Not Present", inplace=True)

    cols = list(filtered_df.columns)

    for each in cols:
        filtered_df[each] = filtered_df.apply(lambda x : impute_value(x[each]), axis=1)
    
    return filtered_df



#################################################### Method to trim message legnth for GPT if needed!

def trim_message_for_gpt(messages):

    system_prompt_length = 0
    
    system_prompt_length = len(messages[0]["content"])

    user_prompt_length = len(messages[1]["content"])

    if system_prompt_length + user_prompt_length>128000:
        ######### Need to curb message!
        print_flush(f"\n\n GPT context length exceeded! \n\n")
        user_prompt = messages[1]["content"]
        user_prompt = user_prompt[:127990 - system_prompt_length]
        messages[1]["content"] = user_prompt

    return messages


#################################################### Time to get the final answer IN STRING ! ####################################################

def get_final_answer(dict_of_nested_questions_with_ans, user_query):

    for k, v in dict_of_nested_questions_with_ans.items():

        if v is not None and isinstance(v, pd.DataFrame):
            v = v.fillna("Not Present")
    
            # v = v[~((v['Item ID_2'] == "Not Present") & (v['Vendor Item Description'] == "Not Present"))]

            v = v.drop_duplicates(subset='Vendor Item Description', keep='first')
    
            dict_of_nested_questions_with_ans[k] = v.to_dict('records')

    print_flush("\n\n*********************************************\n")

    messages = []

    messages.append({"role":"system","content":f''' You are a helpful item chatbot. You will get a JSON and a user's question as inputs. Your job is to check the data present in the values of the JSON, analyze it and then construct a relevant and professional answer from it based on the user's question. Please note that your answer must only show results relevant to the user's question. Always try to help. Order your results based on their relevance to the user's query. Do not mention same accident twice. You will get only ONE question so merge all answers. Make the final answer look like the answer of ONE question. Provide as much accident information possible(DR Number) based on the user's question. '''  })
    
    messages.append({"role":"user","content":f''' Your input dict is : {dict_of_nested_questions_with_ans} and the user's question is : {user_query} '''})

    cost, output = 0.0, ""

    ##### Often messages exceeds 128000 which is maximum context length! Need to do something here!

    reduced_messages = trim_message_for_gpt(messages)

    start_gpt = time.time()
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",     #Gpt-3.5-turbo-16k, gpt-4o
            messages=reduced_messages
        )

        if response:
    
            cost = response.__dict__["_previous"]["usage"]["prompt_tokens"]*0.15/1000000 + response.__dict__["_previous"]["usage"]["completion_tokens"]*0.60/1000000
    
            output = response.choices[0].message['content'].replace("`", "").replace("json", "")

    except Exception as e:

        print_flush(f"\n\n GPT has problem {str(e)} with method get_final_answer .\n\n")
        return "", 0.0

    end_gpt = time.time()

    print_flush(f"\n\n Inside get_final_answer, GPT took time : {end_gpt - start_gpt} \n\n")
    
    return output, cost



def get_final_answer_groq(dict_of_nested_questions_with_ans, user_query):

    for k, v in dict_of_nested_questions_with_ans.items():

        v = v.fillna("Not Present")

        dict_of_nested_questions_with_ans[k] = v.to_dict('records')

    print_flush("\n\n*********************************************\n")
        
    messages = []
    
    answer = ""

    for k, v in dict_of_nested_questions_with_ans.items():
        print_flush(f"\n\n Key is {k} \n\n")
        print_flush(f"\n\n value is {v} \n\n")

    messages.append({"role":"system","content":f''' You are a helpful chatbot. You will get a JSON and a user's question as inputs. Your job is to check the data present in the values of the JSON, analyze it and then construct a relevant and professional answer from it based on the user's question. Please note that your answer must only show results relevant to the user's question. Always try to help. Order your results based on their relevance to the user's query. Do not mention same accident twice. You will get only ONE question so merge all answers. Make the final answer look like the answer of ONE question. Provide as much accident information possible(DR Number mandatory) based on the user's question. '''  })
    
    messages.append({"role":"user","content":f''' Your input dict is : {dict_of_nested_questions_with_ans} and the user's question is : {user_query} '''})

    start_groq = time.time()

    # try:
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        top_p=1,
        stream=True,
        stop=None,
    )
     
    for chunk in completion:
    
        content = chunk.choices[0].delta.content or ""
        
        answer += content

    end_groq = time.time()

    print_flush(f"\n\n Inside get_final_answer_groq, Groq took time : {end_groq - start_groq} \n\n")

    if "</think>" in answer:
        answer = answer.split("</think>")[-1].strip()

    end_groq = time.time()

    print_flush(f"\n\n Inside get_final_answer_groq, Groq took time : {end_groq - start_groq} \n\n")
    
    return answer, 0.0



################################################## This one happens rarely : only when Groq returns all item IDs that are invalid, no record actually exists against them.

def double_check_item_ID(answer):

    corrected_answer = answer.copy()
    
    if answer and isinstance(answer, dict):
        all_item_IDs = list(answer.keys())

        for each in all_item_IDs:
            if df is not None:
                if df[df["DR Number"]==each].shape[0] == 0 :  ##### Retuened DR Number is invalid!
                    del corrected_answer[each]

    return corrected_answer



####################################################  Final Answer in JSON (Key : Item ID, Value : Other Item Info)! ####################################################

def get_final_answer_in_JSON_groq(final_answer):

    messages = []

    template = [{"Input":"As per your request, Accident with DR Number 190319651 occurred in April 2019. It happened near Jefferson Boulevard and the victim was a male.","Output":{"190319651":{"Accident Date":"April 2019","Relevant Attributes":["Male victim, occurred near Jefferson Boulevsrd"]},"OVERALL_FEEDBACK":"This accident answers your question. Please let me know if you need anything else. "}}]
    
    messages.append({"role":"system","content":f''' You will get a text as input. The text contains information about some accidents that occurred on different dates. Your job is to construct a JSON from that text such that the keys of the JSON are the DR Number present in the input and the values are all the other information about that accident present in the answer except the DR Number. Include all DR Number you find in input. Also include all the other, non-accident information present in input text as key `OVERALL_FEEDBACK` in the output JSON. You will always respond in a JSON format. See example here : {template} '''  })

    messages.append({"role":"user","content":f''' Your input is : {final_answer} '''})
    
    answer = ""

    start_groq = time.time()
   
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=messages,
        top_p=1,
        stream=True,
        stop=None,
    )
     
    for chunk in completion:
    
        content = chunk.choices[0].delta.content or ""
        
        answer += content

    end_groq = time.time()

    print_flush(f"\n\n Inside get_final_answer_in_JSON_groq, Groq took time : {end_groq - start_groq} \n\n")
                    
    if "</think>" in answer:
        answer = answer.split("</think>")[-1].strip()

    answer = answer.replace("```","").replace(".json","").replace("json","")

    with open("final_answer_json.txt","w") as file:
        file.write(answer)
    
    answer = json.loads(answer)

    ################### need to double check if the item IDs returned are correct and actually exist in the data!

    corrected_answer = double_check_item_ID(answer)

    if corrected_answer == dict():
        corrected_answer = ""
        
    return corrected_answer, 0.00



def get_final_answer_in_JSON(final_answer):

    messages = []

    template = [{"Input":"As per your request, Accident with DR Number 190319651 occurred in April 2019. It happened near Jefferson Boulevard and the victim was a male.","Output":{"190319651":{"Accident Date":"April 2019","Relevant Attributes":["Male victim, occurred near Jefferson Boulevsrd"]},"OVERALL_FEEDBACK":"This accident answers your question. Please let me know if you need anything else. "}}]
    
    messages.append({"role":"system","content":f''' You will get a text as input. The text contains information about some accidents that occurred on different dates. Your job is to construct a JSON from that text such that the keys of the JSON are the DR Number present in the input and the values are all the other information about that accident present in the answer except the DR Number. Include all DR Number you find in input. Also include all the other, non-accident information present in input text as key `OVERALL_FEEDBACK` in the output JSON. You will always respond in a JSON format. See example here : {template} '''  })

    messages.append({"role":"user","content":f''' Your input is : {final_answer} '''})

    reduced_messages = trim_message_for_gpt(messages)

    try:
        response = openai.ChatCompletion.create(
            model=constants.GPT_MODEL_NAME_FOR_JSON,   #gpt-3.5-turbo-16k
            messages=reduced_messages
        )
    
        cost, output = 0.0, ""
    
        if response:
    
            cost = response.__dict__["_previous"]["usage"]["prompt_tokens"]*0.15/1000000 + response.__dict__["_previous"]["usage"]["completion_tokens"]*0.60/1000000
    
            output = response.choices[0].message['content'].replace("`", "").replace("json", "")

    except Exception as e:

        pass

    return output, cost



########################################### So how much is the total GPT cost for the whole thing? ##########################################


def calculate_total_cost(cost_first, column_for_question_dict, last_cost):

    total_cost = cost_first + last_cost

    for k, v in column_for_question_dict.items():
        cost = v[-1]
        total_cost+=cost

    return total_cost


########################################### Updating the cost bakc to file! ##########################################


def update_cost_to_file(total_cost):


    previous_cost = 0.0

    with open(constants.OPENAI_COST_FILE,"r") as file:
        previous_cost = float(file.read().split("=")[-1])


    with open(constants.OPENAI_COST_FILE,"w") as file:
        file.write("CURRENT_COST="+str(previous_cost + total_cost))


########################################### This method fetches rows from the DataFrame from the top)indices returned for a specific question! ##########################################



def get_ans_for_each_nested_question(new_dict_of_indices_for_each_question, column_for_question_dict):
    

    nested_ques_ans_dict = dict()

    final_df = pd.DataFrame()

    for k, v in new_dict_of_indices_for_each_question.items():  #

        indices = v

        ########################################### Lets first fetch the values from the dataframe! 

        necessary_subdf = get_values_from_df(k, v, column_for_question_dict)

        ########################################## Get answer for only this question! 

        nested_ques_ans_dict[k] = necessary_subdf

    return nested_ques_ans_dict


def get_item_name_in_question(user_query):


    item_name_template = [{"Input":"Show me accidents near Van Nuys", "Output":"Area Name"},{"Input":"Items sold in USA","Output":"NA"}]
    
    messages = []
    
    messages.append({"role":"system","content":f''' You will return the item present in input. Only return the item like for input {item_name_template[0]["Input"]}, your output is : {item_name_template[0]["Output"]}. If no specific item present in input, just say `NA` like for input {item_name_template[1]["Input"]}, your output is {item_name_template[1]["Output"]}. Do not give verbose answers. '''  })

    messages.append({"role":"user","content":f''' Your input is {user_query} '''})

    try:
        
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=1.0
        )
    
        cost, output = 0.0, ""
            
        if response:
    
            cost = response.__dict__["_previous"]["usage"]["prompt_tokens"]*0.15/1000000 + response.__dict__["_previous"]["usage"]["completion_tokens"]*0.60/1000000
    
            output = response.choices[0].message['content'].replace("`", "").replace("json", "").replace("'","")           
        
    except Exception as e:
        return str(e), 0.0

    
    return output, cost



 
########################################### Web setvice method ###########################################


@app.route('/product_bot', methods=['POST'])
@cross_origin(supports_credentials=True)

def get_users_question():

    user_query = request.form['query']   ############## Get the users's question suing GET/POST

    start = time.time()

    ################################################ Fetch the item name first! ################################################

    rate_limit_error_dict = {"Response":"","OVERALL_FEEDBACK":"OpenAI key Rate Limit Error exceeded! Please increase the quota.."}
        
    item_name, item_cost = get_item_name_in_question(user_query)

    # print_flush("The entity/column name returned is : ",item_name," and the price is : ", item_cost,"\n\n")
    
    ################################################ Breaking the complex question into simple questions ################################################
    
    try:
        answer, cost_first = get_steps_to_answer_question_groq(user_query)
        if answer == "":
            answer, cost_first = get_steps_to_answer_question(user_query)
            
    except Exception as e:
        print_flush(f"\n\n Groq had error with get_steps_to_answer_question : {str(e)} \n\n")
        answer, cost_first = get_steps_to_answer_question(user_query)
 
    try:
        all_questions = ast.literal_eval(answer)
    except:
        all_questions = answer.split(",")

    ############### Get most related column for each step
    
    column_for_question_dict = dict()

    start_column = time.time()
    
    for each in all_questions:

        try:
            column_to_consider, cost = get_me_the_column_groq(each)
            if column_to_consider == "":    ##### Groq failed! Need GPT..
                column_to_consider, cost = get_me_the_column(each)
                
        except Exception as e:
            column_to_consider, cost = get_me_the_column(each)
        
        if column_to_consider is not None and column_to_consider!="None":
            column_for_question_dict[each] = (column_to_consider, 0.00)

    end_column = time.time()

    print_flush(f"\n\n Time taken to get all coliumns is {end_column - start_column} \n\n")
    
    ################################################ COlumn Vendor Item Description might not be returned by GPT for get_me_the_column so We are explicitly adding column Vendor Item Description based on the itme present in question!

    item_desc_col_present = False                            ################################################ A simple boolean value to check if column Vendor Item Descriptin is present in the dict!         

    for k, v in column_for_question_dict.items():
        
        if v[0]=="Vendor Item Description":
            
            item_desc_col_present = True
            
            break
    
    if not item_desc_col_present and item_name!="NA":                                 ################################################ If not, we are adding it here!
        
        column_for_question_dict[f"Find items like {item_name} "] = ("Vendor Item Description",cost)
    
    elif not item_desc_col_present and item_name=="NA":

        column_for_question_dict["items to look for "] = ("Vendor Item Description",0.000000)
    
    ################################################ 

    fields = ["Vendor Item Description"]

    for k, v in column_for_question_dict.items():

        if v and len(v)>0 and v[0] is not None and v[0] not in fields:
            fields.append(v[0])
    
    print_flush(f"\n\n ************************************************ \n\n")
    
    ################# Get top related indices for each small question

    top_k_indices, cols_already_considered, new_dict_of_indices_for_each_question = get_top_k_indices(column_for_question_dict)
    
    ################# Time to fetch values from the DataFrame for the other questions based on the top k indices!

    ################# Create a dict, key will be each of the nested question, value will be their answer, then we will think about the final answer!
    
    dict_of_nested_questions_with_ans = get_ans_for_each_nested_question(new_dict_of_indices_for_each_question, column_for_question_dict)

    ################# Top 15K indices will have same item ID multiple times because of the 1-to-many relationship of the table, lets create a simple dict key is item ID  

    ################# And here comes the FINAL ANSWER!

    try:
        # r.append("")
        final_answer, last_cost = get_final_answer_groq(dict_of_nested_questions_with_ans, user_query)
        print_flush(f'''\n\n get_final_answer_groq returns \n\n {final_answer} of length {len(final_answer)} \n\n ''')
        
    except Exception as e:
        
        print_flush(f"\n\n Groq had an error with get_final_answer_groq : {str(e)} \n\n")
        final_answer, last_cost = get_final_answer(dict_of_nested_questions_with_ans, user_query)

    ################# We also need the IDs of items from the final answer!

    # try:
    #     final_answer_in_JSON, cost_json = get_final_answer_in_JSON_groq(final_answer)
    #     print_flush(f"\n\n get_final_answer_in_JSON_groq returns {final_answer_in_JSON} \n\n")

    #     ################ adding key overall feedback to the answer provided by Groq!
    #     if final_answer_in_JSON and isinstance(final_answer_in_JSON, dict):
    #         final_answer_in_JSON["OVERALL_FEEDBACK"] = "Based on your question, these are the items that meet your requirements. Please note, for exact information kindly verify with the manufacturer.  "
    #     else:
    #         final_answer_in_JSON, cost_json = get_final_answer_in_JSON(final_answer)
        
    # except Exception as e:
    #     print_flush(f"\n\n Groq had an error with get_final_answer_in_JSON_groq : {str(e)} \n\n")
    #     final_answer_in_JSON, cost_json = get_final_answer_in_JSON(final_answer)
    
    ################# Time to calculate the cost!

    total_cost = calculate_total_cost(cost_first, column_for_question_dict, last_cost + item_cost)

    print_flush("The cost for the whole request: ", total_cost, "\n**************\n")

    ################# Lets update the cost!

    update_cost_to_file(total_cost)
    
    ################################## Please create a plain simple dict, key is Item ID, value is the entry! ##################################

    end = time.time()

    print_flush("Request took total time : ",end - start,"\n\n")

    with open("final_items.txt", "w") as file:
        json.dump(final_answer, file, indent=4)

    ############################### Keys should be Item ID, Values will be rest!
    
    # if final_answer_in_JSON and isinstance(final_answer_in_JSON, dict) and "OVERALL_FEEDBACK" in final_answer_in_JSON:
        
    #     final_answer_in_JSON["OVERALL_FEEDBACK"] = final_answer_in_JSON["OVERALL_FEEDBACK"] + "\n YOUR VIEW HAS BEEN REFRESHED WITH THE ITEMS\n" 

    #     for k in final_answer_in_JSON:
    #         print_flush(f"\n\n Key is {k} \n\n")
    #         print_flush(f"\n\n Value is {final_answer_in_JSON[k]} \n\n")
            
    #     return jsonify({"Response":final_answer_in_JSON})

    # try:
    #     xc = demjson3.decode(final_answer_in_JSON)
    
    #     if xc and isinstance(xc, dict) and "OVERALL_FEEDBACK" in xc:
    #         xc["OVERALL_FEEDBACK"] = xc["OVERALL_FEEDBACK"] + "\n YOUR VIEW HAS BEEN REFRESHED WITH THE ITEMS \n. \n" 
    #         return jsonify({"Response":xc})
            
    # except Exception as e:
    #     pass
        
    return jsonify({"Response":final_answer.split("\n")})


################ Add the app.run statement
app.run(host="0.0.0.0", port=5006)