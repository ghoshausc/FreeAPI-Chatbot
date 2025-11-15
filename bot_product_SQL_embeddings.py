import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

import re, csv, json, time, ast, io, sys
import torch, time, ast, io, collections, constants

import pyarrow.parquet as pq
from sqlalchemy import create_engine
import psycopg2, gc
import pyarrow as pa

########################################### Better ways of printing! :)  ###########################################

def print_flush(*args, **kwargs):
    
    print(*args, **kwargs)
    
    sys.stdout.flush()


final_df = pd.read_csv("Traffic_Collision_Data_from_2010_to_Present.csv") 

important_columns = list(final_df.columns)

####### item worlflow status, selling size location

######################################## Ready to get embeddings of columns and stakc them in a single file! 

def get_embeddings_and_store_in_file(df):

    start = time.time()
    
    model = SentenceTransformer(constants.EMBEDDING_MODEL_NAME)  ######### for more accurate embeddings, use thenlper-gte/large. this takes more time

    # model = SentenceTransformer('thenlper-gte/large')
    
    data = []

    for col in important_columns:
        
        print_flush(f"Processing column: {col}")

        try:
            column_data = df[col].fillna("Missing").astype(str).tolist()
            
            embeddings = model.encode(column_data, batch_size=32, show_progress_bar=True, device='cpu')  
            
            torch.cuda.empty_cache()
            
            for idx, emb in enumerate(embeddings):
                data.append({
                    'column': col,
                    'index': idx,
                    'embedding': emb.tolist()  # Convert embeddings to list to save as JSON
                })

        except Exception as e:
            print_flush("The error is : ",str(e),"\n\n")
            
    df_embeddings = pd.DataFrame(data)
    
    df_embeddings.to_parquet('column_embeddings.parquet', compression='gzip')
    
    end = time.time()
    
    print_flush("Total time taken to get embeddings of all columns:", end - start)
    


########################## Some columns in joined dataframe have either empty descriptions or have words like "demo" or "test". We DO NOT NEED THESE items!

########################### merging columns finally!

start_begin = time.time()

final_df = final_df.astype(str)


############################ Now the joined dataframe will have 1M/2M rows so now we will merge the attribute classes, codes and values together.


########################### çreating dicts nowm key is Item ID_2 and value is attribute classes merged as a string


############################# Discard rows having False under column "Vendor Item Description"!

final_df = final_df.head(10000)   ########### Checkinf if everything works well with a sample of the data!

print_flush(f"\n\n Final df has columns : {final_df.shape} \n\n")

final_df.to_parquet("product_bot_data.parquet",index=False)

get_embeddings_and_store_in_file(final_df)

end_begin = time.time()

print_flush("Whole process took time : ",end_begin - start_begin,"\n\n")

############################################ Emnbeddings stored locally ############################################

print_flush("Finished getting data embeddings for Product Bot : ready to run the Flask application now!\n")