# FreeAPI-Chatbot
An AI-powered chatbot that analyzes CSV files and answers questions using free LLM APIs (Groq, Cerebras, OpenRouter). A helpful study tool for students, learners, and data beginners. This will be especially helpful for students/ data beginners in LLM. The **Postman Collection for the Input and Output** has been added to repo. 

### This API uses all Free LLM APIs to build an intelligent chatbot that answers questions from a CSV file. 

#### Please create API keys first : 

For Groq : [https://console.groq.com/keys](url)

For Cerebras : [https://cloud.cerebras.ai/platform/org_2f52r2n624x68m9pjtfnpkry/playground?utm_source=homepage](url)

For OpenRouter : [https://openrouter.ai/sign-in?redirect_url=https%3A%2F%2Fopenrouter.ai%2Fsettings%2Fkeys](url)

For OpenAI/GPT : [https://platform.openai.com/docs/quickstart/create-and-export-an-api-key](url)


#### Note that OpenAI key is not free. 

##### Data file originally downloaded from Kaggle : [https://www.kaggle.com/datasets/johnsonmoon/traffic-collision-data-from-2010-to-present
](url)

##### A subset of the data has been uploaded here, please use the full dataset for better results.

#### Please use python >3.9 to run this. Follow the steps below : 

1. Create a conda environment : ```conda create --name=myenv python=3.10```/```python -m venv myenv python=3.10```
2. Activate the virtual environemnt: ```conda activate myenv```
3. Install all necessary dependencies : ```pip install -r requirements.txt --no-cache-dir```
4. Run the file ```bot_product_SQL_embeddings.py``` first. This will locally store the embeddings and the parquet files.
5. Once, the parquet files are saved run the Flask application file ```groq_gpt_data_bot.py```

##### Groq has a free tier, check it out here : [https://console.groq.com/docs/rate-limits](url)

##### Get all available Groq models here : [https://console.groq.com/docs/models](url)

##### Basic Groq API syntax (with OpenAI model) : 

```python
from openai import OpenAI
import os
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

response = client.responses.create(
    input="Explain the importance of fast language models",
    model="openai/gpt-oss-20b",
)
print(response.output_text)
```
##### Groq API syntax (with Llama/other model) : 


##### Cerebras API create key and use Playground here : [https://cloud.cerebras.ai/platform/org_repdydkp566nht4ymvj4wt4d/playground](url)

###### Python code snippet : 

```python
import os
from cerebras.cloud.sdk import Cerebras

client = Cerebras(
    # This is the default and can be omitted
    api_key=os.environ.get("CEREBRAS_API_KEY")
)

stream = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": ""
        }
    ],
    model="llama-3.3-70b",
    stream=True,
    max_completion_tokens=2048,
    temperature=0.2,
    top_p=1
)

for chunk in stream:
  print(chunk.choices[0].delta.content or "", end="")
```

##### OpenRouter code snippet : 

```python
import requests
import json

response = requests.get(
  url="https://openrouter.ai/api/v1/key",
  headers={
    "Authorization": f"Bearer <OPENROUTER_API_KEY>"
  }
)

print(json.dumps(response.json(), indent=2))
```

###### Check out OpenRouter free limits here : [https://openrouter.ai/docs/api-reference/overview](url)


## Examples with Question and Bot's answer : 

#### User's question : How many accidents took place in 2019? What were the areas? Who were the victoms?

#### Bot's answer : 

{
    "Response": [
        "Based on the provided JSON data, I've analyzed the information and constructed a relevant answer to the user's question.",
        "",
        "**Accidents in 2019:**",
        "There were 3 accidents that took place in 2019.",
        "",
        "**Areas:**",
        "The areas where these accidents occurred are:",
        "",
        "1. Central (on 09/01/2019, 07/11/2019, 06/23/2019, 09/02/2019, 09/04/2019, 11/20/2019, 07/12/2019, 07/09/2019, 07/15/2019, 08/30/2019, 08/31/2019, 04/27/2019, 07/08/2019, 07/10/2019, 07/04/2019, 06/29/2019, 07/17/2019, 07/07/2019)",
        "2. Hollywood (on 03/19/2019)",
        "",
        "**Victims:**",
        "Unfortunately, the provided JSON data does not contain information about the victims of these accidents.",
        "",
        "**Additional Information:**",
        "Although not directly related to the question, it's worth noting that there was 1 accident on 03/19/2019 at Hollywood and 2 accidents on other dates are present in 'Identify accidents that took place in 2019 year' section ",
        "",
        "Here is the DR Number and Date for accidents which took place in 2019 as per 'Identify accidents that took place in 2019 year': ",
        "- No DR Number is available. However, Date for one accident is 03/19/2019.",
        "",
        "If you need more information or clarification, please let me know."
    ]
}

#### Screenshot : 

<img width="1046" height="899" alt="freeapi_bot" src="https://github.com/user-attachments/assets/77231fd7-31a5-47ac-a41d-9fd461781098" />
