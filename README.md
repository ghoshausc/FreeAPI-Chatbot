# FreeAPI-Chatbot
An AI-powered chatbot that analyzes CSV files and answers questions using free LLM APIs (Groq, Cerebras, OpenRouter). A helpful study tool for students, learners, and data beginners. This will be especially helpful for students/ beginners in LLM.

### This API uses all Free LLM APIs to build an intelligent chatbot that answers questions from a CSV file. 

#### Please use python >3.9 to run this. Follow the steps below : 

1. Create a conda environment : ```conda create --name=myenv python=3.10```/```python -m venv myenv python=3.10```
2. Activate the virtual environemnt: ```conda activate myenv```
3. Install all necessary dependencies : ```pip install -r requirements.txt --no-cache-dir```
4. Run the file ```bot_product_SQL_embeddings.py``` first. This will locally store the embeddings and the parquet files.
5. Once, the parquet files are saved run the Flask application file ```groq_gpt_data_bot.py```

##### Groq has a free tier, check it out here : [https://console.groq.com/docs/rate-limits](url)

##### Basic Groq API syntax : 

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

#####OpenRouter code snippet : 

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


