The Ollama Python Library
=========================

Ollama provides a Python Library that communicates with the Ollama application via the Ollama HTTP API on your local system.  It also provides a collection of helper applications that facilitate performance of the most common language processing tasks.  Before invoking the Ollama library you must first install it into your local Python environment.  You can do so via the command line or the Python terminal of your Python IDE using the following command:

```
pip install ollama
```

Note:  if working in a notebook, you can run this command directly in a code cell using the "!" special character like this:

```
!pip install ollama
```

Configuring the API Connection
-------------------------------------
The Ollama Python library uses the API function of the Ollama HTTP Server to communicate with the Ollama models.  You need to set the URL for the API before you can communicate with it.  The defult URL for the Ollama HTTP Server is "http://localhost:11434". (Localhost = your local computer, 11434 = the network port that Ollama is assigned to by default). Note that in a team environment you could run Ollama on a server and access it by IP address and port 11434).

To configure the client via Python, we will create a python object that holds a client reference:

```py
client = Client(host='http://localhost:11434')
```

We also need to configure a "model" object that tells the API which model to use for the interaction:

```py
model = 'llama3.2'
```

Prompt and Response
-------------------

Now that we have defined all necessary parameters, we can construct a prompt to send to the model:

```py
# define prompt
prompt = "What is Harry Potter About?"

# Get response
response = client.chat(model=model, messages=[{'role': 'user', 'content': prompt}])

# print to screen
print(response['message']['content'])

```

A Simple Chatbot
----------------

We can functionalize the code above and with a little extra code build a very simple Chatbot application.

```py
from ollama import Client

client = Client(host='http://localhost:11434')
model = 'llama3.2'


def generate_response(prompt):
    response = client.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    
    response = generate_response(user_input)
    print("Bot:", response)
```





