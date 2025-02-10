Why Ollama?
===========

There are many ways to work with Large Langauge Models (LLMs) on your local system, and many LLMs to choose from.  Hugging Face, a digital community, “on a mission to democratize good machine learning,” currently publishes nearly 300,000 open-source language models in a variety of languages and custom suited to specific purposes and domains.  In most cases, working with these models requires advanced technical skills and knowledge.  

Ollama came into being to allow users without this deep knowledge and skillset to begin working directly with LLMs.  As is always the case, however, simplicity is inversely related to flexibility.   Ollama is great place to start your journey into LLMs, but it limits the models available to you and what you can do with them. 

After completing this workshop, those with a need to expand their model choices and the flexibility with which they can interact with them should consider the Python OpenLLM library, which provides a common framework for interacting with any Hugging Face model.  

Most technically advanced users will ultimately find themselves working directly with their LLM(s) of choice using their programming environment of choice in order to streamline code efficiency and processing workflows.


Why run an LLM locally?
-----------------------

There are two primary reasons for running a LLM locally:  Model selection and privacy.  Large Language Models are, as the name suggests, large.  Cloud based systems are trained on billions of texts from hundreds of thousands (of usually unknown) sources and domains.  This breadth of training results in models that are deep and rich but also extremely generalized, often lacking the specificity that is required for focused work in a particular domain.  Fine tuning of these models and processes, such as Retrieval Augmented Generation (RAG), can be used to overcome this generality. However, researchers often find that the domain specificity required for particular research questions requires a less general and more purpose-built language model.

Additionally, interacting with a cloud-based model, either through prompting or tuning, requires transferring your data across the network to the cloud-based system.  And, in some cases, the data you transfer is ingested into the model itself.  This transfer of data often violates a range of data privacy protocols and standards (HIPAA, IRB, FERPA, etc.) and raises significant intellectual property rights concerns.   Working with LLMs locally allows you to maintain proper data control and hygiene.


Workshop Environment Setup
--------------------------

### Download and Install Ollama

Visit the link below to download and install Ollama for your operating system.  Note that while the installation process is slightly different for each operating system it is fairly straightforward.  When you've completed the installation, Ollama will run in the background on your computer.  We will interact with Ollama via a command-line interface and Python code, rather than a Graphical User Interface (GUI).

[https://ollama.com/download](https://ollama.com/download)

### Optional Python Coding Environment

Much of the work that we will perform in this workshop will be conducted via a Command Line Interface application already installed on your computer as part of the operating system installation.  Additional interaction with the LLM will be performed using Python.  Those who wish to code-along during the Python portion of the workshop should have a Python coding environment such as VS Code, Jupyter Notebooks, or via the Command Line available on their system.


