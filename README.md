# Final Project ChatBot

![](images/graph1/../chess.jpeg)


### Introduction
This project is about making a Chatbot using Machine learning and neural networks and other skills. The point of this project was have something similiar to Siri but text version, and also provide the chatbot with functions and commands that make my experience as a user and data analyst easier.


# Implementation of a Chatbot  . 
Simple chatbot implementation with PyTorch. 

- The implementation should be easy to follow.
- Customization for your own use case is super easy. Just modify `intents.json` and `chat.py` with possible patterns and responses and re-run the training.

The approach is inspired by this article and ported to PyTorch: [https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077](https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077).


## Installation

### Create an environment
Its important to make a new environment to avoid conflicts and errors.
Whatever you prefer (e.g. `conda` or `venv`)

### Install PyTorch and dependencies

For Installation of PyTorch see [official website](https://pytorch.org/).

You also need `nltk`:
 ```console
pip install nltk
 ```

If you get an error you also need to install `nltk.tokenize.punkt`:

## Usage
Run
```console
python train.py
```
This will create `data.pth` file. thats need for the program. After run
```console
python app.py
```
## Customize
Have a look at [intents.json](intents.json). You can customize. Just fill intents.json, define a new `tag`, possible `patterns`, and possible `responses` for the chat bot. You have to re-run the training whenever this file is modified.
```console
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "Hi",
        "Hey",
        "How are you",
        "Is anyone there?",
        "Hello",
        "Good day"
      ],
      "responses": [
        "Hey :-)",
        "Hello, thanks for visiting",
        "Hi there, what can I do for you?",
        "Hi there, how can I help?"
      ]
    },
    ...
  ]
}
```