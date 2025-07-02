import os
import json
import random

import nltk
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader    
# nltk.download('wordnet')

class ChatBotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatBotModel, self).__init__()

        self.fc1 = nn.Linear(input_size,128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3=nn.Linear(64, output_size)
        self.relu=nn.ReLU()
        self.dropout = nn.Dropout(0.5)# for regularization
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x= self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ChatbotAssistant:

    def __init__(self, intents_path, function_mappings=None):
        self.model = None
        self.intents_path = intents_path

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}

        self.function_mappings = function_mappings

        self.X = None
        self.y = None

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()

        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]

        return words

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        lemmatizer = nltk.WordNetLemmatizer()

        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)

            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))

                self.vocabulary = sorted(set(self.vocabulary))

    #conevrting the words in to numbers for the neural network
    def prepare_data(self):
        bags=[]
        indices=[]

        for doc in self.documents:
            words=doc[0]
            bag=self.bag_of_words(words)

            intent_index= self.intents.index(doc[1])

            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self,batch_size,lr,epochs):
        X_tensor=torch.tensor(self.X, dtype=torch.float32)
        y_tensor=torch.tensor(self.y, dtype=torch.long)

        dataset=TensorDataset(X_tensor, y_tensor)
        dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatBotModel(input_size=self.X.shape[1], output_size=len(self.intents))

        loss_fn=nn.CrossEntropyLoss()
        optimizer=optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss=0

            for batch_X,batch_y in dataloader:
                optimizer.zero_grad()
                outputs=self.model(batch_X)
                loss_value=loss_fn(outputs, batch_y)
                loss_value.backward()
                optimizer.step()

                total_loss += loss_value.item()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

    def save_model(self, model_path,dimensions_pathj):
        torch.save(self.model.state_dict(), model_path)

        with open(dimensions_pathj, 'w') as f:
            json.dump({
                'input_size': self.X.shape[1],
                'output_size': len(self.intents)
            }, f)

    def load_model(self, model_path, dimensions_pathj):
        with open(dimensions_pathj, 'r') as f:
            dimensions = json.load(f)

        self.model = ChatBotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path,weights_only=True))

    def process_message(self,input_message):
        words=self.tokenize_and_lemmatize(input_message)
        bag=self.bag_of_words(words)

        bag_tensor=torch.tensor(bag, dtype=torch.float32)
        self.model.eval()  # Set the model to evaluation mode   
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions).item()
        predicted_intent = self.intents[predicted_class_index]

        if self.function_mappings:
            if predicted_intent in self.function_mappings:
                return self.function_mappings[predicted_intent]()
            
        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])
        else:
            return "I'm sorry, I don't understand that. Can you please rephrase your question?"
        

if __name__ == "__main__":
    assistant=ChatbotAssistant(intents_path='intents.json')
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(batch_size=8,lr=0.001,epochs=100)
    

    while True:
        message=input('enter your message:')

        if message=='/quit':
            break

        print(assistant.process_message(message))