import uvicorn
from fastapi import FastAPI
from typing import Dict
from pydantic import BaseModel
import pandas as pd 
import joblib
import pickle
import json

app = FastAPI()
test_model_data = pd.read_csv('models/sample_pred.csv')
model = joblib.load('models/prediction.pkl')

class Pred(BaseModel):
    property_type: str
    room_type: str
    accomodates: int
    bathrooms: float
    clean_fee: bool
    city: str
    latitude: float
    longitude: float
    review_scores_rating: float
    zipcode: int
    bedrooms: float
    beds: float
    Dryer: bool
    Parking: bool
    Description_Len: int


class Sentence(BaseModel):
    text: str


class User(BaseModel):
    name: str
    age: int
@app.get('/')
def home():
    return {'hello': "world"}


@app.get('/users/{user_id}')
def get_user_details(user_id: int):
    return {'user_id': user_id}


@app.post('/users')
def save_user(user: User):
    return {'name': user.name,
            'age': user.age}

#make text lowercase
@app.post('/lowercase')
def lower_case(json_data: Dict):
    text = json_data.get('text')
    return {'text': text.lower()}


@app.post('/predict')
def predict(pred: Pred):
    import pandas as pd
    import joblib
    import pickle
    test_model_data = pd.read_csv('models/sample_pred.csv')
    model = joblib.load('models/prediction.pkl')
    response  = round(model.predict(test_model_data)[0],2)
    price_list = ['Prices']
    response_list = [f'{round(response,2)}']
    response_dict = dict(zip(price_list, response_list))
    response_json = json.dumps(response_dict)
    return pred.beds

#make text uppercase
@app.post('/uppercase')
def upper_case(sentence: Sentence):
    return {'text': sentence.text.upper()}

if __name__ == '__main__':
    uvicorn.run(app)
