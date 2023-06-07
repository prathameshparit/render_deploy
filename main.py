from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List
from test import Recommender

app = FastAPI()

# Define the input model
class ArticleInput(BaseModel):
    input_data: str

    @validator('input_data')
    def input_data_length(cls, value):
        if len(value) < 5:
            raise ValueError("Input data should have a minimum length of 5 characters")
        return value

# Define the output model
class ArticleOutput(BaseModel):
    recommended_articles: List[str]

# Define the API endpoint
@app.post("/recommend", response_model=ArticleOutput)
async def recommend(articles: ArticleInput) -> ArticleOutput:
    try:
        input_data = articles.input_data
        recommended_articles = Recommender(input_data)
        return ArticleOutput(recommended_articles=recommended_articles)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))