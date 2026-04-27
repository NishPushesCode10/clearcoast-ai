import streamlit as st
from openai import AzureOpenAI

def get_azure_openai_client():
    return AzureOpenAI(
        azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
        api_key=st.secrets["AZURE_OPENAI_KEY"],
        api_version="2024-02-01"
    )

def summarize_coastal_image(analysis_text):
    client = get_azure_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a coastal monitoring expert."},
            {"role": "user", "content": f"Summarize the coastal analysis: {analysis_text}"}
        ]
    )
    return response.choices[0].message.content
