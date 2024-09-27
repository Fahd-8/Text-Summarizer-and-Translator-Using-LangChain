from flask import Flask, render_template, request, jsonify
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

app = Flask(__name__)

# Summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summarizer_llm = HuggingFacePipeline(pipeline=summarizer)

# Simplification model (using the same summarizer for simplicity, adjust as needed)
simplifier = pipeline("text2text-generation", model="facebook/bart-large-cnn")
simplifier_llm = HuggingFacePipeline(pipeline=simplifier)

# Create Prompt Templates
summarization_template = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text:\n\n{text}\n\nSummary:"
)

simplification_template = PromptTemplate(
    input_variables=["text"],
    template="Translate the following text into easy English:\n\n{text}\n\nSimple English:"
)

# Summarization chain
summarization_chain = LLMChain(
    llm=summarizer_llm,
    prompt=summarization_template,
    output_key="summary"
)

# Simplification chain
simplification_chain = LLMChain(
    llm=simplifier_llm,
    prompt=simplification_template,
    output_key="simple_text"
)

# Sequential chain
sequential_chain = SequentialChain(
    chains=[summarization_chain, simplification_chain],
    input_variables=["text"],
    output_variables=["summary", "simple_text"]
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    user_input = request.json['text']
    result = sequential_chain({"text": user_input})
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
