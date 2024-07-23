# Ques-Ans-system-on-Sheets
 • Designed and built a question-answering system for unstructured spreadsheet data <br>
 • Developed a conversational agentic system using OpenAI-functions (GPT-3.5-Turbo) that reasons and acts based on either
 lookup-type questions or analytical questions<br>
 • Implemented RAG-based RetrievalQA chain for direct answers from unstructured XLSX sheets for lookup questions<br>
 • Preprocessed & extracted tables for analytical questions to avoid LLM hallucination problems, then utilized a computing agent
 that uses PythonAstREPLTool for accurate dataframe computations<br>
### Requirements
Execute the following to install all requirements:
```
pip install -r requirements.txt
```

### Solution Script
Execute the following command to start Ques-Ans system on the sheet
```
python sheetQA.py /path/to/file.xlsx openai_key
```
