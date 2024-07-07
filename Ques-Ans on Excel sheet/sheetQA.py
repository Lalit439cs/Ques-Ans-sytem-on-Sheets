# imports
import os
import pandas as pd
import numpy as np
import re
import sys

from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
import openai
from langchain_openai import OpenAIEmbeddings


from langchain.document_loaders import UnstructuredExcelLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import initialize_agent


# preprocessing unstructured data of XLSX file to Dataframe or CSV
def check_row_title(row):
    # Check if the first entry is a string
    if isinstance(row.iloc[0], str):
        # Check if all other entries are NaN
        return row.iloc[1:].isna().all()
    return False

def all_nan(row):
    return row.isna().all()


def contains_only_digits_and_symbols(s):
    # Regular expression to match only digits and symbols
    pattern = re.compile(r'^[\d\W_]+$')
    if pd.isna(s):
        return False
    return bool(pattern.match(str(s)))

def no_digits_and_symbols_only(row):
    return not any(contains_only_digits_and_symbols(entry) for entry in row)

def add_df_info(temp_title,df,table_count):
    s=("*"*20)+"\n"
    s+=f"Here is brief info table-{table_count}, titled- {temp_title} :\n"
    s+=f"Name of {len(df.columns)} columns in this table- {df.columns} \n"
    s+=f"Name of {len(df)} rows in this table- {list(map(str,df.iloc[:, 0]))} \n"
    return s
    
def extract_tables(xls_file):
    sheet_df =[]
    # Read all sheets into a dictionary of DataFrames
    dfs = pd.read_excel(xls_file, sheet_name=None)
    dfs.keys()
    for sh in dfs:
        sheet_df.append(dfs[sh].drop(columns=dfs[sh].columns[:1], errors='ignore'))
        n =len(sheet_df[-1].columns)
        l=[i for i in range(n)]
        sheet_df[-1].columns =l
        # print(sheet_df[-1].head())
        # print("#"*20)

    prefix_text =""
    # global storing variables
    titles=[]
    headers=[]
    df_list=[] #deciding 

    # preprocessing code for tables extraction
    table_count=0
    for i in range(len(sheet_df)):
        sdf =sheet_df[i]
        prefix_text+=("#"*20)+"\n"
        prefix_text+=f"Information present in sheet number={i} :\n"
        for _ in list(map (str,sdf.iloc[0])):
            if "Notes:" in _:
                prefix_text+=_ +"\n"
                
    #     tables extraction
        temp_head=[]
        temp_title=""
        new_df =pd.DataFrame()
        
        prev_nan=True
        for j, row in sdf.iterrows():
            if (all_nan(row)):
                prev_nan=True
                continue
            elif (check_row_title(row)):

                if (len(new_df)!=0):
                    new_df.columns =temp_head
                    #clean
                    # Drop columns where all entries are NaN
                    new_df = new_df.dropna(axis=1, how='all')
                    df_list.append(new_df)
                    headers.append(new_df.columns)
                    titles.append(temp_title)
                    prefix_text+=add_df_info(temp_title,new_df,table_count)
                    table_count+=1
                new_df =pd.DataFrame()
                temp_title =row.iloc[0]
                #may be changes
                prev_nan=True
                continue
                    
            elif (no_digits_and_symbols_only(row)): #may be cases
                if (prev_nan):
                    if (len(new_df)!=0):
                        new_df.columns =temp_head
                        #clean
                        # Drop columns where all entries are NaN
                        new_df = new_df.dropna(axis=1, how='all')
                        df_list.append(new_df)
                        headers.append(new_df.columns)
                        titles.append(temp_title)
                        prefix_text+=add_df_info(temp_title,new_df,table_count)
                        new_df =pd.DataFrame()
                        table_count+=1
                    temp_head=list(map(str,row))
                    prev_nan=False
                    continue
                    
    #         new_df.append(row)
            new_df = pd.concat([new_df, pd.DataFrame([row])], ignore_index=True)
            prev_nan=False

    print(prefix_text, file=open("preprocessing_stats.txt", "w"))
    return df_list,prefix_text

def solution(xlsx_file,prefix_text, df_list):
    openai_llm =ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai.api_key )#"gpt-3.5-turbo",    
    loader=UnstructuredExcelLoader(xlsx_file)
    embeddings = OpenAIEmbeddings()

    index=VectorstoreIndexCreator(embedding=embeddings)
    doc=index.from_loaders([loader])
    chain = RetrievalQA.from_chain_type(llm=openai_llm, chain_type='stuff',retriever=doc.vectorstore.as_retriever(),input_key="question")#verbose =True,#return_source_documents=True,

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = create_pandas_dataframe_agent(openai_llm, 
                             df_list,
                            #  verbose=True,
                            allow_dangerous_code=True,
                             handle_parsing_errors=True)

    # prompt template
    suffix=''' Please note: if you want to take Action: python_repl_ast then before the action, sanitize Action Input by replacing "```" with "#" otherwise never add backticks "`" around the action input'''
    tools = [
        Tool(
        name="Sheet Answer Lookup",
        func=lambda query: chain({"question": query},return_only_outputs=True),
        description="useful for when you need to answer questions by just looking up information without computation. Input should be a fully formed question.",
            return_direct=True
    ),
        Tool(
        name="Sheet Answer Computation",
        func=lambda query: agent.run(query +suffix),
        description="useful for when you need to query Pandas DataFrames about computation involved in question on sheets. Will return a Pandas DataFrame."#return_direct=True
    ,return_direct=True)
    ]

    agent_exe = initialize_agent(
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        llm=openai_llm,
        tools=tools,
        # verbose=True,
        memory=memory
    )
    # initial context prompt
    agent_exe.run("Remember below contextual information regarding sheet data, which will be used further in question-answer-\n"+prefix_text)

    # new query
    while True:
        query = input("Enter your query: ")
        if query == "exit":
            break
        try:
            out=agent_exe.run(query)
            print("Response:\n",out)
        except:
            try:
                out=chain({"question": query},return_only_outputs=True)
                print("Response:\n",out)
            except Exception as e:
                print("Response:\n","Sorry, I am unable to answer this question.")
                print("Error:",e)
                continue
        # hallucination check

if __name__=="__main__":
    xlsx_file = sys.argv[1]

    # openai key setup
    api_key =input("Enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = api_key
    # Set your API key
    openai.api_key = api_key

    df_list,prefix_text = extract_tables(xlsx_file)
    print("preprocessing_stats")
    print(prefix_text)

    solution(xlsx_file, prefix_text, df_list)



