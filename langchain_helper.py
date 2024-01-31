from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import create_react_agent
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import AgentExecutor

from dotenv import load_dotenv

load_dotenv()

def generate_pet_name(animal_type, pet_color):
  # OpenAI 모델 객체 생성
  # temperature 0(low-creative, low-risk), 1(high-creative, high-risk)
  llm = OpenAI(temperature=0.7)

  # 메시지의 특정 영역을 변수화하여 재사용
  prompt_template_name = PromptTemplate(
    input_variables=['animal_type', 'pet_color'],
    template="""
    I have a {animal_type} pet and I want a cool name for it. 
    it is {pet_color} Suggest me five cool names for my pet. 
    """
  )

  # 컴포넌트들을 연결(prompt -> llm)하여 체인 생성
  name_chain = LLMChain(llm=llm, prompt=prompt_template_name)

  # 체인에 매개변수를 넣어 응답 생성
  response = name_chain({'animal_type': animal_type, 'pet_color': pet_color})

  return response

def langchain_agent():
  llm = OpenAI(temperature=0.5)

  # 사용할 툴(외부 도구들)
  # tools = load_tools(["wikipedia", "llm-math"], llm=llm)
  tools = load_tools(["wikipedia"], llm=llm)

  # 에이전트 생성
  agent = initialize_agent(llm=llm, tools=tools, verbose=True)
  # result = agent.run("What is the average age of a dog? Multiply the age by 3")
  result = agent.run("average age of dogs")

  print(result)

if __name__ == "__main__":
  langchain_agent()
  # print(generate_pet_name("cat", "silver")["text"])
  # print(generate_pet_name("pig", "black")["text"])