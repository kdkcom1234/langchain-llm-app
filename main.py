from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

def generate_pet_name(animal_type):
  # OpenAI 모델 객체 생성
  # temperature 0(low-creative, low-risk), 1(high-creative, high-risk)
  llm = OpenAI(temperature=0.7)

  # 메시지의 특정 영역을 변수화하여 재사용
  prompt_template_name = PromptTemplate(
    input_variables=['animal_type'],
    template="I have a {animal_type} pet and I want a cool name for it. Suggest me five cool names for my pet."
  )

  # 컴포넌트들을 연결(prompt -> llm)하여 체인 생성
  name_chain = LLMChain(llm=llm, prompt=prompt_template_name)

  # 체인에 매개변수를 넣어 응답 생성
  response = name_chain({'animal_type': animal_type})

  return response

if __name__ == "__main__":
  print(generate_pet_name("cat")["text"])
  print(generate_pet_name("pig")["text"])