# https://github.com/rishabkumar7/youtube-assistant-langchain
from langchain.document_loaders.youtube import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

video_url = "https://youtu.be/-Osca2Zax4Y?si=iyOiePxzUy_bUayO"

def create_vertor_db_from_youtube_url(video_url: str) -> FAISS:
  loader = YoutubeLoader.from_youtube_url(video_url)
  transcript = loader.load()

  # LLM(OpenAI)에 보낼 수 있는 토큰 개수가 한정적이어서 텍스트를 분할하고 벡터 저장소에 저장한다.
  # chunk_size는 청크의 글자 개수, chunk_overlap은 인접하는 청크가 교차되는 지점의 글자 개수
  # https://dev.to/peterabel/what-chunk-size-and-chunk-overlap-should-you-use-4338
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
  docs = text_splitter.split_documents(transcript)

  # 파시스는 먼저 데이터를 양자화하여 인덱스를 생성하고, 이후 이 인덱스를 사용하여 유사성 검색을 수행합니다.
  # https://dajeblog.co.kr/16-faiss%EC%97%90-%EB%8C%80%ED%95%9C-%EB%AA%A8%EB%93%A0-%EA%B2%83/
  db = FAISS.from_documents(docs, embeddings)
  return db

def get_response_from_query(db, query, k=4):
    """
    text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    # 벡터DB내에서 질의어로 유사도 검색을 진행하여 문서(유튜브 트랜스크립트의 일부분(청크)) 반환
    # 최대 결과 문서개수는 k개 만큼
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model_name="text-davinci-003")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs
