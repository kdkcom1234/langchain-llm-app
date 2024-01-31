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

  # 벡터는 텍스트를 숫자로 변환한 것
  db = FAISS.from_documents(docs, embeddings)
  return db

db = create_vertor_db_from_youtube_url(video_url)

print(db)