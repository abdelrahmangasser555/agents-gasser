import pickle
from langchain.vectorstores import Pinecone
import hashlib
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.query_constructor.base import AttributeInfo
from classes import SelfQueryRetrieverNew

pinecone.init(api_key="c869cafc-6f9a-4abf-b8ca-d24ebc2f6ccd", environment="us-west1-gcp-free")
# pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="us-west1-gcp-free")

vectorstore = Pinecone.from_existing_index(index_name="rings-new", embedding=OpenAIEmbeddings())

metadata_field_info = [
    AttributeInfo(name="type", description="The type of the product, only one of (ring, necklace, earring)", type="string"),
    AttributeInfo(name="material", description="The material of the product, only one of (gold, silver, platinum)", type="string"),
    AttributeInfo(name="price", description="The price of the product", type="integer"),
]

document_content_description = "Info about the Jewelry store products"
llm = OpenAI(temperature=0)

self_query_retriever_jewelry = SelfQueryRetrieverNew.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
    verbose=True,
    use_original_query=True,
)
