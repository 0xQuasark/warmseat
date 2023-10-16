import dotenv from 'dotenv';
dotenv.config();

const PINECONE_API_KEY = process.env.PINECONE_API_KEY;

import { Pinecone } from '@pinecone-database/pinecone';
import { VectorDBQAChain } from "langchain/chains";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { OpenAI } from "langchain/llms/openai";
import { RetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";

// import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const pinecone = new Pinecone({ 
  apiKey: PINECONE_API_KEY,
  environment: 'gcp-starter'
})

const pineconeIndex = pinecone.Index("warmseat");

const vectorStore = await PineconeStore.fromExistingIndex(
  new OpenAIEmbeddings(),
  { pineconeIndex }
);
  
// console.log(pineconeIndex);

// // console.log(await warmseatIndex.describeIndexStats());

const model = new ChatOpenAI({ modelName: "gpt-3.5-turbo" });
const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

// console.log(vectorStore.asRetriever());

const response = await chain.call({
  query: "What's the summary of this talk?"
});
console.log(response);



