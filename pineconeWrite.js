import dotenv from 'dotenv';
dotenv.config();

import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { YoutubeLoader } from "langchain/document_loaders/web/youtube";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
// import { FaissStore } from "langchain/vectorstores/faiss";
// import { RetrievalQAChain } from "langchain/chains";
// import { ChatOpenAI } from "langchain/chat_models/openai";

// import readline from 'readline';


const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
// console.log(PINECONE_API_KEY);

const loader = YoutubeLoader.createFromUrl("https://youtu.be/bZQun8Y4L2A", {
  language: "en",
  addVideoInfo: false,
});

// Load the data  
const data = await loader.load();


const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 100,
});

// Split the the data into chunks
const splitDocs = await textSplitter.splitDocuments(data);

const pinecone = new Pinecone({ 
  apiKey: PINECONE_API_KEY,
  environment: 'gcp-starter'
})

const pineconeIndex = pinecone.Index("warmseat");


let response = await PineconeStore.fromDocuments(splitDocs, new OpenAIEmbeddings(), {
  pineconeIndex,
  maxConcurrency: 5, // Maximum number of batch requests to allow at once. Each batch is 1000 vectors.
});

console.log(response);

// const vectorStore = await FaissStore.fromDocuments(
//   splitDocs,
//   new OpenAIEmbeddings()
// );


// const model = new ChatOpenAI({ modelName: "gpt-3.5-turbo" });
// const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

// const response = await chain.call({
//   query: "What's the summary of this talk?"
// });
// console.log(response);



// const list = await pinecone.listIndexes();

// await pinecone.describeIndex("warmseat");


// console.log(list);