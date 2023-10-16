import dotenv from 'dotenv';
import axios from 'axios';
dotenv.config();

import fs from 'fs';
import { YoutubeLoader } from "langchain/document_loaders/web/youtube";
import { FaissStore } from "langchain/vectorstores/faiss";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { Document } from "langchain/document";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { RetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";

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

// convert the documents into vectors
const vectorStore = await FaissStore.fromDocuments(
  splitDocs,
  new OpenAIEmbeddings()
);

// Set the Pinecone API key
const apiKey = 'YOUR_API_KEY'; // replace with your Pinecone API key

// Create an axios instance with the Pinecone base URL and headers
const pinecone = axios.create({
  baseURL: 'https://api.pinecone.io',
  headers: {
    'api-key': apiKey,
    'Content-Type': 'application/json'
  }
});

// Function to upsert vectors
async function upsertVectors(indexName, ids, vectors) {
  const response = await pinecone.post(`/v1/index/${indexName}/vectors/upsert`, {
    ids: ids,
    vectors: vectors
  });
  return response.data;
}

// Example usage
const indexName = 'my-index';
const ids = splitDocs.map((doc, index) => `doc${index}`); // create unique ids for each document
const vectors = vectorStore.vectors; // get the vectors from the vectorStore

// upsertVectors(indexName, ids, vectors)
//   .then(response => console.log(response))
//   .catch(error => console.error(error));

console.log('ids: ', ids[0]);
console.log('vectors: ', vectors);

