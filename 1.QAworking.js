import dotenv from 'dotenv';
dotenv.config();

import { YoutubeLoader } from "langchain/document_loaders/web/youtube";
import { FaissStore } from "langchain/vectorstores/faiss";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
// import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { Document } from "langchain/document";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { RetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";

import readline from 'readline';

/////////////////////////////////////
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function getInput(prompt) {
  return new Promise((resolve) => {
    rl.question(prompt, (input) => {
      resolve(input);
    });
  });
}
/////////////////////////////////////

import fs from 'fs';

const loader = YoutubeLoader.createFromUrl("https://youtu.be/bZQun8Y4L2A", {
  language: "en",
  addVideoInfo: false,
});

// // Load the data  
const data = await loader.load();

// fs.writeFileSync('docs.json', JSON.stringify(docs, null, 2));
// let rawdata = fs.readFileSync('./docs.json');
// let docs = JSON.parse(rawdata);
// let data = docs[0].pageContent;
// console.log(typeof output);

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 100,
});

// Split the the data into chunks
const splitDocs = await textSplitter.splitDocuments(data);


// const newOutput = await splitter.createDocuments([output]);
// console.log(newOutput);

const vectorStore = await FaissStore.fromDocuments(
  splitDocs,
  new OpenAIEmbeddings()
);

// const resultOne = await vectorStore.similaritySearch("speakers domain knowledge", 1);
// console.log(resultOne);

const model = new ChatOpenAI({ modelName: "gpt-3.5-turbo" });
const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

// const response = await chain.call({
//   query: "What's the summary of this talk?"
// });
// console.log(response);


let query = '';
while (query !== 'q') {
  query = await getInput("Enter your query (or 'q' to quit): ");
  if (query !== 'q') {
    const response = await chain.call({ query });
    console.log(response);
  }
}


// const text = `Hi.\n\nI'm Harrison.\n\nHow? Are? You?\nOkay then f f f f.
// This is a weird text to write, but gotta test the splittingggg some how.\n\n
// Bye!\n\n-H.`;
// const splitter = new RecursiveCharacterTextSplitter({
//   chunkSize: 10,
//   chunkOverlap: 1,
// });

