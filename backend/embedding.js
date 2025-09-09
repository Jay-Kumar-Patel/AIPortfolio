import fs from "fs";
import path from "path";
import dotenv from "dotenv";
import { LlamaParseReader } from "llamaindex";
import { ChromaClient } from "chromadb";
import OpenAI from "openai";

dotenv.config();

const chromaClient = new ChromaClient({
  host: "localhost",
  port: 8000,
});

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

chromaClient
  .heartbeat()
  .then(() => {
    console.log("ChromaDB is running");
  })
  .catch((err) => {
    console.error("ChromaDB is not running:", err);
  });

function generateCollectionName(prefix = "collection") {
  const timestamp = Date.now();
  const random = Math.random().toString(36).substring(2, 8);
  return `${prefix}_${timestamp}_${random}`;
}

async function saveCollectionName(collectionName, source) {
  const collectionsFile = path.join("app", "collections.txt");
  const entry = `${collectionName}|${source}\n`;

  try {
    if (!fs.existsSync("app")) {
      fs.mkdirSync("app", { recursive: true });
    }
    fs.appendFileSync(collectionsFile, entry);
    console.log(`Saved collection ${collectionName} to collections.txt`);
  } catch (error) {
    console.error("Error saving collection name:", error);
  }
}

async function loadCollectionNames() {
  const collectionsFile = path.join("app", "collections.txt");

  try {
    if (!fs.existsSync(collectionsFile)) {
      return [];
    }

    const content = fs.readFileSync(collectionsFile, "utf8");
    const lines = content
      .trim()
      .split("\n")
      .filter((line) => line.trim() !== "");

    return lines.map((line) => {
      const [name, source] = line.split("|");
      return { name, source };
    });
  } catch (error) {
    console.error("Error loading collection names:", error);
    return [];
  }
}

function clearCollectionsFile() {
  const collectionsFile = path.join("app", "collections.txt");
  try {
    if (fs.existsSync(collectionsFile)) {
      fs.unlinkSync(collectionsFile);
    }
  } catch (error) {
    console.error("Error clearing collections file:", error);
  }
}

async function createEmbedding(text) {
  try {
    const response = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: text,
      encoding_format: "float",
    });
    return response.data[0].embedding;
  } catch (error) {
    console.error("Error creating embedding:", error);
    throw error;
  }
}

function findTextOverlap(text1, text2) {
  const words1 = text1.split(" ");
  const words2 = text2.split(" ");

  let maxOverlap = "";
  for (let i = 1; i <= Math.min(words1.length, words2.length); i++) {
    const ending = words1.slice(-i).join(" ");
    const beginning = words2.slice(0, i).join(" ");

    if (ending === beginning && ending.length > maxOverlap.length) {
      maxOverlap = ending;
    }
  }
  return maxOverlap;
}

async function processSinglePDF(filePath) {
  try {
    const fileName = path.basename(filePath);

    let source = null;
    const splitFileName = fileName.split("_");
    if(splitFileName.length > 1) {
      source = splitFileName[0].toLowerCase();
    } else{
      source = fileName.toLowerCase();
    }

    const collectionName = generateCollectionName(
      `doc_${fileName.replace(/\.[^/.]+$/, "")}`
    );

    console.log(`\nProcessing file: ${fileName}`);
    console.log(`Creating collection: ${collectionName}`);

    const reader = new LlamaParseReader({
      apiKey: process.env.LLAMA_CLOUD_API_KEY,
      chunkSize: 1000,
      chunkOverlap: 100,
      resultType: "markdown",
      verbose: true,
      parsingInstruction: `
  Extract all content with high fidelity, focusing on:
  
  - Preserve document structure and meaning of each word related to that context
  - Extract tables with complete column/row relationships intact
  - Capture project details including what problems it solves, technologies/methodologies used, and outcomes
  - Preserve numerical data with proper context (dates, percentages, metrics)
  - Maintain bullet points and numbered lists with their parent context
  - Capture everything about skills, certifications, education, technical qualifications, experiences, and projects with exact wording
  - Preserve chronological order of all work
  - Identify and flag key achievements and highlights
  - Convert charts, diagrams, and images into textual insights with quantitative data
  
  Process each document segment in context of surrounding content. Prioritize accuracy over brevity, especially for technical terms, proper nouns, and credentials.
`,
      skipDiagonalText: false,
      fastMode: false,
      doNotUnrollColumns: false,
      pagePrefix: "PAGE_",
    });

    const collection = await chromaClient.createCollection({
      name: collectionName,
      embeddingFunction: null,
    });

    const documents = await reader.loadData(filePath);
    console.log(`Extracted ${documents.length} chunks from ${fileName}`);

    const ids = [];
    const metadatas = [];
    const docTexts = [];
    const embeddings = [];

    for (let i = 0; i < documents.length; i++) {
      const doc = documents[i];
      const chunkText = doc.text || doc.getText();

      if (!chunkText || chunkText.trim().length === 0) {
        console.log(`Skipping empty chunk ${i + 1} from ${fileName}`);
        continue;
      }

      console.log(
        `Creating embedding for chunk ${i + 1}/${
          documents.length
        } from ${fileName}...`
      );

      const embedding = await createEmbedding(chunkText);

      let overlapInfo = null;
      if (i > 0) {
        const prevText = documents[i - 1].text || documents[i - 1].getText();
        if (prevText) {
          const overlap = findTextOverlap(prevText, chunkText);
          if (overlap.length > 20) {
            overlapInfo = {
              withChunk: `chunk_${i - 1}`,
              overlapLength: overlap.length,
            };
          }
        }
      }

      ids.push(`chunk_${i}`);
      docTexts.push(chunkText);
      embeddings.push(embedding);
      metadatas.push({
        chunkIndex: i,
        fileName: fileName,
        totalChunksInFile: documents.length,
        wordCount: chunkText.split(" ").length,
        previousChunk: i > 0 ? `chunk_${i - 1}` : "",
        nextChunk: i < documents.length - 1 ? `chunk_${i + 1}` : "",
        hasOverlap: overlapInfo !== null,
        overlapWith: overlapInfo ? overlapInfo.withChunk : "",
        overlapLength: overlapInfo ? overlapInfo.overlapLength : 0,
        source: source,
      });
    }

    if (ids.length > 0) {
      console.log(`Storing ${ids.length} chunks in ChromaDB...`);
      await collection.add({
        ids: ids,
        metadatas: metadatas,
        documents: docTexts,
        embeddings: embeddings,
      });
      console.log(`Successfully stored ${ids.length} chunks from ${fileName}`);
      await saveCollectionName(collectionName, `document:${fileName}`);
    }

    return collectionName;
  } catch (error) {
    console.error(`Error processing file ${filePath}:`, error);
    return null;
  }
}

async function processAllPDFs(files) {
  const collectionNames = [];

  for (const filePath of files) {
    const collectionName = await processSinglePDF(filePath);
    if (collectionName) {
      collectionNames.push(collectionName);
    }
  }

  return collectionNames;
}

async function getAllFiles(dirPath) {
  const files = fs.readdirSync(dirPath);
  const filePaths = [];

  for (const file of files) {
    const filePath = path.join(dirPath, file);
    if (fs.statSync(filePath).isDirectory()) {
      filePaths.push(...(await getAllFiles(filePath)));
    } else {
      filePaths.push(filePath);
    }
  }

  return filePaths;
}

async function searchAllCollections(query, topK = 3) {
  try {
    const collections = await loadCollectionNames();
    let allResults = [];
    
    console.log(`Searching across ${collections.length} collections...`);
    
    for (const collectionInfo of collections) {
      try {
        const collection = await chromaClient.getCollection({ name: collectionInfo.name, embeddingFunction: null });
        const queryEmbedding = await createEmbedding(query);
        
        const results = await collection.query({
          queryEmbeddings: [queryEmbedding],
          nResults: topK,
        });
        
        if (results.documents && results.documents[0].length > 0) {
          const resultsWithSource = results.documents[0].map((doc, index) => ({
            document: doc,
            distance: results.distances[0][index],
            metadata: results.metadatas[0][index],
            source: collectionInfo.source,
            collection: collectionInfo.name
          }));
          
          allResults.push(...resultsWithSource);
        }
      } catch (error) {
        console.log(`Skipping collection ${collectionInfo.name}: ${error.message}`);
      }
    }
    
    allResults.sort((a, b) => a.distance - b.distance);
    return allResults.slice(0, topK * 2);
    
  } catch (error) {
    console.error("Error searching collections:", error);
    throw error;
  }
}

async function generateResponse(userQuestion, relevantResults) {
  try {
    let contextText = "";

    if (relevantResults.length > 0) {
      contextText = relevantResults
        .map(
          (result, index) =>
            `Source ${index + 1} (${result.source}):\n${result.document}`
        )
        .join("\n\n");
    }

    const systemPrompt = `You are an intelligent portfolio assistant for Jay Patel, helping visitors learn about Jay's professional journey, projects, and expertise.

**Your Voice**
- Speak in first person as if you ARE Jay: "I have experience in..." not "Jay has experience in..."
- Be warm, professional, and confident about your accomplishments
- Always give a natural human touch to responses so they feel conversational and not robotic

**Response Guidelines:**
- Always answer in a maximum of 3–4 sentences
- Always answer in paragraph form (not list format)
- Provide concise, specific details directly addressing the question

(Examples of specificity)  
1) If a user asks about my education, only mention degree, school, and duration — don’t list courses unless specifically requested.  
2) If a user asks about my experience background, only mention companies, roles, and duration — don’t describe responsibilities unless the question asks for them.  

- Apply this principle of answering *only what is asked* to all responses
- Do not elaborate beyond the question’s scope
- If a question involves multiple subjects (e.g., different companies, projects, or education), provide a clear high-level summary in paragraph form, keeping it natural and easy to read
- Do not add generic closing statements or invitations for more questions
- Never use “--” for bulleting or emphasis

**When Information is Limited:**
- Say: "As of now I don’t have any knowledge about this, I am really sorry for that."
- Don’t fabricate or add unrelated details

Context from my portfolio materials:  
${contextText}

Remember: Respond as if you ARE Jay Patel discussing your own experience and expertise.`;

    const response = await openai.chat.completions.create({
      model: "gpt-5",
      messages: [
        {
          role: "system",
          content: systemPrompt,
        },
        {
          role: "user",
          content: userQuestion,
        },
      ],
    });

    return response.choices[0].message.content;
  } catch (error) {
    console.error("Error generating response:", error);
    throw error;
  }
}

export async function handleUserQuestion(question) {
  try {
    const searchResults = await searchAllCollections(question);

    if (!searchResults || searchResults.length === 0) {
      return "I couldn't find relevant information in the portfolio documents to answer your question.";
    }

    const response = await generateResponse(question, searchResults);
    return response;
  } catch (error) {
    console.error("Error handling user question:", error);
    return "I'm sorry, I encountered an error while processing your question.";
  }
}

export async function initialize() {
  clearCollectionsFile();
  const files = await getAllFiles("./doc");
  console.log("Processing PDF documents...");
  await processAllPDFs(files);
  console.log("Initialization complete.");
}
