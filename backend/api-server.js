import express from "express";
import cors from "cors";
import { handleUserQuestion } from "./embedding.js";

const app = express();
app.use(cors());
app.use(express.json());

app.post("/api/ask", async (req, res) => {
  try {
    const { question } = req.body;
    if (!question || typeof question !== "string") {
      return res.status(400).json({ error: "Question is required." });
    }
    const answer = await handleUserQuestion(question);
    res.json({ answer });
  } catch (error) {
    res.status(500).json({ error: error.message || "Internal server error" });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`API server running on http://localhost:${PORT}`);
});