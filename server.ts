import "dotenv/config";
import express from "express";
import type { Request, Response } from "express";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { GoogleGenAI, ThinkingLevel, Type } from "@google/genai";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const GEMINI_MODEL = "gemini-4-hydrogen";
const GEMINI_FALLBACK_MODEL = "gemini-3.1-flash-lite";

const LANGUAGE_LABELS: Record<string, string> = {
  en: "English",
  bn: "Bengali (বাংলা)",
  hi: "Hindi (हिन्दी)",
};

function getGenAIClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error(
      "GEMINI_API_KEY is not configured. Please check Settings > Secrets."
    );
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

function isRetryableError(err: unknown): boolean {
  const raw = err instanceof Error ? err.message : String(err);
  return (
    raw.includes("503") ||
    raw.includes("UNAVAILABLE") ||
    raw.includes("429") ||
    raw.includes("RESOURCE_EXHAUSTED") ||
    raw.includes("overloaded") ||
    raw.includes("high demand")
  );
}

async function callWithFallback<T>(
  fn: (model: string) => Promise<T>
): Promise<T> {
  const attempts = [GEMINI_MODEL, GEMINI_MODEL, GEMINI_FALLBACK_MODEL];
  let lastErr: unknown;
  for (let i = 0; i < attempts.length; i++) {
    try {
      return await fn(attempts[i]);
    } catch (err) {
      lastErr = err;
      if (i < attempts.length - 1 && isRetryableError(err)) {
        await new Promise((resolve) => setTimeout(resolve, 600 * (i + 1)));
        continue;
      }
      throw err;
    }
  }
  throw lastErr;
}

function getFriendlyError(err: unknown): { status: number; message: string } {
  const raw = err instanceof Error ? err.message : String(err);
  if (raw.includes("GEMINI_API_KEY")) {
    return { status: 500, message: raw };
  }
  if (
    raw.includes("API_KEY_INVALID") ||
    raw.includes("PERMISSION_DENIED") ||
    raw.includes("403")
  ) {
    return {
      status: 403,
      message:
        "Gemini API key was rejected. Please verify your API key in Settings > Secrets.",
    };
  }
  if (raw.includes("503") || raw.includes("UNAVAILABLE") || raw.includes("high demand")) {
    return {
      status: 503,
      message:
        "Gemini is temporarily busy due to high demand. Please wait a few seconds and tap Try again.",
    };
  }
  if (raw.includes("RESOURCE_EXHAUSTED") || raw.includes("429")) {
    return {
      status: 429,
      message: "Too many requests right now. Please wait a moment and try again.",
    };
  }
  // Extract clean message if raw is JSON-encoded ApiError
  try {
    const parsed = JSON.parse(raw);
    const nestedMsg = parsed?.error?.message;
    if (typeof nestedMsg === "string" && nestedMsg.trim()) {
      return { status: 500, message: nestedMsg.trim() };
    }
  } catch {
    // Not JSON
  }
  return {
    status: 500,
    message: raw || "Failed to reach Gemini. Please try again.",
  };
}

const analyzeSchema = {
  type: Type.OBJECT,
  properties: {
    productName: {
      type: Type.STRING,
      description: "Product name as shown on the label, or 'Unknown product' if not visible.",
    },
    brand: {
      type: Type.STRING,
      description: "Brand name if visible on the label, otherwise 'Not visible on label'.",
    },
    category: {
      type: Type.STRING,
      enum: ["food", "cosmetic", "other"],
      description: "Product category: food, cosmetic, or other.",
    },
    summary: {
      type: Type.STRING,
      description:
        "Plain-language 2-3 sentence overview in the user's chosen language. If the image is not a product label, state that clearly here.",
    },
    extracted: {
      type: Type.OBJECT,
      properties: {
        ingredients: {
          type: Type.ARRAY,
          items: { type: Type.STRING },
          description:
            "Exact ingredient names as printed on the label in order (keep INS/E numbers). Empty array if none visible.",
        },
        nutrition: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              name: { type: Type.STRING },
              amount: { type: Type.STRING },
            },
            required: ["name", "amount"],
          },
          description:
            "Nutrition facts rows exactly as printed on the label (name and amount with unit). Empty array if not visible.",
        },
        allergens: {
          type: Type.ARRAY,
          items: { type: Type.STRING },
          description:
            "Allergen declarations printed on the label (e.g. Contains Milk, Wheat). Empty array if none printed.",
        },
        warnings: {
          type: Type.ARRAY,
          items: { type: Type.STRING },
          description:
            "Usage, storage, or safety warnings printed on the label. Empty array if none printed.",
        },
        manufacturer: {
          type: Type.STRING,
          description: "Manufacturer or marketer name as printed, or 'Not visible on label'.",
        },
        servingSize: {
          type: Type.STRING,
          description: "Serving size or reference basis (e.g. 'Per 100g', '30g'), or 'Not visible on label'.",
        },
      },
      required: [
        "ingredients",
        "nutrition",
        "allergens",
        "warnings",
        "manufacturer",
        "servingSize",
      ],
    },
    ingredientInsights: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          name: { type: Type.STRING },
          purpose: {
            type: Type.STRING,
            description:
              "Short functional role (e.g. Sweetener, Emulsifier, Humectant, Preservative, Surfactant).",
          },
          explanation: {
            type: Type.STRING,
            description:
              "Simple explanation suitable for a school student, in the user's chosen language.",
          },
          concern: {
            type: Type.STRING,
            enum: ["none", "low", "medium", "high"],
          },
        },
        required: ["name", "purpose", "explanation", "concern"],
      },
      description: "4 to 10 key ingredients explained simply in the user's chosen language.",
    },
    nutritionNotes: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description:
        "Short, plain-language observations about the nutrition profile in the user's chosen language.",
    },
    personalAlerts: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          ingredient: { type: Type.STRING },
          reason: { type: Type.STRING },
          severity: {
            type: Type.STRING,
            enum: ["low", "medium", "high"],
          },
        },
        required: ["ingredient", "reason", "severity"],
      },
      description:
        "Personalized alerts matching the user's avoid list or notes, including synonyms and derived sources. Empty array if none.",
    },
    labelReadability: {
      type: Type.STRING,
      description:
        "Short note in the user's chosen language on how clear or legible the label photo was and whether any parts were cut off or blurry.",
    },
  },
  required: [
    "productName",
    "brand",
    "category",
    "summary",
    "extracted",
    "ingredientInsights",
    "nutritionNotes",
    "personalAlerts",
    "labelReadability",
  ],
};

const compareSchema = {
  type: Type.OBJECT,
  properties: {
    summary: {
      type: Type.STRING,
      description: "Short plain-language comparison summary (2-3 sentences) in the user's chosen language.",
    },
    differences: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          title: { type: Type.STRING },
          detail: { type: Type.STRING },
        },
        required: ["title", "detail"],
      },
      description: "3 to 6 clear key differences between the two products in the user's chosen language.",
    },
  },
  required: ["summary", "differences"],
};

async function startServer() {
  const app = express();
  app.use(express.json({ limit: "15mb" }));

  app.post("/api/analyze-label", async (req: Request, res: Response) => {
    try {
      const {
        imageBase64,
        mimeType = "image/jpeg",
        language = "en",
        avoidList = [],
        notes = "",
      } = req.body ?? {};

      if (!imageBase64 || typeof imageBase64 !== "string") {
        res.status(400).json({ error: "Missing label image data." });
        return;
      }

      const ai = getGenAIClient();
      const targetLang = LANGUAGE_LABELS[language] ?? "English";
      const avoidItems = Array.isArray(avoidList)
        ? avoidList.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
        : [];
      const userNotes = typeof notes === "string" ? notes.trim() : "";

      const prompt = [
        "You are LabelLens, a careful, neutral product label reader and ingredient explainer.",
        `Target language for all interpretations, summaries, purposes, explanations, nutritionNotes, personalAlerts reasons, and labelReadability: ${targetLang}.`,
        "",
        "STRICT RULES:",
        '1. "extracted" MUST contain ONLY text actually printed on the label image. Keep ingredient names exactly as printed (including INS / E numbers like INS 500(ii) or E322). NEVER invent or guess missing label facts. If a field is not visible, use an empty array [] or the exact string "Not visible on label".',
        `2. Everything outside "extracted" (summary, ingredientInsights, nutritionNotes, personalAlerts, labelReadability) MUST be written in ${targetLang}, in clear, everyday language simple enough for a school student.`,
        "3. Cover 4 to 10 of the most important or notable ingredients in ingredientInsights (or fewer if the product has fewer ingredients). Do not exaggerate risks; follow mainstream regulatory consensus (FDA, EFSA, FSSAI, CIR).",
        "4. For cosmetics and skincare products, clearly explain each ingredient's cosmetic function in 'purpose' (e.g., humectant, preservative, emollient, surfactant, antioxidant, pH adjuster) and explain what it does simply.",
        "5. Check the extracted ingredients and allergens against the user's avoid list and personal notes for personalAlerts — INCLUDE synonyms, chemical names, and derived sources (for example: casein, whey, lactose, milk solids -> milk/dairy; arachis oil -> peanut; sodium lauryl sulfate -> sulphate; parfum/linalool -> fragrance; wheat/barley/maida -> gluten). If nothing conflicts, return an empty array [].",
        "6. Never give medical diagnoses or medical advice.",
        "7. If the photo does NOT show a food, cosmetic, or consumer product label (or is completely unreadable), clearly explain that in 'summary', set category to 'other', keep extracted lists empty and strings as 'Not visible on label', and describe the issue in 'labelReadability'.",
        "",
        `User avoid list: ${avoidItems.length > 0 ? avoidItems.join(", ") : "None"}`,
        `User notes: ${userNotes || "None"}`,
      ].join("\n");

      const response = await callWithFallback((model) =>
        ai.models.generateContent({
          model,
          contents: {
            parts: [
              {
                inlineData: {
                  mimeType: typeof mimeType === "string" ? mimeType : "image/jpeg",
                  data: imageBase64,
                },
              },
              { text: prompt },
            ],
          },
          config: {
            responseMimeType: "application/json",
            responseSchema: analyzeSchema,
            thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          },
        })
      );

      const rawText = (response.text ?? "").trim();
      if (!rawText) {
        res.status(502).json({ error: "Gemini returned an empty response. Please try again." });
        return;
      }

      const parsed = JSON.parse(rawText);
      res.json(parsed);
    } catch (err) {
      console.error("[/api/analyze-label] Error:", err);
      const { status, message } = getFriendlyError(err);
      res.status(status).json({ error: message });
    }
  });

  app.post("/api/compare-products", async (req: Request, res: Response) => {
    try {
      const {
        productA,
        productB,
        language = "en",
        avoidList = [],
        notes = "",
      } = req.body ?? {};

      if (!productA || !productB) {
        res.status(400).json({ error: "Two products are required for comparison." });
        return;
      }

      const ai = getGenAIClient();
      const targetLang = LANGUAGE_LABELS[language] ?? "English";
      const avoidItems = Array.isArray(avoidList)
        ? avoidList.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
        : [];
      const userNotes = typeof notes === "string" ? notes.trim() : "";

      const prompt = [
        "You are LabelLens comparing two scanned consumer products side-by-side.",
        `Write the entire JSON response ('summary' and each item in 'differences') in ${targetLang}, using simple, friendly, school-student-level language.`,
        "Give an objective 2-3 sentence summary and 3 to 6 concrete key differences covering nutrition, key ingredients, additives, allergens, or formulation differences.",
        "Do not exaggerate risks; follow mainstream regulatory consensus; never give medical diagnoses.",
        avoidItems.length > 0 || userNotes
          ? `Keep the user's preferences in mind when relevant — Avoid list: ${avoidItems.join(", ") || "None"}; Notes: ${userNotes || "None"}.`
          : "",
        "",
        `Product A: ${JSON.stringify(productA)}`,
        `Product B: ${JSON.stringify(productB)}`,
      ]
        .filter(Boolean)
        .join("\n");

      const response = await callWithFallback((model) =>
        ai.models.generateContent({
          model,
          contents: prompt,
          config: {
            responseMimeType: "application/json",
            responseSchema: compareSchema,
            thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          },
        })
      );

      const rawText = (response.text ?? "").trim();
      if (!rawText) {
        res.status(502).json({ error: "Gemini returned an empty comparison. Please try again." });
        return;
      }

      const parsed = JSON.parse(rawText);
      res.json(parsed);
    } catch (err) {
      console.error("[/api/compare-products] Error:", err);
      const { status, message } = getFriendlyError(err);
      res.status(status).json({ error: message });
    }
  });

  const isProd = process.env.NODE_ENV === "production";
  const distDir = path.resolve(__dirname, "dist");

  if (isProd && fs.existsSync(distDir)) {
    app.use(express.static(distDir));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distDir, "index.html"));
    });
  } else {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  }

  const port = Number(process.env.PORT) || 3000;
  app.listen(port, "0.0.0.0", () => {
    console.log(`LabelLens server listening on http://0.0.0.0:${port}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
