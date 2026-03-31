"""
rag_system.py — Retrieval-Augmented Generation using ChromaDB.

Stores wellness knowledge articles (digital detox tips, research summaries,
CBT-based strategies) as vector embeddings and retrieves the most semantically
relevant passages to enrich LLM responses.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config import get_settings

logger = logging.getLogger(__name__)
app_settings = get_settings()


# ─── Wellness knowledge base ──────────────────────────────────────────────────
# In production, load from files or a CMS. These are built-in seed articles.

WELLNESS_KNOWLEDGE_BASE = [
    {
        "id": "wk_001",
        "title": "The 30-minute rule before bedtime",
        "content": (
            "Research shows that blue light from screens suppresses melatonin production. "
            "Avoid phone usage 30–60 minutes before sleep. Replace with reading, journaling, "
            "or light stretching. This single change improves sleep quality by up to 40%."
        ),
        "tags": ["sleep", "night", "bedtime", "blue light"],
    },
    {
        "id": "wk_002",
        "title": "The Pomodoro Technique for digital detox",
        "content": (
            "Work in 25-minute focused blocks followed by 5-minute breaks. "
            "During breaks, step away from all screens. After 4 Pomodoros, take a 15-30 minute break. "
            "This trains your attention span and reduces compulsive phone-checking habits."
        ),
        "tags": ["focus", "productivity", "pomodoro", "breaks"],
    },
    {
        "id": "wk_003",
        "title": "Social media scroll fatigue",
        "content": (
            "Social media platforms use variable-reward schedules (like slot machines) "
            "to trigger dopamine. Symptoms: mindless scrolling, feeling empty after, "
            "compulsive checking. Counter-strategies: set app timers, use grayscale mode, "
            "designate social media windows (twice daily, 15 minutes each)."
        ),
        "tags": ["social media", "addiction", "dopamine", "scrolling"],
    },
    {
        "id": "wk_004",
        "title": "Offline activity substitutions",
        "content": (
            "Replace phone time with: walking (10 min reduces phone craving), "
            "drawing/sketching, calling a friend instead of messaging, cooking a new recipe, "
            "gardening, board games, reading physical books, or meditation apps (limited use)."
        ),
        "tags": ["offline", "activities", "replacement", "hobbies"],
    },
    {
        "id": "wk_005",
        "title": "Notification management strategy",
        "content": (
            "Notifications are the #1 cause of fragmented attention. Best practice: "
            "disable all non-critical notifications, enable only calls and calendar alerts. "
            "Use 'Do Not Disturb' during work and sleep hours. Check messages on a schedule "
            "rather than reactively — this reclaims 2+ hours daily."
        ),
        "tags": ["notifications", "attention", "do not disturb", "focus"],
    },
    {
        "id": "wk_006",
        "title": "Digital sunset routine",
        "content": (
            "Create a phone-free bedtime ritual: 1 hour before sleep, plug your phone in "
            "a different room. Use an analogue alarm clock. This creates physical distance "
            "from the device and signals your brain that sleep is coming."
        ),
        "tags": ["sleep", "routine", "bedtime", "night"],
    },
    {
        "id": "wk_007",
        "title": "The 20-20-20 rule for eye and mental health",
        "content": (
            "Every 20 minutes of screen time: look at something 20 feet away for 20 seconds. "
            "Reduces digital eye strain and serves as a mindfulness break. "
            "Set a recurring 20-minute reminder or use an app timer."
        ),
        "tags": ["eye health", "breaks", "mindfulness", "fatigue"],
    },
    {
        "id": "wk_008",
        "title": "App category limits and screen time goals",
        "content": (
            "Research recommends: social media < 30 min/day, entertainment < 1 hour/day, "
            "total screen time < 3 hours/day for adults. Start by tracking, then cut usage "
            "by 15% each week — gradual reduction is more sustainable than cold-turkey approaches."
        ),
        "tags": ["limits", "goals", "screen time", "targets"],
    },
]


# ─── RAG System ──────────────────────────────────────────────────────────────

class WellnessRAGSystem:
    """
    Manages the ChromaDB vector store and semantic retrieval for RAG.
    """

    COLLECTION_NAME = "wellness_knowledge"

    def __init__(self):
        self.embedder = SentenceTransformer(app_settings.EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(
            path=app_settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._seed_knowledge_base()

    # ── Indexing ──────────────────────────────────────────────────────────────

    def _seed_knowledge_base(self) -> None:
        """Insert built-in knowledge articles if collection is empty."""
        existing = self.collection.count()
        if existing >= len(WELLNESS_KNOWLEDGE_BASE):
            logger.info("RAG: Knowledge base already indexed (%d docs)", existing)
            return

        texts = [doc["content"] for doc in WELLNESS_KNOWLEDGE_BASE]
        embeddings = self.embedder.encode(texts, batch_size=16).tolist()

        self.collection.upsert(
            ids=[doc["id"] for doc in WELLNESS_KNOWLEDGE_BASE],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {"title": doc["title"], "tags": ",".join(doc["tags"])}
                for doc in WELLNESS_KNOWLEDGE_BASE
            ],
        )
        logger.info("RAG: Seeded %d wellness knowledge articles", len(WELLNESS_KNOWLEDGE_BASE))

    def add_document(self, content: str, title: str, tags: list[str]) -> str:
        """Add a custom wellness document to the knowledge base."""
        doc_id = f"custom_{uuid.uuid4().hex[:8]}"
        embedding = self.embedder.encode(content).tolist()
        self.collection.add(
            ids=[doc_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[{"title": title, "tags": ",".join(tags)}],
        )
        return doc_id

    def add_user_interaction(self, user_id: str, message: str, response: str) -> None:
        """
        Optionally index high-quality past interactions for personalised retrieval.
        Only called when the interaction contains actionable advice.
        """
        combined = f"User concern: {message}\nAdvice given: {response}"
        self.add_document(
            content=combined,
            title=f"Personalised advice for user {user_id}",
            tags=["personalised", "interaction"],
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        n_results: int = 3,
        filter_tags: Optional[list[str]] = None,
    ) -> str:
        """
        Retrieve the most relevant wellness knowledge for a query.
        Returns a formatted string ready for LLM context injection.
        """
        query_embedding = self.embedder.encode(query).tolist()

        where_clause = None
        if filter_tags:
            # ChromaDB metadata filter: any document tagged with any of the tags
            where_clause = {"$or": [{"tags": {"$contains": tag}} for tag in filter_tags]}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count()),
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return ""

        context_parts = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = round(1 - dist, 3)
            if similarity > 0.3:  # only include sufficiently relevant docs
                context_parts.append(f"[{meta['title']}] {doc}")

        return "\n\n".join(context_parts)

    def retrieve_for_intent(self, query: str, intent: str) -> str:
        """Intent-aware retrieval with pre-mapped tag filters."""
        intent_tag_map = {
            "report_usage":    ["addiction", "limits"],
            "request_advice":  ["replacement", "focus", "limits"],
            "check_progress":  ["goals", "targets"],
            "start_focus":     ["pomodoro", "focus", "breaks"],
            "general_chat":    [],
        }
        tags = intent_tag_map.get(intent, [])
        return self.retrieve(query, n_results=2, filter_tags=tags or None)
