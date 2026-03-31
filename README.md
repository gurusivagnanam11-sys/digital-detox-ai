# Digital Detox AI Chatbot

Research-level AI system for digital wellness using LLMs, ML habit analysis, and behavioral analytics.

## Project Structure

```
digital_detox_ai/
├── backend/
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Configuration & environment
│   ├── models/
│   │   ├── database.py            # SQLAlchemy models
│   │   └── schemas.py             # Pydantic request/response schemas
│   ├── api/
│   │   ├── routes/
│   │   │   ├── auth.py            # Authentication endpoints
│   │   │   ├── usage.py           # Screen time tracking endpoints
│   │   │   ├── chatbot.py         # LLM chatbot endpoints
│   │   │   ├── analytics.py       # Analytics & reports endpoints
│   │   │   └── focus.py           # Focus mode / Pomodoro endpoints
│   │   └── middleware.py          # Rate limiting, logging
│   ├── pipeline/
│   │   ├── preprocessor.py        # Data preprocessing
│   │   ├── feature_engineer.py    # Feature engineering
│   │   ├── nlp_pipeline.py        # NLP preprocessing
│   │   ├── vectorizer.py          # Embeddings & vectorization
│   │   └── data_collector.py      # Data ingestion
│   ├── analytics/
│   │   ├── habit_engine.py        # ML habit detection models
│   │   ├── wellness_score.py      # Digital wellness score
│   │   ├── streak_tracker.py      # Detox streak logic
│   │   ├── report_generator.py    # Weekly report builder
│   │   └── anomaly_detector.py    # Anomaly/addiction detection
│   ├── chatbot/
│   │   ├── llm_engine.py          # LLM integration (OpenAI/Llama)
│   │   ├── rag_system.py          # RAG retrieval system
│   │   ├── prompt_templates.py    # System & user prompt templates
│   │   └── conversation_manager.py# Multi-turn conversation state
│   ├── vector_db/
│   │   ├── chroma_store.py        # ChromaDB vector operations
│   │   └── faiss_index.py         # FAISS similarity search
│   └── utils/
│       ├── logger.py              # Structured logging
│       ├── cache.py               # Redis caching
│       └── validators.py          # Input validation helpers
├── frontend/
│   ├── index.html                 # Main SPA
│   ├── app.js                     # React application
│   └── styles.css                 # Design system styles
├── ml_models/
│   ├── train_habit_classifier.py  # Training script
│   ├── train_anomaly_detector.py  # Anomaly model training
│   └── saved_models/              # Persisted model artifacts
├── data/
│   ├── sample_usage_data.json     # Sample behavioral data
│   └── wellness_knowledge_base/   # RAG knowledge documents
├── tests/
│   ├── test_pipeline.py
│   ├── test_analytics.py
│   └── test_chatbot.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
copy .env.example .env  # Windows PowerShell / cmd
# Edit .env with your OpenAI API key and DB config

# Initialize database (creates SQLite DB via SQLAlchemy)
python -c "from database import init_db; init_db()"

# Run the server (module is `main.py` at repo root)
uvicorn main:app --reload --port 8000

# Serve the frontend (simple static server from project root)
# Option 1: open index.html directly in a browser
# Option 2: run a simple HTTP server on port 3000
python -m http.server 3000

# Run tests
pytest tests/ -v
```
