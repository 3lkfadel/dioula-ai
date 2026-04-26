"""
Dioula-AI — API FastAPI
Expose le modèle fine-tuné via des endpoints REST (réponses JSON)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from contextlib import asynccontextmanager
from inference import get_engine
import uuid
import time


# =============================================================
# Schémas de données
# =============================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    max_tokens: Optional[int] = Field(default=256, ge=10, le=512)
    temperature: Optional[float] = Field(default=0.3, ge=0.0, le=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Traduis en Dioula : Bonjour, comment vas-tu ?",
                "session_id": "user_123",
                "max_tokens": 200,
                "temperature": 0.3
            }
        }


class ChatResponse(BaseModel):
    success: bool = True
    session_id: str
    reponse: str
    langue_detectee: Optional[str] = None
    langue_reponse: Optional[str] = None
    mode: Optional[str] = None          # "traduction" | "conversation"
    latence_ms: Optional[int] = None
    metadata: Optional[dict] = None


class TranslationRequest(BaseModel):
    texte: str = Field(..., min_length=1, max_length=1000)
    direction: str = Field(
        default="fr_to_dioula",
        pattern="^(fr_to_dioula|dioula_to_fr)$"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "texte": "Merci beaucoup, à demain !",
                "direction": "fr_to_dioula"
            }
        }


class TranslationResponse(BaseModel):
    success: bool = True
    texte_original: str
    traduction: str
    direction: str
    langue_source: str
    langue_cible: str
    latence_ms: Optional[int] = None


# =============================================================
# App FastAPI
# =============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pré-charge le modèle au démarrage (évite le cold start sur le 1er appel)
    print("🔄 Pré-chargement du modèle Dioula-AI...")
    get_engine()
    print("✅ API prête !")
    yield

app = FastAPI(
    title="Dioula-AI API",
    description=(
        "API de traduction et conversation en Dioula (Jula) — "
        "Modèle LLaMA 3.2 fine-tuné localement sur Mac M1"
    ),
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================
# Endpoints
# =============================================================

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Dioula-AI",
        "version": "1.0.0",
        "status": "running",
        "langues": ["dioula", "français"],
        "endpoints": ["/api/v1/chat", "/api/v1/translate", "/docs"]
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "model": "llama-3.2-3b-dioula-finetuned"}


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Conversation en Dioula / Français.
    La langue est détectée automatiquement.
    Supporte les traductions et les conversations libres.
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())
        start = time.time()

        engine = get_engine()
        result = engine.generate_response(
            message=request.message,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        latence = int((time.time() - start) * 1000)

        # Détecte le mode (traduction vs conversation)
        msg_lower = request.message.lower()
        mode = "traduction" if any(
            w in msg_lower for w in ["traduis", "traduction", "translate", "comment dit-on"]
        ) else "conversation"

        return ChatResponse(
            success=True,
            session_id=session_id,
            reponse=result["reponse"],
            langue_detectee=result["langue_detectee"],
            langue_reponse=result["langue_reponse"],
            mode=mode,
            latence_ms=latence,
            metadata={"model": "llama-3.2-3b-dioula"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail={"success": False, "error": str(e)})


@app.post("/api/v1/translate", response_model=TranslationResponse, tags=["Traduction"])
async def translate(request: TranslationRequest):
    """
    Traduction directe Français ↔ Dioula.
    direction: 'fr_to_dioula' ou 'dioula_to_fr'
    """
    try:
        start = time.time()
        engine = get_engine()

        if request.direction == "fr_to_dioula":
            prompt = f"Traduis en Dioula : {request.texte}"
            langue_source, langue_cible = "français", "dioula"
        else:
            prompt = f"Traduis en français : {request.texte}"
            langue_source, langue_cible = "dioula", "français"

        result = engine.generate_response(
            message=prompt,
            max_tokens=256,
            temperature=0.1,  # Plus déterministe pour la traduction
        )

        latence = int((time.time() - start) * 1000)

        return TranslationResponse(
            success=True,
            texte_original=request.texte,
            traduction=result["reponse"],
            direction=request.direction,
            langue_source=langue_source,
            langue_cible=langue_cible,
            latence_ms=latence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail={"success": False, "error": str(e)})
