import os
import torch
from sentence_transformers import SentenceTransformer

class SBERTLangEncoder:
    def __init__(self,device):
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        model_name = "sentence-transformers/all-mpnet-base-v2"
        self.device = device
        
        self.model = SentenceTransformer(model_name, device=device)
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def get_lang_emb(self, lang):
        if lang is None:
            return None
        
        with torch.no_grad():
            lang_emb = self.model.encode(
                lang,
                convert_to_tensor=True,
                device=self.device,
                normalize_embeddings=True,  
                show_progress_bar=False
            )
            lang_emb = lang_emb.detach()
        
        if isinstance(lang, str):
            lang_emb = lang_emb.unsqueeze(0)  
            lang_emb = lang_emb[0]           
        
        return lang_emb