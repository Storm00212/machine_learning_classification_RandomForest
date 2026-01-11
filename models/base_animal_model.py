
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
import joblib
from datetime import datetime
import json
from pathlib import Path

from config.animal_configs import AnimalType, AnimalModelConfig

class BaseAnimalModel(ABC):
    
    
    def __init__(self, animal_type: AnimalType, config: AnimalModelConfig):
        self.animal_type = animal_type
        self.config = config
        self.model = None
        self.preprocessor = None
        self.feature_encoders = {}
        self.target_encoder = None
        self.metrics = {}
        self.trained_at = None
        self.model_version = "1.0.0"
        self.feature_columns = []
        
    @abstractmethod
    def validate_input_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate input data against schema"""
        pass
    
    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Preprocess data according to animal type"""
        pass
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, float]:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame, return_confidence: bool = False) -> Union[np.ndarray, Tuple]:
        """Make predictions"""
        pass
    
    def predict_with_details(self, X: pd.DataFrame) -> List[Dict]:
        """Make predictions with detailed information"""
        predictions, probabilities = self.predict(X, return_confidence=True)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            # Decode prediction
            if self.target_encoder:
                pred_decoded = self.target_encoder.inverse_transform([pred])[0]
            else:
                pred_decoded = str(pred)
            
            # Get top diseases
            top_indices = probs.argsort()[-3:][::-1]
            top_diseases = []
            if self.target_encoder:
                disease_names = self.target_encoder.inverse_transform(top_indices)
                for idx, disease in zip(top_indices, disease_names):
                    top_diseases.append({
                        "disease": disease,
                        "probability": float(probs[idx]),
                        "confidence": "high" if probs[idx] > 0.7 else "medium" if probs[idx] > 0.4 else "low"
                    })
            
            results.append({
                "animal_id": f"{self.animal_type.value}_{i}",
                "predicted_disease": pred_decoded,
                "confidence_score": float(probs.max()),
                "top_disease_probabilities": top_diseases,
                "prediction_timestamp": datetime.now().isoformat()
            })
        
        return results
    
    def save_model(self, filepath: str):
        """Save model and all components"""
        model_data = {
            'model': self.model,
            'feature_encoders': self.feature_encoders,
            'target_encoder': self.target_encoder,
            'animal_type': self.animal_type.value,
            'config': self.config,
            'metrics': self.metrics,
            'trained_at': self.trained_at,
            'model_version': self.model_version,
            'saved_at': datetime.now().isoformat(),
            'feature_columns': self.feature_columns,
            'model_type': self.__class__.__name__
        }
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")
        print(f"  Animal Type: {self.animal_type.value}")
        print(f"  Version: {self.model_version}")
        print(f"  Features: {len(self.feature_columns)}")
        print(f"  Accuracy: {self.metrics.get('validation_accuracy', 0):.2%}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load saved model"""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        print(f"✓ Model loaded from {filepath}")
        print(f"  Type: {model_data['animal_type']}")
        print(f"  Model Class: {model_data.get('model_type', 'Unknown')}")
        print(f"  Trained: {model_data['trained_at']}")
        print(f"  Version: {model_data.get('model_version', '1.0.0')}")
        print(f"  Features: {len(model_data.get('feature_columns', []))}")
        return model_data
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        diseases = []
        if self.target_encoder and hasattr(self.target_encoder, 'classes_'):
            diseases = list(self.target_encoder.classes_)
        
        return {
            'animal_type': self.animal_type.value,
            'model_version': self.model_version,
            'trained_at': self.trained_at,
            'metrics': self.metrics,
            'features_used': self.feature_columns,
            'diseases_covered': diseases,
            'feature_count': len(self.feature_columns),
            'model_class': self.__class__.__name__
        }
    
    def _calculate_feature_importance(self, X: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance if model supports it"""
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(X.columns, self.model.feature_importances_))
            # Sort by importance
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        return {}
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract feature columns from data"""
        # Remove target column if present
        columns = [col for col in df.columns if col != self.config.target_column]
        
        # Add encoded columns if they exist
        encoded_cols = [f"{col}_encoded" for col in self.feature_encoders.keys()]
        all_cols = columns + [col for col in encoded_cols if col in df.columns]
        
        return list(set(all_cols))  # Remove duplicates