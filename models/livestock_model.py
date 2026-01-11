
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Any, Tuple, List, Union
import warnings
warnings.filterwarnings('ignore')

from models.base_animal_model import BaseAnimalModel
from config.animal_configs import AnimalType, AnimalModelConfig, AnimalConfigManager
from config.disease_mappings import DiseaseSymptomMapper

class LivestockDiseaseModel(BaseAnimalModel):
    """Livestock disease prediction model for Kenyan context"""
    
    def __init__(self, config: AnimalModelConfig):
        super().__init__(AnimalType.LIVESTOCK, config)
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.species_encoder = LabelEncoder()
        self.model_version = "1.1.0-livestock"
        
    def validate_input_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate livestock data"""
        errors = []
        
        # Check required columns
        required_cols = self.config.required_features
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Validate data ranges
        validation_rules = self.config.validation_rules
        
        if 'age_months' in data.columns:
            if data['age_months'].min() < validation_rules.get('min_age', 1):
                errors.append(f"Age too low. Minimum: {validation_rules.get('min_age', 1)} months")
            if data['age_months'].max() > validation_rules.get('max_age', 240):
                errors.append(f"Age too high. Maximum: {validation_rules.get('max_age', 240)} months")
        
        if 'body_temperature' in data.columns:
            min_temp, max_temp = validation_rules.get('valid_temperatures', (35, 42))
            invalid_temps = data[
                (data['body_temperature'] < min_temp) | 
                (data['body_temperature'] > max_temp)
            ]
            if not invalid_temps.empty:
                errors.append(f"Body temperature outside valid range ({min_temp}-{max_temp}°C)")
        
        if 'county' in data.columns:
            valid_counties = validation_rules.get('required_counties', [])
            invalid_counties = data[~data['county'].isin(valid_counties)]['county'].unique()
            if len(invalid_counties) > 0:
                errors.append(f"Invalid counties: {list(invalid_counties)}. Valid: {valid_counties}")
        
        # Validate animal types
        if 'animal_type' in data.columns:
            valid_types = ['dairy_cattle', 'beef_cattle', 'goats', 'sheep']
            invalid_types = data[~data['animal_type'].isin(valid_types)]['animal_type'].unique()
            if len(invalid_types) > 0:
                errors.append(f"Invalid animal types: {list(invalid_types)}. Valid: {valid_types}")
        
        return len(errors) == 0, errors
    
    def preprocess_data(self, data: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Preprocess livestock data"""
        df = data.copy()
        
        # Store feature columns
        self.feature_columns = self._get_feature_columns(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Encode animal species
        if 'animal_type' in df.columns:
            if is_training:
                df['animal_type_encoded'] = self.species_encoder.fit_transform(df['animal_type'])
                self.feature_encoders['animal_type'] = self.species_encoder
            else:
                # Handle unseen species during prediction
                unseen_mask = ~df['animal_type'].isin(self.species_encoder.classes_)
                if unseen_mask.any():
                    df.loc[unseen_mask, 'animal_type'] = self.species_encoder.classes_[0]
                df['animal_type_encoded'] = self.species_encoder.transform(df['animal_type'])
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols 
                          if col != self.config.target_column and col != 'animal_type']
        
        for col in categorical_cols:
            if col in df.columns:
                if is_training:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.feature_encoders[col] = le
                elif col in self.feature_encoders:
                    le = self.feature_encoders[col]
                    # Handle unseen categories
                    unseen_mask = ~df[col].isin(le.classes_)
                    if unseen_mask.any():
                        df.loc[unseen_mask, col] = le.classes_[0]
                    df[col] = le.transform(df[col].astype(str))
        
        # Scale numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols 
                         if col != self.config.target_column and 'encoded' not in col]
        
        if len(numerical_cols) > 0:
            if is_training:
                df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            else:
                df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in livestock data"""
        # For dairy cattle, milk production is important
        if 'milk_production' in df.columns and 'animal_type' in df.columns:
            # Set milk production to 0 for non-dairy animals if missing
            non_dairy_mask = df['animal_type'].isin(['beef_cattle', 'goats', 'sheep'])
            df.loc[non_dairy_mask & df['milk_production'].isna(), 'milk_production'] = 0
        
        # Numerical columns: fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Categorical columns: fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, float]:
        """Train livestock disease model"""
        # Preprocess data
        X_processed = self.preprocess_data(X, is_training=True)
        
        # Encode target variable
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Get model parameters
        model_params = self.config.model_hyperparams.copy()
        model_type = model_params.pop('model_type', 'random_forest')
        
        # Initialize and train model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        train_accuracy = self.model.score(X_train, y_train)
        val_accuracy = self.model.score(X_val, y_val)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Feature importance
        self.feature_importance = self._calculate_feature_importance(X_processed)
        
        # Get classification report
        y_val_pred = self.model.predict(X_val)
        report = classification_report(y_val, y_val_pred, output_dict=True)
        
        # Store metrics
        self.metrics = {
            'train_accuracy': float(train_accuracy),
            'validation_accuracy': float(val_accuracy),
            'cross_val_mean': float(cv_scores.mean()),
            'cross_val_std': float(cv_scores.std()),
            'feature_importance': self.feature_importance,
            'classification_report': report,
            'n_samples': len(X),
            'n_features': X_processed.shape[1],
            'animal_types': list(self.species_encoder.classes_) if hasattr(self.species_encoder, 'classes_') else []
        }
        
        self.trained_at = datetime.now().isoformat()
        
        print(f"✓ Livestock model trained successfully")
        print(f"  Validation Accuracy: {val_accuracy:.2%}")
        print(f"  Cross-Validation: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
        print(f"  Diseases learned: {len(self.target_encoder.classes_)}")
        print(f"  Animal types: {self.metrics['animal_types']}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame, return_confidence: bool = False) -> Union[np.ndarray, Tuple]:
        """Predict diseases for livestock"""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Preprocess input data
        X_processed = self.preprocess_data(X, is_training=False)
        
        # Ensure all required columns are present
        expected_cols = self.feature_columns
        
        # Add missing columns with default values
        for col in expected_cols:
            if col not in X_processed.columns:
                X_processed[col] = 0
        
        # Reorder columns to match training
        X_processed = X_processed[expected_cols]
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        if return_confidence:
            probabilities = self.model.predict_proba(X_processed)
            return predictions, probabilities
        
        return predictions
    
    def predict_with_recommendations(self, X: pd.DataFrame) -> List[Dict]:
        """Make predictions with Kenyan-specific recommendations"""
        predictions_with_details = self.predict_with_details(X)
        
        for i, result in enumerate(predictions_with_details):
            disease = result['predicted_disease']
            
            # Extract symptoms from input data
            symptoms = self._extract_symptoms(X.iloc[i] if i < len(X) else X.iloc[0])
            
            # Get disease information
            disease_info = AnimalConfigManager.get_disease_info(
                AnimalType.LIVESTOCK, disease
            )
            
            if disease_info:
                result['disease_info'] = {
                    'common_name': disease_info.common_name,
                    'key_symptoms': disease_info.key_symptoms,
                    'high_risk_seasons': disease_info.high_risk_seasons,
                    'species_affected': disease_info.species_affected
                }
                
                result['recommendations'] = {
                    'immediate_actions': self._get_immediate_actions(disease, symptoms),
                    'treatment_guidelines': disease_info.treatment_guidelines,
                    'prevention_measures': self._get_prevention_measures(disease),
                    'estimated_cost_kes': disease_info.treatment_guidelines.get('cost_kes', '500-2000'),
                    'reporting_requirements': self._get_reporting_requirements(disease)
                }
            else:
                result['recommendations'] = {
                    'immediate_actions': 'Consult veterinarian immediately',
                    'general_advice': 'Isolate animal, monitor symptoms, keep records',
                    'contact': 'County Veterinary Office or 0800722001 (Kenya Vet Helpline)'
                }
            
            # Add risk assessment
            result['risk_assessment'] = self._assess_risk(disease, X.iloc[i] if i < len(X) else X.iloc[0])
            
            # Add economic impact
            result['economic_impact'] = self._assess_economic_impact(disease, X.iloc[i] if i < len(X) else X.iloc[0])
        
        return predictions_with_details
    
    def _extract_symptoms(self, sample: pd.Series) -> List[str]:
        """Extract symptoms from sample data"""
        symptoms = []
        
        # Map features to symptoms
        symptom_mapping = {
            'body_temperature': lambda x: 'fever' if x > 39.5 else None,
            'milk_production': lambda x: 'reduced_milk' if x < 10 else None,
            'feed_intake': lambda x: 'loss_of_appetite' if x in ['reduced', 'very_low'] else None
        }
        
        for feature, mapper in symptom_mapping.items():
            if feature in sample:
                symptom = mapper(sample[feature])
                if symptom:
                    symptoms.append(symptom)
        
        return symptoms
    
    def _get_immediate_actions(self, disease: str, symptoms: List[str]) -> List[str]:
        """Get immediate actions for disease"""
        actions = {
            'east_coast_fever': [
                "CONTACT VET WITHIN 48 HOURS for Buparvaquone injection",
                "Isolate sick animal immediately",
                "Spray animal and housing with acaricides",
                "Provide clean water and shade",
                "Monitor temperature twice daily"
            ],
            'foot_and_mouth': [
                "ISOLATE ANIMAL IMMEDIATELY - Highly contagious!",
                "Report to County Veterinary Officer within 24 hours",
                "Disinfect all equipment, vehicles, and premises",
                "Restrict movement of animals and people",
                "Set up foot baths with disinfectant"
            ],
            'mastitis': [
                "Milk out affected quarter completely every 2-3 hours",
                "Apply teat dip after every milking",
                "Improve milking hygiene practices",
                "Consult vet for appropriate antibiotic treatment",
                "Record milk yield and appearance"
            ],
            'lumpy_skin_disease': [
                "Isolate affected animals",
                "Provide soft, palatable feed",
                "Treat secondary infections with antibiotics",
                "Control flies and other vectors",
                "Consider vaccination if available"
            ]
        }
        
        default_actions = [
            "Consult local veterinarian",
            "Monitor temperature twice daily",
            "Provide clean water and good quality feed",
            "Keep records of symptoms and treatments",
            "Isolate from healthy animals"
        ]
        
        return actions.get(disease, default_actions)
    
    def _get_prevention_measures(self, disease: str) -> List[str]:
        """Get prevention measures for disease"""
        measures = {
            'east_coast_fever': [
                "Regular tick control (spraying/dipping every 7-14 days)",
                "Avoid grazing in tick-infested areas during rainy seasons",
                "Vaccinate if available in your area (Consult KALRO)",
                "Maintain good body condition through proper nutrition",
                "Use tick-repellant ear tags"
            ],
            'foot_and_mouth': [
                "Vaccinate every 6 months (consult vet for schedule)",
                "Maintain strict farm biosecurity",
                "Quarantine new animals for 21 days",
                "Control movement of people, vehicles, and equipment",
                "Avoid sharing equipment with other farms"
            ],
            'mastitis': [
                "Practice proper udder washing before milking (clean water, single towels)",
                "Use post-milking teat dip after every milking",
                "Treat all dry cows with dry cow therapy",
                "Maintain clean, dry bedding",
                "Regularly maintain and check milking machines"
            ],
            'lumpy_skin_disease': [
                "Vaccinate if available (live attenuated vaccine)",
                "Control insect populations (flies, mosquitoes)",
                "Isolate new animals for 28 days",
                "Provide good nutrition to boost immunity",
                "Regular health monitoring"
            ]
        }
        
        return measures.get(disease, [
            "Regular vaccination program",
            "Good nutrition and clean water",
            "Proper housing and ventilation",
            "Regular deworming program",
            "Biosecurity measures"
        ])
    
    def _get_reporting_requirements(self, disease: str) -> Dict[str, str]:
        """Get reporting requirements for disease"""
        reportable_diseases = {
            'foot_and_mouth': {
                'agency': 'County Veterinary Office & National Government',
                'timeline': 'Within 24 hours',
                'contact': 'Director of Veterinary Services - 0202712801',
                'penalty': 'Fine up to KES 500,000 or imprisonment'
            },
            'lumpy_skin_disease': {
                'agency': 'County Veterinary Office',
                'timeline': 'Within 48 hours',
                'contact': 'County Director of Veterinary Services',
                'penalty': 'Movement restrictions, mandatory vaccination'
            }
        }
        
        return reportable_diseases.get(disease, {
            'agency': 'County Veterinary Office (optional)',
            'timeline': 'When convenient',
            'contact': 'Local veterinarian',
            'penalty': 'None'
        })
    
    def _assess_risk(self, disease: str, sample: pd.Series) -> Dict[str, Any]:
        """Assess risk level based on disease and context"""
        risk_levels = {
            'east_coast_fever': 'HIGH' if sample.get('county') in ['Nakuru', 'Uasin_Gishu'] else 'MEDIUM',
            'foot_and_mouth': 'CRITICAL',
            'mastitis': 'MEDIUM',
            'lumpy_skin_disease': 'HIGH' if sample.get('season') == 'dry_season' else 'MEDIUM'
        }
        
        risk_factors = []
        if disease == 'east_coast_fever' and sample.get('season') in ['long_rains', 'short_rains']:
            risk_factors.append("High rainfall season - increased tick activity")
        
        if disease == 'foot_and_mouth':
            risk_factors.append("Highly contagious - rapid spread likely")
            risk_factors.append("Economic impact severe - trade restrictions")
        
        return {
            'risk_level': risk_levels.get(disease, 'LOW'),
            'risk_factors': risk_factors,
            'containment_urgency': 'IMMEDIATE' if disease in ['foot_and_mouth'] else 'WITHIN 48 HOURS'
        }
    
    def _assess_economic_impact(self, disease: str, sample: pd.Series) -> Dict[str, Any]:
        """Assess economic impact of disease"""
        animal_type = sample.get('animal_type', 'dairy_cattle')
        
        # Economic impact estimates in KES
        impacts = {
            'east_coast_fever': {
                'treatment_cost': 1500,
                'milk_loss_per_day': 500,
                'recovery_time_days': 14,
                'mortality_rate': '10-30% if untreated'
            },
            'foot_and_mouth': {
                'treatment_cost': 2000,
                'milk_loss_per_day': 1000,
                'weight_loss': '20-30%',
                'trade_impact': 'Export ban for 3 months',
                'mortality_rate': 'Low in adults, high in young'
            },
            'mastitis': {
                'treatment_cost': 800,
                'milk_loss_per_day': 300,
                'chronic_cases': 'May require culling',
                'mortality_rate': 'Low with treatment'
            },
            'lumpy_skin_disease': {
                'treatment_cost': 1200,
                'milk_loss_per_day': 400,
                'hide_damage': 'Reduced value',
                'mortality_rate': '1-5%'
            }
        }
        
        impact = impacts.get(disease, {
            'treatment_cost': 1000,
            'recovery_time_days': 7,
            'mortality_rate': 'Variable'
        })
        
        # Add species-specific considerations
        if animal_type == 'dairy_cattle':
            impact['production_loss'] = 'Milk production drop 30-70%'
        elif animal_type == 'beef_cattle':
            impact['production_loss'] = 'Weight loss 10-25%'
        
        return impact