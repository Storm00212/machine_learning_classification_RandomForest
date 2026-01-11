
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from config.animal_configs import AnimalType, AnimalConfigManager
from models.model_registry import ModelRegistry
from preprocessing.data_validator import AnimalDataValidator
from utils.data_loader import DataLoader
from utils.prediction_explainer import PredictionExplainer

class AnimalDiseasePredictor:
    """Main orchestrator for animal disease prediction"""
    
    def __init__(self, models_dir: str = "trained_models"):
        self.registry = ModelRegistry(models_dir)
        self.validator = AnimalDataValidator()
        self.data_loader = DataLoader()
        
    def train_model(self, animal_type_str: str, data_file: str, 
                   model_name: str = None, test_size: float = 0.2) -> dict:
        """Train a new model"""
        try:
            # Convert string to AnimalType enum
            animal_type = AnimalType(animal_type_str.lower())
        except ValueError:
            return {
                'success': False,
                'error': f"Invalid animal type: {animal_type_str}. Use 'livestock' or 'poultry'"
            }
        
        try:
            # Clean and validate data
            data = self.data_loader.load_data(data_file)
            data = self.validator.clean_data(data, animal_type_str)
            
            if animal_type == AnimalType.LIVESTOCK:
                is_valid, errors = self.validator.validate_livestock_data(data)
            else:
                is_valid, errors = self.validator.validate_poultry_data(data)
            
            if not is_valid:
                return {
                    'success': False,
                    'error': 'Data validation failed',
                    'validation_errors': errors
                }
            
            # Train and save model
            result = self.registry.train_and_save(
                animal_type, data_file, model_name, test_size
            )
            
            result['success'] = True
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'animal_type': animal_type_str
            }
    
    def predict(self, animal_type_str: str, input_data: dict, 
               model_path: str = None, explain: bool = True) -> dict:
        """Make a prediction"""
        try:
            animal_type = AnimalType(animal_type_str.lower())
        except ValueError:
            return {
                'success': False,
                'error': f"Invalid animal type: {animal_type_str}. Use 'livestock' or 'poultry'"
            }
        
        try:
            result = self.registry.predict(animal_type, input_data, model_path)
            
            if result.get('error'):
                return result
            
            # Add explanation if requested
            if explain and result.get('predictions'):
                prediction = result['predictions'][0]  # Single prediction
                
                if animal_type == AnimalType.LIVESTOCK:
                    explanation = PredictionExplainer.explain_livestock_prediction(
                        prediction, input_data
                    )
                else:
                    explanation = PredictionExplainer.explain_poultry_prediction(
                        prediction, input_data
                    )
                
                result['explanation'] = explanation
                
                # Generate text report
                result['text_report'] = PredictionExplainer.generate_text_report(
                    prediction, input_data, animal_type_str
                )
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'animal_type': animal_type_str
            }
    
    def batch_predict(self, animal_type_str: str, data_file: str,
                     model_path: str = None) -> dict:
        """Make batch predictions from file"""
        try:
            animal_type = AnimalType(animal_type_str.lower())
        except ValueError:
            return {
                'success': False,
                'error': f"Invalid animal type: {animal_type_str}"
            }
        
        try:
            result = self.registry.batch_predict(animal_type, data_file, model_path)
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'animal_type': animal_type_str
            }
    
    def validate_data(self, animal_type_str: str, data_file: str) -> dict:
        """Validate data file"""
        try:
            animal_type = AnimalType(animal_type_str.lower())
        except ValueError:
            return {
                'success': False,
                'error': f"Invalid animal type: {animal_type_str}"
            }
        
        try:
            data = self.data_loader.load_data(data_file)
            data = self.validator.clean_data(data, animal_type_str)
            
            if animal_type == AnimalType.LIVESTOCK:
                is_valid, errors = self.validator.validate_livestock_data(data)
            else:
                is_valid, errors = self.validator.validate_poultry_data(data)
            
            # Check data quality
            quality_report = self.validator.check_data_quality(data, animal_type_str)
            
            return {
                'success': True,
                'is_valid': is_valid,
                'validation_errors': errors,
                'quality_report': quality_report,
                'data_summary': {
                    'row_count': len(data),
                    'column_count': len(data.columns),
                    'columns': list(data.columns),
                    'sample_data': data.head(3).to_dict('records')
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_disease_info(self, animal_type_str: str, disease_name: str = None) -> dict:
        """Get disease information"""
        try:
            animal_type = AnimalType(animal_type_str.lower())
        except ValueError:
            return {
                'success': False,
                'error': f"Invalid animal type: {animal_type_str}"
            }
        
        config_manager = AnimalConfigManager()
        
        if disease_name:
            disease_info = config_manager.get_disease_info(animal_type, disease_name)
            if disease_info:
                return {
                    'success': True,
                    'disease': disease_name,
                    'info': disease_info.__dict__,
                    'animal_type': animal_type_str
                }
            else:
                return {
                    'success': False,
                    'error': f"Disease '{disease_name}' not found for {animal_type_str}"
                }
        else:
            diseases = config_manager.get_all_diseases(animal_type)
            species_mapping = config_manager.get_species_mapping(animal_type)
            county_zones = config_manager.get_county_zones()
            seasonal_risks = config_manager.get_seasonal_risks()
            
            return {
                'success': True,
                'animal_type': animal_type_str,
                'diseases': diseases,
                'disease_count': len(diseases),
                'species_mapping': species_mapping,
                'county_zones': county_zones,
                'seasonal_risks': seasonal_risks
            }
    
    def get_model_info(self, animal_type_str: str = None, 
                      model_path: str = None) -> dict:
        """Get information about trained models"""
        try:
            if animal_type_str:
                animal_type = AnimalType(animal_type_str.lower())
                
                # Get specific model information
                try:
                    model = self.registry.load_model(animal_type, model_path)
                    return {
                        'success': True,
                        'animal_type': animal_type_str,
                        'model_info': model.get_model_info(),
                        'available_at': datetime.now().isoformat()
                    }
                except Exception as e:
                    return {
                        'success': False,
                        'error': f"Could not load model: {str(e)}"
                    }
            else:
                # Get all available models
                available_models = self.registry.get_available_models()
                
                return {
                    'success': True,
                    'available_models': available_models,
                    'model_registry_path': str(self.registry.models_dir),
                    'total_models': sum(len(models) for models in available_models.values()),
                    'available_at': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_system_info(self) -> dict:
        """Get system information"""
        available_models = self.registry.get_available_models()
        total_models = sum(len(models) for models in available_models.values())
        
        return {
            'success': True,
            'system_name': 'Kenyan Animal Disease Prediction System',
            'version': '1.0.0',
            'description': 'Machine learning system for predicting livestock and poultry diseases in Kenya',
            'supported_animal_types': ['livestock', 'poultry'],
            'available_models': available_models,
            'total_trained_models': total_models,
            'model_registry_path': str(self.registry.models_dir),
            'timestamp': datetime.now().isoformat(),
            'capabilities': [
                'Disease prediction for livestock and poultry',
                'Kenyan context-aware recommendations',
                'Batch prediction from CSV/JSON files',
                'Model training from historical data',
                'Comprehensive data validation',
                'Prediction explanations and reports'
            ]
        }
    
    def generate_sample_data(self, animal_type_str: str, n_samples: int = 100, 
                            save_path: str = None) -> dict:
        """Generate sample data for testing"""
        try:
            animal_type = AnimalType(animal_type_str.lower())
        except ValueError:
            return {
                'success': False,
                'error': f"Invalid animal type: {animal_type_str}"
            }
        
        try:
            sample_data = self.data_loader.load_sample_data(animal_type_str, n_samples)
            
            if save_path:
                self.data_loader.save_data(sample_data, save_path)
                return {
                    'success': True,
                    'animal_type': animal_type_str,
                    'n_samples': n_samples,
                    'saved_to': save_path,
                    'sample_preview': sample_data.head(5).to_dict('records')
                }
            else:
                return {
                    'success': True,
                    'animal_type': animal_type_str,
                    'n_samples': n_samples,
                    'data': sample_data.to_dict('records'),
                    'columns': list(sample_data.columns)
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Animal Disease Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a livestock model
  python main.py --mode train --animal-type livestock --data-file farm_data.csv
  
  # Make a prediction
  python main.py --mode predict --animal-type poultry --input '{"poultry_type": "layers", ...}'
  
  # Batch prediction from file
  python main.py --mode batch --animal-type livestock --data-file new_cases.csv
  
  # Validate data file
  python main.py --mode validate --animal-type poultry --data-file poultry_data.csv
  
  # Get disease information
  python main.py --mode diseases --animal-type livestock --disease east_coast_fever
  
  # Get system information
  python main.py --mode info
  
  # Generate sample data
  python main.py --mode sample --animal-type poultry --n-samples 50 --save-path sample_poultry.csv
        """
    )
    
    parser.add_argument('--mode', choices=['train', 'predict', 'batch', 'validate', 
                                          'diseases', 'info', 'models', 'sample'],
                       required=True, help='Operation mode')
    parser.add_argument('--animal-type', choices=['livestock', 'poultry'], 
                       help='Type of animal (required for most modes)')
    parser.add_argument('--data-file', help='Path to data file (CSV/JSON/Excel)')
    parser.add_argument('--input', help='Input data as JSON string for prediction')
    parser.add_argument('--model-name', help='Name for saved model')
    parser.add_argument('--model-path', help='Path to specific model file')
    parser.add_argument('--disease', help='Disease name for information lookup')
    parser.add_argument('--n-samples', type=int, default=100, 
                       help='Number of samples to generate (for sample mode)')
    parser.add_argument('--save-path', help='Path to save output')
    parser.add_argument('--explain', action='store_true', default=True,
                       help='Generate explanation for predictions (default: True)')
    parser.add_argument('--no-explain', action='store_false', dest='explain',
                       help='Do not generate explanation for predictions')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test size ratio for training (default: 0.2)')
    
    args = parser.parse_args()
    predictor = AnimalDiseasePredictor()
    
    # Execute based on mode
    if args.mode == 'train':
        if not args.animal_type or not args.data_file:
            print("Error: --animal-type and --data-file required for training")
            return
        
        result = predictor.train_model(
            args.animal_type, 
            args.data_file, 
            args.model_name,
            args.test_size
        )
    
    elif args.mode == 'predict':
        if not args.animal_type:
            print("Error: --animal-type required for prediction")
            return
        
        if not args.input:
            print("Error: --input required for prediction")
            return
        
        try:
            input_data = json.loads(args.input)
        except json.JSONDecodeError:
            print("Error: Invalid JSON input")
            return
        
        result = predictor.predict(
            args.animal_type, 
            input_data, 
            args.model_path,
            args.explain
        )
    
    elif args.mode == 'batch':
        if not args.animal_type or not args.data_file:
            print("Error: --animal-type and --data-file required for batch prediction")
            return
        
        result = predictor.batch_predict(
            args.animal_type, 
            args.data_file, 
            args.model_path
        )
    
    elif args.mode == 'validate':
        if not args.animal_type or not args.data_file:
            print("Error: --animal-type and --data-file required for validation")
            return
        
        result = predictor.validate_data(args.animal_type, args.data_file)
    
    elif args.mode == 'diseases':
        if not args.animal_type:
            print("Error: --animal-type required for disease information")
            return
        
        result = predictor.get_disease_info(args.animal_type, args.disease)
    
    elif args.mode == 'models':
        result = predictor.get_model_info(args.animal_type, args.model_path)
    
    elif args.mode == 'info':
        result = predictor.get_system_info()
    
    elif args.mode == 'sample':
        if not args.animal_type:
            print("Error: --animal-type required for sample generation")
            return
        
        result = predictor.generate_sample_data(
            args.animal_type, 
            args.n_samples, 
            args.save_path
        )
    
    # Output result
    if args.save_path:
        with open(args.save_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {args.save_path}")
    else:
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()