"""User Promotion Targeting Prediction Web Service

Flask API for predicting which users should receive promotions using a Random Forest model.
This service loads a pre-trained model and exposes REST endpoints for both single and batch predictions.

Author: MLOps Team
Version: 1.0
"""

import logging
import pickle
import traceback
from datetime import datetime
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from etl import UserGenerator
from feature_engineer import FeatureEngineer
from flask import Flask, jsonify, request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model
model = None
feature_columns = None


def load_model():
    """
    Load the Random Forest model at application startup.
    
    Returns:
        tuple: (model, feature_columns) or raises exception
    """
    global model, feature_columns
    try:
        with open('models/model_rf.pkl', 'rb') as f:
            logger.info('🔄 Loading Random Forest model...')
            model = pickle.load(f)
            
            # Get feature names from the model if available
            if hasattr(model, 'feature_names_in_'):
                feature_columns = list(model.feature_names_in_)
            else:
                # Default feature columns based on the training pipeline
                feature_columns = [
                    'age_group', 'location', 'device_type', 'subscription_type',
                    'days_since_registration', 'total_purchases', 'avg_order_value',
                    'last_purchase_days', 'sessions_last_30_days', 'time_on_site_minutes',
                    'pages_per_session', 'cart_abandonment_rate', 'purchase_frequency',
                    'total_purchases_per_day', 'days_between_first_and_last_purchase',
                    'bucket_avg_order_value'
                ]
            
            logger.info(f'✅ Model loaded successfully. Features: {len(feature_columns)}')
            return model, feature_columns
    except FileNotFoundError:
        logger.error('❌ Error: models/model_rf.pkl file not found')
        raise
    except Exception as e:
        logger.error(f'❌ Error loading model: {e}')
        raise


def prepare_single_user_data(user_data: Dict) -> pd.DataFrame:
    """
    Prepare a single user's data for prediction.
    
    Args:
        user_data (dict): Dictionary with user information
    
    Returns:
        pd.DataFrame: DataFrame with single user ready for feature engineering
    """
    # Create DataFrame from single user
    df = pd.DataFrame([user_data])
    
    # Apply feature engineering
    feature_engineer = FeatureEngineer(df)
    df = feature_engineer.create_features()
    
    logger.info(f"✅ Features prepared for user: {user_data.get('user_id', 'Unknown')}")
    return df


def prepare_batch_data(users_data: List[Dict]) -> pd.DataFrame:
    """
    Prepare batch of users' data for prediction.
    
    Args:
        users_data (list): List of dictionaries with user information
    
    Returns:
        pd.DataFrame: DataFrame with users ready for feature engineering
    """
    # Create DataFrame from batch
    df = pd.DataFrame(users_data)
    
    # Apply feature engineering
    feature_engineer = FeatureEngineer(df)
    df = feature_engineer.create_features()
    
    logger.info(f"✅ Features prepared for {len(df)} users")
    return df


def generate_synthetic_batch(n_samples: int = 100) -> pd.DataFrame:
    """
    Generate synthetic user data for batch prediction.
    
    Args:
        n_samples (int): Number of synthetic users to generate
    
    Returns:
        pd.DataFrame: DataFrame with synthetic users and features
    """
    logger.info(f"🎲 Generating {n_samples} synthetic users...")
    
    # Use the UserGenerator from etl.py
    user_generator = UserGenerator(n_samples=n_samples)
    df = user_generator.create_dataset()
    
    # Apply feature engineering
    feature_engineer = FeatureEngineer(df)
    df = feature_engineer.create_features()
    
    logger.info(f"✅ Generated and processed {len(df)} synthetic users")
    return df


def predict_promotion(df: pd.DataFrame) -> np.ndarray:
    """
    Perform promotion targeting prediction using the loaded model.
    
    Args:
        df (pd.DataFrame): Features prepared DataFrame
    
    Returns:
        np.ndarray: Prediction probabilities
    """
    # Select only the features used for training
    # Handle categorical columns properly
    categorical_columns = ['age_group', 'location', 'device_type', 'subscription_type', 'bucket_avg_order_value']
    
    # Convert categorical columns to appropriate format
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Make prediction
    predictions_proba = model.predict_proba(df)
    logger.info(f"🎯 Predictions made for {len(df)} users")
    return predictions_proba


def format_single_prediction(user_id: str, prediction_proba: np.ndarray) -> Dict:
    """
    Format single user prediction response.
    
    Args:
        user_id (str): User identifier
        prediction_proba (np.ndarray): Prediction probabilities
    
    Returns:
        dict: Formatted response
    """
    # Get probabilities
    prob_no_promotion = float(prediction_proba[0][0])
    prob_promotion = float(prediction_proba[0][1])
    
    # Determine recommendation
    should_receive_promotion = prob_promotion > 0.5
    
    return {
        'user_id': user_id,
        'should_receive_promotion': 'Si' if should_receive_promotion else 'No',
        'confidence': max(prob_no_promotion, prob_promotion),
        'probabilities': {
            'no_promotion': prob_no_promotion,
            'promotion': prob_promotion
        },
        'timestamp': datetime.now().isoformat()
    }


def format_batch_predictions(df: pd.DataFrame, predictions_proba: np.ndarray) -> List[Dict]:
    """
    Format batch predictions response.
    
    Args:
        df (pd.DataFrame): Original DataFrame with user data
        predictions_proba (np.ndarray): Prediction probabilities
    
    Returns:
        list: List of formatted predictions
    """
    results = []
    
    for idx, row in df.iterrows():
        prob_no_promotion = float(predictions_proba[idx][0])
        prob_promotion = float(predictions_proba[idx][1])
        should_receive_promotion = prob_promotion > 0.5
        
        result = {
            'user_id': row.get('user_id', f'USER-{idx:06d}'),
            'should_receive_promotion': 'Si' if should_receive_promotion else 'No',
            'confidence': max(prob_no_promotion, prob_promotion),
            'probabilities': {
                'no_promotion': prob_no_promotion,
                'promotion': prob_promotion
            }
        }
        results.append(result)
    
    return results


# Create Flask application
app = Flask('promotion-prediction')

# Load model on startup
try:
    load_model()
except Exception as e:
    logger.error(f"Failed to load model on startup: {e}")
    # Continue running but endpoints will return errors


@app.route('/predict', methods=['POST'])
def predict_single_endpoint():
    """
    REST endpoint for single user promotion prediction.
    
    Method: POST
    Content-Type: application/json
    
    Request Body:
        {
            "user_id": str,                    # User identifier
            "age_group": str,                   # Age group (18-25, 26-35, etc.)
            "location": str,                    # User location
            "device_type": str,                 # Device type (Mobile, Desktop, Tablet)
            "subscription_type": str,           # Subscription level
            "days_since_registration": int,     # Days since registration
            "total_purchases": int,             # Total number of purchases
            "avg_order_value": float,          # Average order value
            "last_purchase_days": int,         # Days since last purchase
            "sessions_last_30_days": int,      # Sessions in last 30 days
            "time_on_site_minutes": float,     # Average time on site
            "pages_per_session": float,        # Pages viewed per session
            "cart_abandonment_rate": float,    # Cart abandonment rate
            "purchase_frequency": float        # Purchase frequency
        }
    
    Response:
        {
            "user_id": str,
            "should_receive_promotion": str,   # "Si" or "No"
            "confidence": float,               # Confidence score
            "probabilities": {
                "no_promotion": float,
                "promotion": float
            },
            "timestamp": str
        }
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get JSON data from request
        user_data = request.get_json()
        
        if not user_data:
            logger.error("❌ Request without JSON data")
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = [
            'age_group', 'location', 'device_type', 'subscription_type',
            'days_since_registration', 'total_purchases', 'avg_order_value',
            'last_purchase_days', 'sessions_last_30_days', 'time_on_site_minutes',
            'pages_per_session', 'cart_abandonment_rate', 'purchase_frequency'
        ]
        
        missing_fields = [field for field in required_fields if field not in user_data]
        if missing_fields:
            logger.error(f"❌ Missing required fields: {missing_fields}")
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
        
        user_id = user_data.get('user_id', 'Unknown')
        logger.info(f"🚀 New prediction request for user: {user_id}")
        
        # Prepare data and predict
        df = prepare_single_user_data(user_data)
        predictions_proba = predict_promotion(df)
        
        # Format response
        result = format_single_prediction(user_id, predictions_proba)
        
        logger.info(f"✅ Prediction sent for user {user_id}: {result['should_receive_promotion']}")
        return jsonify(result)
        
    except KeyError as e:
        logger.error(f"❌ Missing field in request: {e}")
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"❌ Error in prediction: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch_endpoint():
    """
    REST endpoint for batch user promotion predictions.
    
    Method: POST
    Content-Type: application/json
    
    Request Body:
        {
            "users": [
                {
                    # Same fields as single prediction
                },
                ...
            ]
        }
    
    Response:
        {
            "predictions": [
                {
                    "user_id": str,
                    "should_receive_promotion": str,
                    "confidence": float,
                    "probabilities": {...}
                },
                ...
            ],
            "total_users": int,
            "users_to_target": int,
            "timestamp": str
        }
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'users' not in data:
            logger.error("❌ Request without users data")
            return jsonify({'error': 'No users data provided'}), 400
        
        users_data = data['users']
        
        if not isinstance(users_data, list) or len(users_data) == 0:
            logger.error("❌ Invalid users data format")
            return jsonify({'error': 'Users must be a non-empty list'}), 400
        
        logger.info(f"📦 New batch prediction request for {len(users_data)} users")
        
        # Prepare data and predict
        df = prepare_batch_data(users_data)
        predictions_proba = predict_promotion(df)
        
        # Format predictions
        predictions = format_batch_predictions(df, predictions_proba)
        
        # Calculate summary statistics
        users_to_target = sum(1 for p in predictions if p['should_receive_promotion'] == 'Si')
        
        result = {
            'predictions': predictions,
            'total_users': len(predictions),
            'users_to_target': users_to_target,
            'target_percentage': round((users_to_target / len(predictions)) * 100, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Batch predictions sent: {len(predictions)} users, {users_to_target} to target")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Error in batch prediction: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


@app.route('/predict_synthetic', methods=['POST'])
def predict_synthetic_endpoint():
    """
    REST endpoint for generating and predicting synthetic users.
    
    Method: POST
    Content-Type: application/json
    
    Request Body:
        {
            "n_samples": int,    # Number of synthetic users to generate (default: 100)
            "seed": int          # Random seed for reproducibility (optional)
        }
    
    Response:
        {
            "predictions": [...],
            "total_users": int,
            "users_to_target": int,
            "target_percentage": float,
            "generation_info": {
                "n_samples_requested": int,
                "n_samples_generated": int,
                "seed": int
            },
            "timestamp": str
        }
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get parameters
        data = request.get_json() or {}
        n_samples = data.get('n_samples', 100)
        seed = data.get('seed', 42)
        
        # Validate parameters
        if not isinstance(n_samples, int) or n_samples <= 0:
            return jsonify({'error': 'n_samples must be a positive integer'}), 400
        
        if n_samples > 10000:
            return jsonify({'error': 'n_samples cannot exceed 10000 for performance reasons'}), 400
        
        logger.info(f"🎲 Generating {n_samples} synthetic users with seed {seed}")
        
        # Generate synthetic data
        user_generator = UserGenerator(n_samples=n_samples, seed=seed)
        df = user_generator.create_dataset()
        
        # Apply feature engineering
        feature_engineer = FeatureEngineer(df)
        df = feature_engineer.create_features()
        
        # Make predictions
        predictions_proba = predict_promotion(df)
        
        # Format predictions
        predictions = format_batch_predictions(df, predictions_proba)
        
        # Calculate summary statistics
        users_to_target = sum(1 for p in predictions if p['should_receive_promotion'] == 'Si')
        
        result = {
            'predictions': predictions,
            'total_users': len(predictions),
            'users_to_target': users_to_target,
            'target_percentage': round((users_to_target / len(predictions)) * 100, 2),
            'generation_info': {
                'n_samples_requested': n_samples,
                'n_samples_generated': len(df),
                'seed': seed
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Synthetic predictions: {len(predictions)} users, {users_to_target} to target")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Error in synthetic prediction: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


@app.route('/export_predictions', methods=['POST'])
def export_predictions_endpoint():
    """
    REST endpoint for exporting predictions to CSV format.
    
    Method: POST
    Content-Type: application/json
    
    Request Body:
        {
            "n_samples": int,           # Number of synthetic users (default: 100000)
            "include_probabilities": bool,  # Include probability columns (default: false)
            "format": str              # Export format: "csv" (default: "csv")
        }
    
    Response:
        {
            "message": str,
            "file_saved": str,
            "total_records": int,
            "users_to_target": int,
            "timestamp": str
        }
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get parameters
        data = request.get_json() or {}
        n_samples = data.get('n_samples', 100000)
        include_probabilities = data.get('include_probabilities', False)
        export_format = data.get('format', 'csv')
        
        logger.info(f"📁 Exporting predictions for {n_samples} users")
        
        # Generate synthetic data
        user_generator = UserGenerator(n_samples=n_samples)
        df = user_generator.create_dataset()
        
        # Apply feature engineering
        feature_engineer = FeatureEngineer(df)
        df = feature_engineer.create_features()
        
        # Make predictions
        predictions_proba = predict_promotion(df)
        
        # Add predictions to dataframe
        if include_probabilities:
            df['prediction_no_promotion'] = predictions_proba[:, 0]
            df['prediction_promotion'] = predictions_proba[:, 1]
            df['should_receive_promotion'] = (predictions_proba[:, 1] > 0.5).astype(int)
            df['should_receive_promotion'] = df['should_receive_promotion'].map({0: "No", 1: "Si"})
            filename = 'predictions/predictions_proba.csv'
        else:
            df['should_receive_promotion'] = (predictions_proba[:, 1] > 0.5).astype(int)
            df['should_receive_promotion'] = df['should_receive_promotion'].map({0: "No", 1: "Si"})
            filename = 'predictions/predictions.csv'
        
        # Save to file
        df.to_csv(filename, index=False)
        
        # Calculate statistics
        users_to_target = (df['should_receive_promotion'] == 'Si').sum()
        
        result = {
            'message': 'Predictions exported successfully',
            'file_saved': filename,
            'total_records': len(df),
            'users_to_target': int(users_to_target),
            'target_percentage': round((users_to_target / len(df)) * 100, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Exported {len(df)} predictions to {filename}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Error exporting predictions: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify service status.
    
    Returns:
        JSON response with service status and model information
    """
    health_status = {
        'status': 'healthy' if model is not None else 'degraded',
        'model_loaded': model is not None,
        'model_type': type(model).__name__ if model else None,
        'service': 'User Promotion Targeting Prediction',
        'version': '1.0',
        'endpoints': {
            '/predict': 'Single user prediction',
            '/predict_batch': 'Batch user predictions',
            '/predict_synthetic': 'Generate and predict synthetic users',
            '/export_predictions': 'Export predictions to CSV',
            '/health': 'Health check',
            '/model_info': 'Model information'
        },
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(health_status)


@app.route('/model_info', methods=['GET'])
def model_info():
    """
    Get information about the loaded model.
    
    Returns:
        JSON response with model details
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        info = {
            'model_type': type(model).__name__,
            'model_params': model.get_params() if hasattr(model, 'get_params') else {},
            'n_features': len(feature_columns) if feature_columns else 'Unknown',
            'feature_names': feature_columns if feature_columns else [],
            'model_file': 'models/model_rf.pkl',
            'timestamp': datetime.now().isoformat()
        }
        
        # Add Random Forest specific information
        if hasattr(model, 'n_estimators'):
            info['n_estimators'] = model.n_estimators
        if hasattr(model, 'max_depth'):
            info['max_depth'] = model.max_depth
        if hasattr(model, 'feature_importances_'):
            # Get top 10 most important features
            importances = model.feature_importances_
            if feature_columns:
                feature_importance = sorted(
                    zip(feature_columns, importances),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                info['top_features'] = [
                    {'feature': feat, 'importance': float(imp)}
                    for feat, imp in feature_importance
                ]
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"❌ Error getting model info: {e}")
        return jsonify({'error': 'Error retrieving model information'}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist. Check /health for available endpoints.',
        'status': 404
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        'error': 'Method not allowed',
        'message': 'The HTTP method is not allowed for this endpoint.',
        'status': 405
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred. Please try again later.',
        'status': 500
    }), 500


if __name__ == "__main__":
    """
    Main entry point to run the Flask server.
    
    Configuration:
        - Debug: True (development only)
        - Host: 0.0.0.0 (accepts external connections)
        - Port: 9696
        
    Usage:
        python api.py
        
    Example requests:
        # Health check
        curl http://localhost:9696/health
        
        # Single prediction
        curl -X POST http://localhost:9696/predict \
             -H "Content-Type: application/json" \
             -d @user_data.json
        
        # Batch prediction
        curl -X POST http://localhost:9696/predict_batch \
             -H "Content-Type: application/json" \
             -d '{"users": [...]}'
        
        # Generate synthetic predictions
        curl -X POST http://localhost:9696/predict_synthetic \
             -H "Content-Type: application/json" \
             -d '{"n_samples": 1000}'
    """
    logger.info("🚀 Starting Flask server for User Promotion Prediction...")
    logger.info("📍 Server will be available at http://localhost:9696")
    logger.info("📚 Check http://localhost:9696/health for API documentation")
    
    app.run(debug=True, host='0.0.0.0', port=9696)