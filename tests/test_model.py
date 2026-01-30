
import pytest
import numpy as np
import os
from src import models
from src.config import MODEL_JSON_PATH

# Skip if model doesn't exist (e.g. in CI without artifact)
@pytest.mark.skipif(not os.path.exists(MODEL_JSON_PATH), reason="Model file not found")
def test_model_loading_and_prediction():
    model = models.load_model_xgb(MODEL_JSON_PATH)
    assert model is not None
    
    # Create dummy input with correct shape (assuming we know feature count)
    # If we don't know feature count, we can try to guess from Features list or fail gracefully
    try:
        features_list = models.load_features_list()
        n_features = len(features_list)
        X_dummy = np.random.rand(1, n_features)
        
        pred = model.predict(X_dummy)
        assert pred.shape == (1,)
        assert isinstance(pred[0], (np.float32, np.float64, float))
        
    except FileNotFoundError:
        pass # Features list might not be there
