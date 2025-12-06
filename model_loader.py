import joblib
import os
import pandas as pd

class ModelLoader:
    def __init__(self, base_path):
        self.base_path = base_path
        self.models = {}
        self.scalers = {}
        self.features_lists = {}
        
    def load_models(self):
        # Paths
        c_path = os.path.join(self.base_path, 'Data', 'data_c')
        java_path = os.path.join(self.base_path, 'Data', 'data_java', 'java_final')
        
        # Load C/C++ Model
        try:
            self.models['c++'] = joblib.load(os.path.join(c_path, 'best_bug_predictor_model.pkl'))
            self.scalers['c++'] = joblib.load(os.path.join(c_path, 'scaler.pkl'))
            # Try to load features list if exists, otherwise use default
            try:
                self.features_lists['c++'] = joblib.load(os.path.join(c_path, 'features_list.pkl'))
            except:
                print("Warning: features_list.pkl not found for C++, using default columns.")
        except Exception as e:
            print(f"Error loading C++ model: {e}")

        # Load Java Model
        try:
            self.models['java'] = joblib.load(os.path.join(java_path, 'best_java_bug_predictor.pkl'))
            # Check for scaler names, I saw 'scaler_java.pkl' and 'scaler_java_CORRECT.pkl'
            if os.path.exists(os.path.join(java_path, 'scaler_java_CORRECT.pkl')):
                self.scalers['java'] = joblib.load(os.path.join(java_path, 'scaler_java_CORRECT.pkl'))
            else:
                self.scalers['java'] = joblib.load(os.path.join(java_path, 'scaler_java.pkl'))
                
            try:
                self.features_lists['java'] = joblib.load(os.path.join(java_path, 'features_java.pkl'))
            except:
                print("Warning: features_java.pkl not found for Java, using default columns.")
        except Exception as e:
            print(f"Error loading Java model: {e}")

    def predict(self, features_df, language):
        if language not in self.models:
            raise ValueError(f"Model for {language} not loaded.")
            
        model = self.models[language]
        scaler = self.scalers.get(language)
        
        # Ensure features match what the model expects
        if language in self.features_lists:
            expected_features = self.features_lists[language]
            # Reorder/Filter columns
            features_df = features_df[expected_features]
        
        # Scale if scaler exists
        if scaler:
            X_scaled = scaler.transform(features_df)
            X_final = pd.DataFrame(X_scaled, columns=features_df.columns)
        else:
            X_final = features_df
            
        # Predict
        prediction = model.predict(X_final)[0]
        probability = model.predict_proba(X_final)[0][1] if hasattr(model, 'predict_proba') else 0.0
        
        return prediction, probability
