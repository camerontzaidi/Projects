import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
file_path = '/content/car_data2.csv'

# Define a preprocessing pipeline
numeric_features = [
    'subscription_length', 'vehicle_age', 'customer_age', 'region_density',
    'airbags', 'displacement', 'cylinder', 'turning_radius', 'length', 'width',
    'gross_weight', 'ncap_rating'
]
categorical_features = [
    'region_code', 'segment', 'model', 'fuel_type', 'max_torque', 'max_power',
    'engine_type', 'is_esc', 'is_adjustable_steering', 'is_tpms', 'is_parking_sensors',
    'is_parking_camera', 'rear_brakes_type', 'transmission_type', 'steering_type',
    'is_front_fog_lights', 'is_rear_window_wiper', 'is_rear_window_washer',
    'is_rear_window_defogger', 'is_brake_assist', 'is_power_door_locks',
    'is_central_locking', 'is_power_steering', 'is_driver_seat_height_adjustable',
    'is_day_night_rear_view_mirror', 'is_ecw', 'is_speed_alert'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Load the entire dataset to preprocess and perform the grid search
data = pd.read_csv(file_path)
data = data.drop(columns=['policy_id'])

# Split the data into features and target
features = data.drop(columns=['claim_status'])
labels = data['claim_status']

# Preprocess the features
features = preprocessor.fit_transform(features)
input_shape = features.shape[1]

# Define the model creation function
def create_model(params):
    model = Sequential([
        Dense(int(params['neurons']), activation=params['activation'], input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(params['dropout_rate']),
        Dense(int(params['neurons']//2), activation=params['activation']),
        BatchNormalization(),
        Dropout(params['dropout_rate']),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=params['optimizer'], loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define the objective function for Hyperopt
def objective(params):
    model = create_model(params)
    history = model.fit(features, labels, epochs=int(params['epochs']), batch_size=int(params['batch_size']), validation_split=0.2, verbose=0)
    val_predictions = model.predict(features)
    roc_auc = roc_auc_score(labels, val_predictions)
    return {'loss': -roc_auc, 'status': STATUS_OK, 'model': model}

# Define the hyperparameter space
space = {
    'neurons': hp.quniform('neurons', 64, 256, 1),
    'activation': hp.choice('activation', ['relu', 'tanh']),
    'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),
    'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),
    'batch_size': hp.quniform('batch_size', 32, 128, 1),
    'epochs': hp.quniform('epochs', 10, 50, 1)
}

# Run the hyperparameter optimization
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=20, trials=trials)

# Get the best model from the trials
best_model = None
for trial in trials.trials:
    if trial['result']['status'] == STATUS_OK:
        if best_model is None or trial['result']['loss'] < best_model['result']['loss']:
            best_model = trial

# Predict probabilities using the best model
best_keras_model = best_model['result']['model']
predictions = best_keras_model.predict(features)

# Calculate ROC-AUC score
roc_auc = roc_auc_score(labels, predictions)
print(f"Best Hyperparameters: {best}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(labels, predictions)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

# Save the best model
best_keras_model.save('best_car_claim_model.h5')
