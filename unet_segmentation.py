import mlflow
import datetime
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import numpy as np




# 1. Définition de la convention de nommage
def get_run_name(arch="UNet", opt="Adam", loss="Dice"):
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    return f"{arch}_{opt}_{loss}_{date}"

# Exemple d'initialisation MLflow
mlflow.set_experiment("Medical_Segmentation_TP4")

with mlflow.start_run(run_name=get_run_name(arch="Simplified_UNet", opt="Adam", loss="Binary_Dice")):
    # 2. Log des paramètres
    mlflow.log_param("architecture", "U-Net")
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("loss_function", "DiceLoss")
    
    # mlflow.log_metric("val_dice_coef", dice_value)


# EXERCICE 1 : ARCHITECTURE U-NET

def conv_block(input_tensor, num_filters):
    # Core convolutional block: Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU
    x = keras.layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    
    x = keras.layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    return x

def build_unet(input_shape=(128, 128, 1)):
    inputs = keras.Input(input_shape)

    # ENCODER PATH (Contracting )
    c1 = conv_block(inputs, 32)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 64)   # TODO complété : step 2 (64 filtres)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 128)  # TODO complété : step 3 (128 filtres)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)

    # BRIDGE / BOTTLENECK
    b = conv_block(p3, 256)

    # DECODER (Expansive Path)
    # Step 1
    u1 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b)
    u1 = keras.layers.Concatenate()([u1, c3]) # Skip connection
    d1 = conv_block(u1, 128)

    # Step 2 (TODO complété)
    u2 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(d1)
    u2 = keras.layers.Concatenate()([u2, c2]) # Skip connection
    d2 = conv_block(u2, 64)

    # Step 3 (TODO complété)
    u3 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(d2)
    u3 = keras.layers.Concatenate()([u3, c1]) # Skip connection
    d3 = conv_block(u3, 32)

    # OUTPUT LAYER
    outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(d3)

    return keras.Model(inputs=[inputs], outputs=[outputs])

# EXERCICE 2 : MÉTRIQUES SPÉCIFIQUES 

def dice_coeff(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_metric(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# Compilation du modèle
model = build_unet()
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=[dice_coeff, iou_metric])


# --- EXERCICE 3 : BLOC CONV3D ---

def simple_conv3d_block(input_shape=(32, 32, 32, 1)):
    # Bloc de démonstration : D x H x W x C
    inputs = keras.Input(input_shape)
    
    # Premier bloc (16 filtres)
    x = keras.layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = keras.layers.MaxPool3D((2, 2, 2))(x)
    
    # TODO : Ajout d'un second bloc Conv3D (32 filtres)
    x = keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPool3D((2, 2, 2))(x)
    
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x) # Sortie binaire
    
    return keras.Model(inputs, outputs)

# --- TRACKING MLFLOW (Engineering Practice) ---

if __name__ == "__main__":
    mlflow.set_experiment("3D_Volumetric_Analysis")
    
    with mlflow.start_run(run_name="Conv3D_Baseline"):
        # Initialisation du modèle
        model_3d = simple_conv3d_block()
        
        # 1. Log de l'architecture (Engineering Practice)
        model_config = model_3d.to_json()
        mlflow.log_dict({"model_config": model_config}, "artifacts/model_architecture.json")
        
        # 2. Log des Hyperparamètres
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("filters_start", 16)
        mlflow.log_param("input_depth", 32)
        
        # 3. Simulation de l'entraînement (TODO complété)
        # Imaginons une boucle d'entraînement ici
        final_val_loss = 0.45  # Valeur simulée
        final_val_accuracy = 0.82
        
        mlflow.log_metric("final_val_loss", final_val_loss)
        mlflow.log_metric("final_val_accuracy", final_val_accuracy)
        
        print("MLflow tracking complete for 3D block experiment.")
        

# Simulation d'un résultat pour l'exemple
test_img = np.random.rand(128, 128, 1)
pred_mask = model.predict(test_img[np.newaxis, ...])[0]

# Création de la figure
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(test_img[:,:,0], cmap='gray')
ax[0].set_title("Image Médicale (Input)")
ax[1].imshow(pred_mask[:,:,0], cmap='jet')
ax[1].set_title("Segmentation (Output)")

# Enregistrement dans MLflow
mlflow.log_figure(fig, "predictions/sample_segmentation.png")
plt.close(fig) # Important pour libérer la mémoire