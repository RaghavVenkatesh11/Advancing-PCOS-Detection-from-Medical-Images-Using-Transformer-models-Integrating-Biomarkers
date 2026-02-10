import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from catboost import CatBoostClassifier

# --- Configuration ---
# Match these filenames exactly to what you have in your local 'models' folder
TABULAR_MODEL_PATH = 'models/catboost_pcos_tabular.cbm'
FEATURE_NAMES_PATH = 'models/feature_names.pkl'
IMAGE_MODEL_PATH = 'models/swin_pcos_image.pth' 

# --- Weight Configuration (Late Fusion) ---
# 70% Clinical Data (Biomarkers), 30% Ultrasound Image
WEIGHT_TABULAR = 0.70
WEIGHT_IMAGE = 0.30

# Page Setup
st.set_page_config(
    page_title="PCOS Multimodal Diagnostic System", 
    layout="wide", 
    page_icon="🧬"
)

# --- 1. Load Resources (Cached) ---
@st.cache_resource
def load_tabular_resources():
    """Loads the CatBoost model and feature names."""
    if not os.path.exists(FEATURE_NAMES_PATH) or not os.path.exists(TABULAR_MODEL_PATH):
        return None, None
    
    try:
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        
        # Initialize and Load CatBoost Model
        model = CatBoostClassifier()
        model.load_model(TABULAR_MODEL_PATH)
        
        return model, feature_names
    except Exception as e:
        st.error(f"Error loading tabular model: {e}")
        return None, None

@st.cache_resource
def load_image_model():
    """Loads the Swin Transformer (Tiny) model."""
    if not os.path.exists(IMAGE_MODEL_PATH):
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 1. Initialize Swin-T Architecture (No pre-trained weights needed here, we load ours)
        model = models.swin_t(weights=None)
        
        # 2. Recreate the Head EXACTLY as defined in training
        # Training used: nn.Sequential(Dropout(0.3), Linear(num_ftrs, 2))
        num_ftrs = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 2)
        )
        
        # 3. Load the saved weights
        state_dict = torch.load(IMAGE_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval() # Important: Sets Dropout to 0 for inference
        return model
        
    except Exception as e:
        st.error(f"Error loading Swin model: {e}")
        return None

# --- 2. Preprocessing Functions ---
def preprocess_image(image):
    """Resizes and normalizes image for Swin Transformer."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Standard ImageNet normalization required for Swin
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) # Add batch dimension

# --- 3. Main Interface ---
st.title("🧬 PCOS Multimodal Diagnostic System")
st.markdown("### Late Fusion Analysis (CatBoost + Swin Transformer)")
st.info("This system fuses data from **Clinical Biomarkers** (70% weight) and **Ultrasound Imaging** (30% weight) using state-of-the-art architectures.")

# Load Models
tab_model, feature_names = load_tabular_resources()
img_model = load_image_model()

# Check for model files
if not tab_model or not img_model:
    st.error(f"⚠️ Model files missing! Ensure the following exist in a 'models/' folder:\n- {TABULAR_MODEL_PATH}\n- {FEATURE_NAMES_PATH}\n- {IMAGE_MODEL_PATH}")
    st.stop()

# --- INPUT FORM ---
with st.form("fusion_form"):
    col1, col2 = st.columns([1, 1], gap="large")
    
    # === Left Column: Clinical Data (The Golden 8) ===
    with col1:
        st.subheader("1. Clinical Profile")
        
        # Group 1: Physical
        c1, c2 = st.columns(2)
        with c1:
            val_age = st.number_input("Age (yrs)", min_value=10, max_value=60, value=26)
            val_weight = st.number_input("Weight (Kg)", min_value=30.0, max_value=150.0, value=60.0)
        with c2:
            val_bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=24.0, help="Body Mass Index")
            val_whr = st.number_input("Waist:Hip Ratio", min_value=0.5, max_value=1.5, value=0.85, step=0.01)

        # Group 2: Cycle & Hormones
        val_cycle = st.number_input("Cycle Length (days)", min_value=0, max_value=90, value=30, help="Avg days between periods")
        
        st.divider()
        st.markdown("**Hormonal Markers** (Crucial)")
        
        h1, h2, h3 = st.columns(3)
        with h1:
            val_fsh = st.number_input("FSH (mIU/mL)", value=5.0)
        with h2:
            val_lh = st.number_input("LH (mIU/mL)", value=6.0)
        with h3:
            val_amh = st.number_input("AMH (ng/mL)", value=4.5, help="Anti-Müllerian Hormone. >4.0 is high risk.")

    # === Right Column: Ultrasound Image ===
    with col2:
        st.subheader("2. Ultrasound Imaging")
        uploaded_file = st.file_uploader("Upload Transvaginal Scan", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Scan', width=350)
        else:
            st.info("Awaiting image upload...")

    # === Submit Button ===
    st.markdown("---")
    submit_btn = st.form_submit_button("🔍 Run Fusion Analysis", type="primary")

# --- 4. Prediction Logic ---
if submit_btn:
    if uploaded_file is None:
        st.warning("⚠️ Please upload an ultrasound image to complete the multimodal analysis.")
    else:
        # ---------------------------------------------------------
        # STEP A: Tabular Prediction (CatBoost)
        # ---------------------------------------------------------
        
        # 1. Store user inputs in a Clean Dictionary
        input_dict = {
            'Age (yrs)': val_age,
            'Weight (Kg)': val_weight,
            'BMI': val_bmi,
            'Cycle length(days)': val_cycle,
            'Waist:Hip Ratio': val_whr,
            'FSH(mIU/mL)': val_fsh,
            'LH(mIU/mL)': val_lh,
            'AMH(ng/mL)': val_amh
        }
        
        # 2. Match inputs to Trained Feature Names (Handling spaces)
        try:
            ordered_input = []
            for feature in feature_names:
                # Strip spaces from the saved feature name to match our clean keys
                clean_feature = feature.strip() 
                
                if clean_feature in input_dict:
                    ordered_input.append(input_dict[clean_feature])
                else:
                    st.error(f"Logic Error: Model expects feature '{feature}', but app input_dict has '{clean_feature}'. Check spelling.")
                    st.stop()
            
            # Create DataFrame
            df_input = pd.DataFrame([ordered_input], columns=feature_names)
            
            # --- CATBOOST PREDICTION ---
            # predict_proba returns [Prob_Class0, Prob_Class1]
            prob_array = tab_model.predict_proba(df_input)
            prob_tabular = prob_array[0][1] # Probability of PCOS (Class 1)
            
        except Exception as e:
            st.error(f"Input Processing Error: {e}")
            st.stop()

        # ---------------------------------------------------------
        # STEP B: Image Prediction (Swin Transformer)
        # ---------------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = preprocess_image(image).to(device)
        
        with torch.no_grad():
            outputs = img_model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # MAPPING LOGIC (Based on Standard ImageFolder alphabetical sorting)
            # Index 0 = 'infected' (PCOS)
            # Index 1 = 'noninfected' (Healthy)
            prob_image = probs[0][0].item()

        # ---------------------------------------------------------
        # STEP C: Late Fusion Calculation
        # ---------------------------------------------------------
        final_score = (prob_tabular * WEIGHT_TABULAR) + (prob_image * WEIGHT_IMAGE)
        
        # ---------------------------------------------------------
        # STEP D: Visualization & Results
        # ---------------------------------------------------------
        st.divider()
        st.header("Diagnostic Results")
        
        # Use columns for layout
        res1, res2, res3 = st.columns(3)
        
        with res1:
            st.metric("Biomarker Risk", f"{prob_tabular*100:.1f}%", help="Based on 8 Clinical Features (CatBoost)")
            if prob_tabular > 0.5:
                st.caption("⚠️ Elevated Biomarkers")
            else:
                st.caption("✅ Normal Range")
            
        with res2:
            st.metric("Ultrasound Risk", f"{prob_image*100:.1f}%", help="Swin Transformer Analysis")
            if prob_image > 0.5:
                st.caption("⚠️ Cystic Patterns Detected")
            else:
                st.caption("✅ Clear Ovaries")
            
        with res3:
            st.metric("🛡️ FINAL FUSION SCORE", f"{final_score*100:.1f}%", delta_color="inverse")

        # Final Interpretation
        st.markdown("### Interpretation")
        if final_score > 0.5:
            st.error(f"**High Probability of PCOS ({final_score*100:.2f}%)**")
            st.write("The multimodal analysis indicates a strong likelihood of PCOS. The patient shows elevated risk factors in clinical biomarkers and/or cystic patterns in ultrasound imaging.")
        else:
            st.success(f"**Low Probability of PCOS ({final_score*100:.2f}%)**")
            st.write("The multimodal analysis suggests the patient is likely healthy. Clinical biomarkers and ultrasound imaging do not show strong convergence toward a PCOS diagnosis.")

        # Optional: Technical Details Expander
        with st.expander("See Calculation Details"):
            st.latex(r'''
            Score_{final} = (P_{clinical} \times 0.70) + (P_{Swin} \times 0.30)
            ''')
            st.write(f"**Calculation:** ({prob_tabular:.3f} × 0.70) + ({prob_image:.3f} × 0.30) = **{final_score:.3f}**")
            
            if val_lh > 0:
                st.write(f"**LH/FSH Ratio:** {val_lh/val_fsh:.2f} (Clinical Indicator)")

st.sidebar.info("Disclaimer: This tool is a decision support system and should not replace professional medical diagnosis.")