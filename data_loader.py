import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(path="moodmeal_survey_data_2000.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def get_binary_columns(df):
    return [c for c in df.columns if df[c].dropna().isin([0, 1]).all() and df[c].dtype in ['int64', 'float64', 'int32']]

def get_categorical_columns(df):
    return [c for c in df.columns if df[c].dtype == 'object' and c != 'respondent_id']

def encode_ordinal_income(val):
    mapping = {'Below 25K': 12500, '25K-50K': 37500, '50K-1L': 75000, '1L-2L': 150000, 'Above 2L': 250000}
    return mapping.get(val, 0)

def encode_ordinal_spend(val):
    mapping = {'Below 1K': 500, '1K-3K': 2000, '3K-6K': 4500, '6K-10K': 8000, 'Above 10K': 13000}
    return mapping.get(val, 0)

def encode_ordinal_wtp(val):
    mapping = {'Below 150': 100, '150-250': 200, '250-400': 325, '400-600': 500, 'Above 600': 750}
    return mapping.get(val, 0)

def encode_social_media(val):
    mapping = {'Less than 1 hour': 0.5, '1-2 hours': 1.5, '2-4 hours': 3, 'More than 4 hours': 5}
    return mapping.get(val, 1)

def encode_meals_out(val):
    mapping = {'0': 0, '1': 1, '2': 2, '3+': 3.5}
    return mapping.get(str(val), 0)

def encode_order_freq(val):
    mapping = {'Daily': 7, '3-5 times/week': 4, '1-2 times/week': 1.5, 'Few times/month': 0.5, 'Rarely/unsure': 0.1}
    return mapping.get(val, 0)

def encode_target(val):
    mapping = {'Definitely yes': 4, 'Probably yes': 3, 'Neutral': 2, 'Probably no': 1, 'Definitely no': 0}
    return mapping.get(val, 2)

def build_feature_matrix(df):
    df_feat = df.copy()
    df_feat['income_numeric'] = df_feat['monthly_income'].map(encode_ordinal_income)
    df_feat['spend_numeric'] = df_feat['monthly_food_spend'].map(encode_ordinal_spend)
    df_feat['wtp_numeric'] = df_feat['max_wtp_meal_bowl'].map(encode_ordinal_wtp)
    df_feat['social_media_numeric'] = df_feat['social_media_hours'].map(encode_social_media)
    df_feat['meals_out_numeric'] = df_feat['meals_ordered_per_day'].map(encode_meals_out)
    df_feat['order_freq_numeric'] = df_feat['expected_order_frequency'].map(encode_order_freq)
    df_feat['target_numeric'] = df_feat['likelihood_to_try_moodmeal'].map(encode_target)

    cat_cols_to_encode = ['age_group', 'gender', 'city_tier', 'occupation', 'dietary_preference',
                          'biggest_frustration', 'eco_packaging_willingness', 'subscription_interest',
                          'discovery_channel_1', 'ordering_channel_1', 'combo_movie_night',
                          'combo_wfh_lunch', 'combo_post_gym', 'differentiator_1']

    label_encoders = {}
    for col in cat_cols_to_encode:
        if col in df_feat.columns:
            le = LabelEncoder()
            df_feat[col + '_enc'] = le.fit_transform(df_feat[col].astype(str))
            label_encoders[col] = le

    return df_feat, label_encoders

def get_classification_features(df_feat):
    binary_cols = get_binary_columns(df_feat)
    numeric_cols = ['income_numeric', 'spend_numeric', 'wtp_numeric', 'social_media_numeric',
                    'meals_out_numeric', 'order_freq_numeric', 'mood_food_sensitivity', 'clean_label_importance']
    encoded_cols = [c for c in df_feat.columns if c.endswith('_enc')]
    feature_cols = list(set(binary_cols + numeric_cols + encoded_cols))
    feature_cols = [c for c in feature_cols if c in df_feat.columns and c != 'target_numeric']
    return feature_cols

def get_clustering_features(df_feat):
    binary_cols = get_binary_columns(df_feat)
    numeric_cols = ['income_numeric', 'spend_numeric', 'wtp_numeric', 'social_media_numeric',
                    'meals_out_numeric', 'order_freq_numeric', 'mood_food_sensitivity', 'clean_label_importance']
    feature_cols = list(set(binary_cols + numeric_cols))
    feature_cols = [c for c in feature_cols if c in df_feat.columns]
    return feature_cols

def get_regression_features(df_feat):
    binary_cols = get_binary_columns(df_feat)
    numeric_cols = ['income_numeric', 'wtp_numeric', 'social_media_numeric',
                    'meals_out_numeric', 'order_freq_numeric', 'mood_food_sensitivity', 'clean_label_importance']
    encoded_cols = [c for c in df_feat.columns if c.endswith('_enc')]
    feature_cols = list(set(binary_cols + numeric_cols + encoded_cols))
    feature_cols = [c for c in feature_cols if c in df_feat.columns and c not in ['target_numeric', 'spend_numeric']]
    return feature_cols
