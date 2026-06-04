import pandas as pd
import numpy as np
import ast
import joblib
import requests

# 1. Memuat objek encoder yang sudah disimpan dari tahap training
print("--> Memuat objek encoder joblib...")
tfidf = joblib.load("./preprocessing/tfidf.pkl")
genres_mlb = joblib.load("./preprocessing/genres_mlb.pkl")
producers_mlb = joblib.load("./preprocessing/producers_mlb.pkl")
licensors_mlb = joblib.load("./preprocessing/licensors_mlb.pkl")
studios_mlb = joblib.load("./preprocessing/studios_mlb.pkl")
kolom_ohe = joblib.load("./preprocessing/kolom_ohe.pkl")

def konversi_ke_menit(teks):
    try:
        waktu_str = str(teks).split()[-1]
        jam, menit, detik = waktu_str.split(":")
        return (int(jam) * 60) + int(menit)
    except Exception:
        return 0

def clean_synopsis(text):
    patterns = [
        "Written by MAL Rewrite", "MAL Rewrite", "(Source: MAL News)", 
        "(Source: MAL Rewrite)", "(Source: AniDB)", "(Source: ANN)", 
        "(Source: Wikipedia)", "(Source: Crunchyroll)", "mal rewrite", 
        "written mal", "rewrite"
    ]
    text = str(text)
    for p in patterns:
        text = text.replace(p, "")
    return text

def parse_list(x):
    if isinstance(x, str) and x.startswith("["):
        try:
            return ast.literal_eval(x)
        except Exception:
            return [x]
    elif isinstance(x, list):
        return x
    return [x]

def predict_anime_score(raw_input):
    # Mengubah dictionary input menjadi DataFrame 1 baris
    df_input = pd.DataFrame([raw_input])
    
    # TAHAP 1: Preprocessing Fitur Numerik
    df_input["duration_minutes"] = df_input["episode_duration"].apply(konversi_ke_menit).astype(int)
    fitur_numerik = df_input[["episodes", "duration_minutes"]].reset_index(drop=True)
    
    # TAHAP 2: Transformasi One-Hot Encoding (type, source, status) via Reindex
    type_source_df = pd.get_dummies(df_input[["type", "source", "status"]], dtype=int)
    type_source_df = type_source_df.reindex(columns=kolom_ohe, fill_value=0).reset_index(drop=True)
    
    # TAHAP 3: Transformasi MultiLabelBinarizer (genres, studios, licensors, producers)
    df_input["genres"] = df_input["genres"].apply(parse_list)
    genres_df = pd.DataFrame(genres_mlb.transform(df_input["genres"]), columns=genres_mlb.classes_).reset_index(drop=True)
    
    df_input["studios"] = df_input["studios"].apply(parse_list)
    studios_df = pd.DataFrame(studios_mlb.transform(df_input["studios"]), columns=studios_mlb.classes_).reset_index(drop=True)
    
    df_input["licensors"] = df_input["licensors"].apply(parse_list)
    licensors_df = pd.DataFrame(licensors_mlb.transform(df_input["licensors"]), columns=licensors_mlb.classes_).reset_index(drop=True)
    
    df_input["producers"] = df_input["producers"].apply(parse_list)
    producers_df = pd.DataFrame(producers_mlb.transform(df_input["producers"]), columns=producers_mlb.classes_).reset_index(drop=True)
    
    # TAHAP 4: Transformasi TF-IDF Synopsis
    df_input["synopsis"] = df_input["synopsis"].apply(clean_synopsis)
    synopsis_encoded = tfidf.transform(df_input["synopsis"].fillna(""))
    synopsis_df = pd.DataFrame(synopsis_encoded.toarray(), columns=tfidf.get_feature_names_out()).reset_index(drop=True)
    
    # TAHAP 5: Penggabungan Semua Fitur Sementara
    final_df = pd.concat([
        fitur_numerik,
        type_source_df,
        studios_df,
        licensors_df,
        producers_df,
        genres_df,
        synopsis_df
    ], axis=1)
    
    # Membuang nama kolom yang duplikat agar reindex tidak crash
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    
    # TAHAP 6: Skema Alignment (Penyelarasan Kolom dengan data Training)
    df_blueprint = pd.read_csv("./anime_preprocessing.csv").drop(columns=["score"])
    kolom_final_model = df_blueprint.columns
    final_df = final_df.reindex(columns=kolom_final_model, fill_value=0)
    
    # TAHAP 7: Membungkus ke Format JSON MLflow & Kirim ke API Flask
    payload = {
        "dataframe_records": final_df.to_dict(orient="records"),
    }
    
    api_url = "http://127.0.0.1:8000/predict"
    try:
        response = requests.post(api_url, json=payload)
        return response.json()
    except Exception as e:
        return {"error": f"Gagal terhubung ke API Flask: {str(e)}"}

if __name__ == "__main__":
    # Data Baru yang belum pernah dilihat model
    anime_baru = {
        "genres": "['Action', 'Award Winning', 'Supernatural']",
        "synopsis": "Ever since the death of his father, the burden of supporting the family has fallen upon Tanjirou Kamado's shoulders. Though living impoverished on a remote mountain, the Kamado family are able to enjoy a relatively peaceful and happy life. One day, Tanjirou decides to go down to the local village to make a little money selling charcoal. On his way back, night falls, forcing Tanjirou to take shelter in the house of a strange man, who warns him of the existence of flesh-eating demons that lurk in the woods at night.\n\nWhen he finally arrives back home the next day, he is met with a horrifying sight—his whole family has been slaughtered. Worse still, the sole survivor is his sister Nezuko, who has been turned into a bloodthirsty demon. Consumed by rage and hatred, Tanjirou swears to avenge his family and stay by his only remaining sibling. Alongside the mysterious group calling themselves the Demon Slayer Corps, Tanjirou will do whatever it takes to slay the demons and protect the remnants of his beloved sister's humanity.",
        "type": "TV",
        "episodes": 26,
        "producers": "['Aniplex', 'Studio Mausu', 'Shueisha']",
        "licensors": "['Aniplex of America']",
        "studios": "['ufotable']",
        "source": "Manga",
        "episode_duration": "00:23:45",
        "status": "Finished Airing"
    }
    
    print("\n--> Mengirim data anime baru ke pipeline inference...")
    hasil = predict_anime_score(anime_baru)
    print("\n[HASIL PREDIKSI SKOR]:", hasil)