import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

def konversi_ke_menit(teks):
    try:
        waktu_str = str(teks).split()[-1]
        jam, menit, detik = waktu_str.split(":")
        total_menit = (int(jam) * 60) + int(menit)
        return total_menit
    except Exception:
        return 0

def safe_eval(x):
    if isinstance(x, str) and x.startswith("["):
        try:
            return ast.literal_eval(x)
        except Exception:
            return [x]
    return [x]

def clean_and_preprocess(input_path, output_path=None):
    df = pd.read_csv(input_path)
    
    fitur_target = ["genres", "synopsis", "type", "episodes", "producers", "licensors", "studios", "source", "episode_duration", "score", "status"]
    clean_df = df[fitur_target].copy()
    
    clean_df = clean_df.dropna()
    
    kolom_bermasalah = ["producers", "licensors", "studios", "genres"]
    for kol in kolom_bermasalah:
        clean_df[kol] = clean_df[kol].replace("[]", "Unknown")
        
    clean_df["duration_minutes"] = clean_df["episode_duration"].apply(konversi_ke_menit)
    clean_df["duration_minutes"] = clean_df["duration_minutes"].astype(int)
    
    for kol in kolom_bermasalah:
        clean_df[kol] = clean_df[kol].apply(safe_eval)
        
    genres_mlb = MultiLabelBinarizer()
    genres_df = pd.DataFrame(genres_mlb.fit_transform(clean_df["genres"]), columns=genres_mlb.classes_)
    
    producers_mlb = MultiLabelBinarizer()
    producers_df = pd.DataFrame(producers_mlb.fit_transform(clean_df["producers"]), columns=producers_mlb.classes_)
    kolom_producers_lolos = producers_df.sum().nlargest(50).index
    producers_df_clean = producers_df[kolom_producers_lolos]
    
    licensors_mlb = MultiLabelBinarizer()
    licensors_df = pd.DataFrame(licensors_mlb.fit_transform(clean_df["licensors"]), columns=licensors_mlb.classes_)
    kolom_licensors_lolos = licensors_df.sum().nlargest(50).index
    licensors_df_clean = licensors_df[kolom_licensors_lolos]
    
    studios_mlb = MultiLabelBinarizer()
    studios_df = pd.DataFrame(studios_mlb.fit_transform(clean_df["studios"]), columns=studios_mlb.classes_)
    kolom_studios_lolos = studios_df.sum().nlargest(50).index
    studios_df_clean = studios_df[kolom_studios_lolos]
    
    patterns = ["Written by MAL Rewrite", "MAL Rewrite", "(Source: MAL News)", "(Source: MAL Rewrite)", 
                "(Source: AniDB)", "(Source: ANN)", "(Source: Wikipedia)", "(Source: Crunchyroll)", 
                "mal rewrite", "written mal", "rewrite"]
    for p in patterns:
        clean_df["synopsis"] = clean_df["synopsis"].str.replace(p, "", regex=False)
        
    tfidf = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1,2), max_df=0.8, min_df=5)
    synopsis_encoded = tfidf.fit_transform(clean_df["synopsis"].fillna(""))
    synopsis_df = pd.DataFrame(synopsis_encoded.toarray(), columns=tfidf.get_feature_names_out())
    
    type_source_df = pd.get_dummies(clean_df[["type", "source", "status"]], dtype=int).reset_index(drop=True)
    
    clean_df = clean_df.reset_index(drop=True)
    fitur_numerik = clean_df[["episodes", "duration_minutes"]]
    
    final_df = pd.concat([
        fitur_numerik,
        clean_df["score"],
        type_source_df,
        studios_df_clean.reset_index(drop=True),
        licensors_df_clean.reset_index(drop=True),
        producers_df_clean.reset_index(drop=True),
        genres_df.reset_index(drop=True),
        synopsis_df.reset_index(drop=True)
    ], axis=1)
    
    if output_path:
        final_df.to_csv(output_path, index=False)
        
    return final_df

if __name__ == "__main__":
    clean_and_preprocess("anime.csv", "anime_preprocessing.csv")