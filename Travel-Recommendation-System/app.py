import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page configurations with a wide and responsive layout
st.set_page_config(
    page_title="🌍 Travel Recommender",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium glassmorphism styling, hover effects, and modern aesthetics
st.markdown("""
<style>
    /* Font styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main container background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
        color: #f8fafc;
    }
    
    /* Center and style the title section */
    .main-title {
        text-align: center;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        padding-top: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Glassmorphism sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Radio and widget card styling */
    div[data-testid="stRadio"] > label {
        color: #38bdf8 !important;
        font-weight: 600 !important;
    }
    
    .stSelectbox label, .stTextInput label {
        color: #818cf8 !important;
        font-weight: 600 !important;
    }
    
    /* Styled container cards */
    .dest-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .dest-card:hover {
        transform: translateY(-5px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
        background: rgba(30, 41, 59, 0.7);
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 20px;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
    }
    
    .badge-beach { background-color: rgba(14, 116, 144, 0.3); color: #22d3ee; border: 1px solid #22d3ee; }
    .badge-culture { background-color: rgba(109, 40, 217, 0.3); color: #c084fc; border: 1px solid #c084fc; }
    .badge-adventure { background-color: rgba(194, 65, 12, 0.3); color: #fb923c; border: 1px solid #fb923c; }
    .badge-nature { background-color: rgba(21, 128, 61, 0.3); color: #4ade80; border: 1px solid #4ade80; }
    
    /* Activity tag styling */
    .activity-tag {
        display: inline-block;
        background-color: rgba(255, 255, 255, 0.05);
        color: #cbd5e1;
        font-size: 0.8rem;
        padding: 0.15rem 0.5rem;
        margin: 0.2rem;
        border-radius: 4px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Button custom styling */
    div.stButton > button {
        background: linear-gradient(90deg, #0ea5e9, #6366f1) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.5rem 2rem !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        width: 100%;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(99, 102, 241, 0.4) !important;
        background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<div class='main-title'>🌴 Travel Recommendation System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Find your next dream destination using machine learning suggestions</div>", unsafe_allow_html=True)

# ----------------- 2. Load the Dataset -----------------
csv_path = "travel_data_with_activities_final.csv"

@st.cache_data
def load_data():
    if not os.path.exists(csv_path):
        # Fallback to create a mock dataset if file doesn't exist (safety check)
        data = {
            "Destination": ["Bali", "Paris", "Queenstown", "Kyoto", "Cape Town", "Maldives", "Barcelona", "Reykjavik", "Phuket", "Petra", "Goa", "Jaipur", "Ladakh", "Kerala", "Rishikesh"],
            "Country": ["Indonesia", "France", "New Zealand", "Japan", "South Africa", "Maldives", "Spain", "Iceland", "Thailand", "Jordan", "India", "India", "India", "India", "India"],
            "Type": ["Beach", "Culture", "Adventure", "Culture", "Nature", "Beach", "Culture", "Nature", "Beach", "Culture", "Beach", "Culture", "Adventure", "Nature", "Adventure"],
            "Description": [
                "Known for its stunning beaches, vibrant nightlife, and temples.",
                "The city of lights, known for its art, cuisine, and the Eiffel Tower.",
                "Adventure capital with bungee jumping, skiing, and lake activities.",
                "Historic city with traditional temples, gardens, and tea ceremonies.",
                "Features beaches, mountains, and the iconic Table Mountain.",
                "Tropical paradise with turquoise waters and coral reefs.",
                "Famous for Gaudí architecture, beaches, and vibrant street life.",
                "Capital city known for its natural beauty, geysers, and culture.",
                "Thai island known for beaches, diving spots, and nightlife.",
                "Archaeological site famous for rock-cut architecture and tombs.",
                "Known for its stunning beaches, vibrant nightlife, and Portuguese heritage.",
                "Famous for its palaces, forts, and vibrant Rajasthani culture.",
                "High-altitude desert known for its mountain passes, monasteries, and trekking routes.",
                "Known for its backwaters, lush greenery, Ayurvedic treatments, and houseboats.",
                "A spiritual town on the Ganges River, known for yoga, rafting, and Himalayan views."
            ],
            "Activities": [
                "Surfing, Beach hopping, Temple tours",
                "Museum visits, Eiffel Tower tour, River Seine cruise",
                "Bungee jumping, Skiing, Jet boating",
                "Temple visits, Tea ceremonies, Kimono experience",
                "Hiking Table Mountain, Wine tasting, Beach days",
                "Snorkeling, Diving, Luxury spa",
                "Gaudi architecture tour, Tapas tasting, Beach lounging",
                "Northern lights tours, Hot springs, Whale watching",
                "Island hopping, Night markets, Scuba diving",
                "Historical site tours, Camel rides, Hiking",
                "Beach parties, Water sports, Fort visits",
                "Palace tours, Traditional shopping, Camel rides",
                "Trekking, Monastery visits, Biking",
                "Backwater cruises, Ayurvedic spa, Cultural shows",
                "Yoga sessions, River rafting, Cliff jumping"
            ]
        }
        df_mock = pd.DataFrame(data)
        df_mock["combined_features"] = df_mock["Type"] + " " + df_mock["Description"] + " " + df_mock["Activities"]
        return df_mock
    
    return pd.read_csv(csv_path)

df = load_data()

# ----------------- 3. Vectorize the Text -----------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(tfidf_matrix)

# ----------------- 4. Define Recommendation Functions -----------------

def recommend_by_destination(destination, top_n=3):
    if destination not in df["Destination"].values:
        return []
    idx = df[df["Destination"] == destination].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    # Sort by similarity score in descending order, skip the queried destination itself
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]
    return [df.iloc[i[0]].to_dict() for i in scores]

def recommend_by_activity(activity, top_n=3):
    # Case-insensitive filtering of activities
    matches = df[df["Activities"].str.contains(activity, case=False, na=False)]
    return matches.head(top_n).to_dict(orient="records")

# Helper function to get correct badge class based on type
def get_badge_class(dest_type):
    t = dest_type.lower()
    if "beach" in t:
        return "badge-beach"
    elif "culture" in t:
        return "badge-culture"
    elif "adventure" in t:
        return "badge-adventure"
    else:
        return "badge-nature"

# Helper function to render a destination card
def render_destination_card(res):
    # Find matching image file with supported extensions (.jpg, .jpeg, .png, .webp)
    img_path = None
    extensions = ['.jpg', '.jpeg', '.png', '.webp']
    for ext in extensions:
        test_path = f"images/{res['Destination']}{ext}"
        if os.path.exists(test_path):
            img_path = test_path
            break
            
    badge_class = get_badge_class(res['Type'])
    activities_list = [act.strip() for act in res['Activities'].split(',')]
    
    # Render card
    with st.container():
        st.markdown(f"""
        <div class="dest-card">
            <div>
                <span class="badge {badge_class}">{res['Type']}</span>
                <h3 style="margin: 0.2rem 0; color: #f8fafc; font-size: 1.4rem;">{res['Destination']}</h3>
                <p style="color: #38bdf8; font-size: 0.9rem; margin-bottom: 0.75rem; font-weight: 500;">📍 {res['Country']}</p>
                <p style="color: #cbd5e1; font-size: 0.95rem; margin-bottom: 1rem; line-height: 1.5;">{res['Description']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display image inside card if found, otherwise show a nice stylized banner
        if img_path:
            st.image(img_path, use_column_width=True)
        else:
            # Styled fallback colored bar
            st.markdown("""
            <div style="height: 150px; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 8px; display: flex; align-items: center; justify-content: center; border: 1px dashed rgba(255,255,255,0.1);">
                <span style="color: #64748b; font-size: 0.9rem;">🌅 No image available</span>
            </div>
            """, unsafe_allow_html=True)
            
        # Display Activities
        st.markdown("<p style='margin-top: 0.5rem; margin-bottom: 0.2rem; font-size: 0.8rem; color: #94a3b8; font-weight: 600;'>ACTIVITIES:</p>", unsafe_allow_html=True)
        activities_html = "".join([f"<span class='activity-tag'>{act}</span>" for act in activities_list])
        st.markdown(f"<div>{activities_html}</div>", unsafe_allow_html=True)
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# ----------------- Sidebar -----------------
with st.sidebar:
    st.markdown("### 🗺️ Exploring the World")
    st.write("This engine uses **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Cosine Similarity** to compare travel features and give recommendations.")
    
    st.markdown("---")
    st.markdown("### 📊 Database Insights")
    st.metric("Total Destinations", len(df))
    st.metric("Countries Represented", len(df["Country"].unique()))
    
    st.markdown("---")
    st.markdown("### 💡 Quick Guide")
    st.markdown("""
    - **By Destination**: Select a city you've liked in the past, and we will find cities with similar descriptions, climates, and activities.
    - **By Activity**: Type an activity like *diving*, *hiking*, or *spa* to search through active programs.
    """)
    st.info("💡 Powered by machine learning & content-based filtering.")

# ----------------- 5. Create Radio Button for User Choice -----------------
# We use container and columns to keep the user controls neatly arranged and clean
control_card = st.container()
with control_card:
    option = st.radio(
        "Choose how you want to get recommendations:", 
        ["By Destination", "By Activity"]
    )
    
st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 2rem 0;' />", unsafe_allow_html=True)

# ----------------- 6. Handle Selections -----------------
if option == "By Destination":
    st.subheader("🔍 Find Similar Places")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        place = st.selectbox("Select a destination you like:", df["Destination"].unique())
    with col2:
        st.write(" ") # alignment spacer
        st.write(" ") 
        btn_clicked = st.button("Recommend Similar Places")
        
    if btn_clicked:
        results = recommend_by_destination(place)
        if results:
            st.success(f"Here are top 3 places similar to **{place}**:")
            cols = st.columns(3)
            for idx, res in enumerate(results):
                with cols[idx]:
                    render_destination_card(res)
        else:
            st.error("No recommendations found.")

else:
    st.subheader("🔍 Find by Activities")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        activity = st.text_input("Enter an activity (e.g., hiking, diving, shopping, yoga, beach)")
    with col2:
        st.write(" ") # alignment spacer
        st.write(" ") 
        btn_clicked = st.button("Find Cities")
        
    if btn_clicked:
        if activity.strip() == "":
            st.warning("Please enter an activity first!")
        else:
            results = recommend_by_activity(activity)
            if results:
                st.success(f"Top places offering **'{activity}'**:")
                cols = st.columns(len(results))
                for idx, res in enumerate(results):
                    with cols[idx]:
                        render_destination_card(res)
            else:
                st.info(f"No destinations found with the activity '{activity}'. Try another one like 'hiking', 'temple', or 'beach'.")
