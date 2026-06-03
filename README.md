#  Destinex - Travel Recommendation System

##  Overview

Destinex is an AI & Machine Learning-based Travel Recommendation System that helps users discover travel destinations based on their interests and preferences.

The system uses Content-Based Filtering techniques to analyze destination features such as destination type, description, and activities, and then recommends similar travel locations. Users can also search for destinations based on activities like hiking, diving, shopping, cultural exploration, and more.

This project was developed as a Mini Project for the B.Tech Artificial Intelligence & Data Science program.

---

##  Features

* Recommend destinations based on a selected location
* Find destinations based on preferred activities
* Content-Based Recommendation System
* TF-IDF Vectorization for text processing
* Cosine Similarity for recommendation generation
* Interactive Streamlit Web Application
* Destination images and detailed descriptions
* Beginner-friendly implementation

---

##  Technologies Used

### Programming Language

* Python

### Frontend

* Streamlit

### Libraries

* Pandas
* NumPy
* Scikit-Learn

### Machine Learning Techniques

* TF-IDF Vectorization
* Cosine Similarity

### Development Tools

* VS Code

##  Project Structure

```text
Travel-Recommendation-System/
│
├── app.py
├── travel_recommendation.ipynb
├── travel_data_with_activities_final.csv
├── requirements.txt
├── README.md
│
├── images/
│   ├── Bali.jpg
│   ├── Paris.jpg
│   ├── Kyoto.jpg
│   └── ...

```

##  Dataset Information

The dataset contains information about popular travel destinations and includes:

* Destination Name
* Country
* Destination Type
* Description
* Activities
* Combined Features

The Combined Features column merges destination type, description, and activities into a single text field used by the recommendation engine.

##  Working of the Recommendation System

### Step 1: Data Preprocessing

Destination information is collected and combined into a single text feature.

### Step 2: TF-IDF Vectorization

The textual data is converted into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency).

### Step 3: Similarity Calculation

Cosine Similarity is applied to measure how similar destinations are to one another.

### Step 4: Recommendation Generation

The system returns the most relevant destinations based on similarity scores.

---

## How to Run the Project

### Clone the Repository

```bash
git clone https://github.com/your-username/Travel-Recommendation-System.git
```

### Navigate to the Project Folder

```bash
cd Travel-Recommendation-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.


##  Project Objectives

* Provide personalized travel recommendations.
* Improve travel planning experience.
* Demonstrate practical use of Machine Learning in recommendation systems.
* Build an interactive and user-friendly application.

##  Future Scope

* Real-time weather integration
* Budget-based recommendations
* Hotel and flight suggestions
* User authentication and profiles
* Deep Learning-based recommendation models
* Integration with travel booking platforms

##  License

This project is developed for educational and academic purposes.
