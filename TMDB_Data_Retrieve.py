import requests
import time
import json
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from datetime import datetime

###step1: Retrieve data from TMDB api: with movie details endpoint and movie credits endpoint combined by movie id ###
# return as .json format #

API_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI1Zjc2NjgzYzZiMzQ2MThlZTVjZTY4ZTU3M2ZlYzc0NyIsIm5iZiI6MTczMjQ4ODAyOS43MjcsInN1YiI6IjY3NDNhYjVkNjg4MjMwMDRjYTljYjJjMiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.KmOn1hc0C0Q0VGZ8HxPraIeu9f9GvYUAK-MbiekEFK0"

# Headers for TMDB API requests
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json;charset=utf-8"
}

# Base URLs
DISCOVER_URL = "https://api.themoviedb.org/3/discover/movie"
MOVIE_DETAILS_URL = "https://api.themoviedb.org/3/movie/{}"
MOVIE_CREDITS_URL = "https://api.themoviedb.org/3/movie/{}/credits"

# Parameters for searching movies
DISCOVER_PARAMS = {
    "sort_by": "popularity.desc",
    "release_date.gte": "1950-01-01",
    "release_date.lte": "2024-12-31",
    "language": "en-US",
    "page": 1
}

def get_popular_movies():
    movies = []
    page = 1
    total_pages = 1  # Will be updated after first request

    while page <= total_pages and page <= 500:  # Limit to first 50 pages
        DISCOVER_PARAMS['page'] = page
        response = requests.get(DISCOVER_URL, headers=HEADERS, params=DISCOVER_PARAMS)
        if response.status_code == 200:
            data = response.json()
            total_pages = data['total_pages']
            movies.extend(data['results'])
            print(f"Fetched page {page}/{total_pages}")
        else:
            print(f"Failed to fetch movies on page {page}: {response.status_code}")
            break
        page += 1
        time.sleep(0.25)  # To respect rate limits

    return movies

# Fetch popular movies
popular_movies = get_popular_movies()

# Save to JSON file
with open("popular_movies.json", "w", encoding='utf-8') as f:
    json.dump(popular_movies, f, ensure_ascii=False, indent=4)


def get_movie_details_and_credits(movie_id):
    # Get movie details
    details_response = requests.get(MOVIE_DETAILS_URL.format(movie_id), headers=HEADERS)
    # Get movie credits
    credits_response = requests.get(MOVIE_CREDITS_URL.format(movie_id), headers=HEADERS)
    
    if details_response.status_code == 200 and credits_response.status_code == 200:
        details = details_response.json()
        credits = credits_response.json()
        return details, credits
    else:
        print(f"Failed to fetch data for movie ID {movie_id}")
        return None, None

# Prepare lists to hold combined data
movies_data = []

# Loop through popular movies and get details and credits
for movie in popular_movies:
    movie_id = movie['id']
    details, credits = get_movie_details_and_credits(movie_id)
    if details and credits:
        # Merge details and credits
        movie_data = details.copy()
        movie_data.update({
            'cast': credits.get('cast', []),
            'crew': credits.get('crew', [])
        })
        movies_data.append(movie_data)
    time.sleep(0.25)  # Respect rate limits

# Save combined data to JSON file
with open("movies_data.json", "w", encoding='utf-8') as f:
    json.dump(movies_data, f, ensure_ascii=False, indent=4)

# Load the combined data from the JSON file
with open("movies_data.json", "r", encoding='utf-8') as f:
    movies_data = json.load(f)

# Display the first 5 entries
for i, movie in enumerate(movies_data[:5], start=1):
    print(f"Movie {i}:")
    print(json.dumps(movie, indent=4))
    print("\n---\n")


###step1: Retrieve data from TMDB api: with movie details endpoint and movie credits endpoint combined by movie id ###

###step2: Retrieve targarted variables in the json data and output as .csv file ###
import csv
from datetime import datetime

# Load the combined data from the JSON file you created
with open("movies_data.json", "r", encoding='utf-8') as f:
    movies_data = json.load(f)

# Prepare CSV file
output_file = "movies_actors.csv"
with open(output_file, "w", encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header
    header = [
        "movie_id",
        "title",
        "genres",
        "homepage",
        "overview",
        "release_date",
        "vote_average",
        "vote_count",
        "movie_popularity",
        "actor_id",
        "actor_name",
        "actor_gender",
        "actor_popularity"
    ]
    writer.writerow(header)
    
    # Iterate over each movie record
    for movie in movies_data:
        movie_id = movie.get("id")
        title = movie.get("title", "")
        
        # Extract genres as a pipe-separated string of names
        genre_names = [g["name"] for g in movie.get("genres", [])]
        genres_str = "|".join(genre_names)
        
        homepage = movie.get("homepage", "")
        overview = movie.get("overview", "").replace("\n", " ").strip()
        release_date = movie.get("release_date", "")
        vote_average = movie.get("vote_average", "")
        vote_count = movie.get("vote_count", "")
        movie_popularity = movie.get("popularity", "")
        
        # Extract actors (cast)
        cast_list = movie.get("cast", [])
        
        # Filter only entries where known_for_department == "Acting"
        cast_list = [actor for actor in cast_list if actor.get("known_for_department") == "Acting"]
        
        for actor in cast_list:
            actor_id = actor.get("id", "")
            actor_name = actor.get("name", "")
            actor_gender = actor.get("gender", "")  # 1 = female, 2 = male, 0/other = not specified
            actor_popularity = actor.get("popularity", "")
            
            # Write row for each (movie, actor)
            writer.writerow([
                movie_id,
                title,
                genres_str,
                homepage,
                overview,
                release_date,
                vote_average,
                vote_count,
                movie_popularity,
                actor_id,
                actor_name,
                actor_gender,
                actor_popularity
            ])

