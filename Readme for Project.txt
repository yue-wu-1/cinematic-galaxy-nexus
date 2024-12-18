### Cinematic Galaxy Nexus - A Movie Search and Recommendation System

## Overview
Cinematic Galaxy Nexus is an interactive movie search and recommendation system that uses a graph-based approach to explore relationships between movies and actors. This web-based tool allows users to:
1.Search for movies by title, genre, or actor.
2.Discover the most influential actors within a specific year range.
3.Find the shortest connection between two actors based on shared movies.
4.Generate movie and actor recommendations with visualized connection graphs.
Note: The system is built using Flask (backend) and JavaScript (frontend), combined with graph-based algorithms via NetworkX for processing relationships between movies and actors.

## Interactions and Prompts

# Movie Search Engine
1.Prompts:
    "Search By" (Title, Actor, Genre) - drop down box
    "Search Query" (Input field)
    "Sort By" (Release Time, Average Vote Rate, Vote Count, Popularity) - drop down box
    "Order" (Ascending or Descending) - drop down box
2.Output:
    A list of movies with Name, Genres, Average Vote, Vote Count, Popularity, Release Date, Overview and hyperlink nevigating to homepage.
# Most Influential Actor
1.Prompts:
    "Start Year" and "End Year" (Numeric inputs for a year range, limited from 2015 to 2024)
2.Output:
    The most influential actor in a given time period, based on page-rank weighted by popularity of movie.
# Actor Recommendation
1.Prompts:
    "Actor Name" (Text input)
2.Output:
    1)Top 10 Recommended Actors who have significant shared connections (co-starring in movies).
    2)Graph Visualization of direct actor connections, weighted by co-starring movies and its popularity.
# Shortest Actor Connection
1.Prompts:
    "Actor Name 1" and "Actor Name 2" (Text inputs)
2.Output: 
    The shortest connection (path) between the two actors, with an explanation of the movies connecting them.
# Movie Recommendation
1.Prompts:
    "Movie Name" (Text input)
2.Output:
    1)Top 10 Recommended Movies sharing significantly similar actors or genres based on Jaccard Index.
    2)Graph Visualization of direct movie connections.

## Special Instructions
# Required Files:
    1.movies_actors.csv (Dataset containing movie and actor data, extracted from TMDB API.)
        this csv file is already in the project directory. It is created by TMDB_Data_Retrieve.py and it contains movie data from 2015-2024. If you want to explore movie data out of this range, you can modify TMDB_Data_Retrieve.py. (API keys: eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI1Zjc2NjgzYzZiMzQ2MThlZTVjZTY4ZTU3M2ZlYzc0NyIsIm5iZiI6MTczMjQ4ODAyOS43MjcsInN1YiI6IjY3NDNhYjVkNjg4MjMwMDRjYTljYjJjMiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.KmOn1hc0C0Q0VGZ8HxPraIeu9f9GvYUAK-MbiekEFK0)

    Note: Ensure movies_actors.csv is placed in the same directory as the project.


# Static Directory:
The project generates graph images dynamically. Ensure a static/ directory exists in your project folder for saving these images.

# Required Python Packages
Flask, pandas, numpy, networkx, matplotlib, requests, textwrap
Install them using the following command:
#pip install flask pandas numpy networkx matplotlib requests textwrap

# Running the Program:
Use python app.py to start the Flask server.
Open http://127.0.0.1:5000/ in your browser.

## Directory Structure
movie_app/
│
├── _pychche_
├── static/
│   ├── style.css
│   └── images/
│       └── movie_universe_background.png
├── templates
│   └── index.html
├── app.py
├── DataStructrue.py
├── movies_actors.csv


Network (Graph) Organization
1. Actor Graph
Nodes: Actors (represented by actor IDs and names).
Edges: Connections between actors who starred in the same movie.
Edge Weights: Calculated based on movie popularity (normalized values).

2. Movie Graph
Nodes: Movies (represented by movie IDs and titles).
Edges: Connections between movies sharing Actors (shared actors) and Genres (common genres).
Edge Weights: Calculated using Jaccard Index based on shared actors and genres

Output Visualizations
Actor and Movie Graphs are dynamically visualized with:
Node colors: Pink (actors), Skyblue (movies)
Edge colors: Skyblue for actors, Pink for movies

Enjoy navigating the Cinematic Galaxy! 🚀