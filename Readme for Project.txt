------------------------------------------2020-12-19-----------------------------------------
IMMPORTANT UPDATE: 
1. Deployed this project on pythonanywhere, so users can access the website directly by this URL:
Yue165.pythonanywhere.com
2. Expanded the movie data search range to 1950 - 2024.

------------------------------------------2020-12-18-----------------------------------------
### Cinematic Galaxy Nexus - A Movie Search and Recommendation System
Demo.video [Google doc link: https://drive.google.com/file/d/1VTGARQuzOcoQVUXma9YA7MZSjREZu_dR/view?usp=drive_link]

## Overview
Cinematic Galaxy Nexus is an interactive movie search and recommendation system that uses a graph-based approach to explore relationships between movies and actors. This web-based tool allows users to:

1.Search for movies by title, genre, or actor.
2.Discover the most influential actors within a specific year range.
3.Find the shortest connection path between any two actors based on co-starring movies.
4.Generate movie and actor recommendations with visualized connection graphs.
Note: This system is built using Flask framework

## Interactions and Prompts

# Movie Search Engine
1.Prompts:
    Drop Down Box:   "Search By" (Title, Actor, Genre) 
    User-input Text: "Search Query"
    Drop Down Box:   "Sort By" (Release Time, Average Vote Rate, Vote Count, Popularity) 
    Drop Down Box:   "Order" (Ascending or Descending) 
2.Output:
    A list of movies with Name, Genres, Average Vote, Vote Count, Popularity, Release Date, Overview and hyperlink navigating to movie's homepage.

# Most Influential Actor
1.Prompts:
    User-input Number: "Enter Start Year" | "Enter End Year" (limited from 2015 to 2024)
2.Output:
    The most influential actor in a given time period, based on page-rank weighted by popularity of movies.

# Actor Recommendation
1.Prompts:
    User-input Text: "Enter Actor Name"
2.Output:
    1)Top 10 Recommended actors based on relevance to target actor
    2)Graph Visualization of direct actors' connections, weighted by co-starring movies and corresponding popularity.

# Shortest Actor Connection
1.Prompts:
    User-input Text: "Enter Actor Name 1" and "Enter Actor Name 2"
2.Output: 
    The shortest connection path between any two actors, with navigation using co-starring movies

# Movie Recommendation
1.Prompts:
    User-input Text: "Movie Name"
2.Output:
    1)Top 10 Recommended Movies based on shared actors or genres.
    2)Graph Visualization of movies' connection, weighted by Jaccard Index.

API note: To save time for launching project. I have saved data retrieved from TMDB API into movies_actors.csv file. So you do not need API key to launch this project.
This file contains most movies (first 50 pages) released from 2015 to 2024 on TMDB database. If you want to retrieve even more movie data, which may spend long time, from TMDB API, you can modify release_year parameters in TMDB_Data_Retrieve.py. and run it. Then the exist movies_actors.csv file will be replaced and you can use this recommendation system to explore more movie data. Have fun!

API keys (Optional): eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI1Zjc2NjgzYzZiMzQ2MThlZTVjZTY4ZTU3M2ZlYzc0NyIsIm5iZiI6MTczMjQ4ODAyOS43MjcsInN1YiI6IjY3NDNhYjVkNjg4MjMwMDRjYTljYjJjMiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.KmOn1hc0C0Q0VGZ8HxPraIeu9f9GvYUAK-MbiekEFK0)

******************************************Launch Instruction************************************
1. Download the whole project directory to local computer [Do not change directory structure]
2. Required Python Packages: Flask, pandas, numpy, networkx, matplotlib, requests
	Install them using the following command:
	#pip install flask pandas numpy networkx matplotlib requests
3. Running run.py on terminal using following command:
	#python3 run.py
4. If successful, you will see "Running on http://xxx.x.x.x:5000" on your terminal. Open that "http://xxx.x.x.x:5000/" in your browser then you can interact with this website.

## Directory Structure [do not change]

movie_app/
│
├── _pychche_
├── static/
│   ├── css
│   │   
│   ├── images
│   └── music
├── templates
│   
├── run.py
├── DataStructrue.py
├── movies_actors.csv
├── TMDB_Data_Retrieve.py



Network (Graph) Organization
1. Actor Graph
Nodes: Actors (represented by actor IDs and names).
Edges: Connections between actors who co-starring same movies.
Edge Weights: Calculated based on movie popularity (normalized values).

2. Movie Graph
Nodes: Movies (represented by movie IDs and titles).
Edges: Connections between movies with shared Actors and Genres.
Edge Weights: Calculated using Jaccard Index based on shared actors and genres

Output Visualizations
1. Actors' connection visualization shows the input actor in the center, with top 10 most relevant actors radiating outward.
Numbers on edge indicates degree centrality weighted by popularity. Node size represents relevance.
2. Movies' connection visualization shows the input movie in the center, with their top 10 most similar movies radiating outward. Numbers on edge indicate Jaccard Index based on genres and actors. Node size represents degree of similarity.


Welcome to Cinematic Galaxy Nexus. Have Fun! 🚀🚀🚀
