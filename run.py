"""
Flask Web Application for Movie and Actor Recommendation System

This application provides a web interface for exploring and visualizing connections 
between movies and actors based on co-starring relationships, genres, and popularity.

Key Features:
- Find the most influential actor in a given time range.
- Explore an actor's connections and visualize their relationships with other actors.
- Find the shortest path between two actors in the actor graph.
- Recommend movies similar to a given movie based on shared actors or genres.
- Search for movies based on title, actor, or genre.
- Visualize direct connections for movies and actors using interactive graph plots.

Modules:
    - `MovieData`: Handles data loading, cleaning, and aggregation for movies and actors.
    - `GraphBuilder`: Constructs weighted graphs for actors and movies based on relationships.
    - `Recommender`: Provides recommendation and search functionalities.
    - `visualize_direct_links`: A utility function to visualize direct connections in a graph.

Dependencies:
    - Flask: Web framework for building the application.
    - pandas: For data manipulation and analysis.
    - numpy: For numerical operations.
    - networkx: For graph-based operations.
    - matplotlib: For graph visualizations.
    - DataStructure (custom module): Contains classes and functions for data processing and graph-building logic.

How to Run:
1. Ensure the necessary dependencies are installed.
2. Place the required `movies_actors.csv` file in the working directory.
3. Run the script using Python.
4. Open the application in a web browser (default: http://127.0.0.1:5000).

Static Assets:
    - Graph visualizations and generated images are saved in the `static/` directory.

Environment:
    - The application uses Flask's development server in debug mode for testing.

Author:
    Yue Wu - University of Michigan
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import networkx as nx
from DataStructure import MovieData, GraphBuilder, Recommender, visualize_direct_links
#from openai import OpenAI
import os

app = Flask(__name__)

# Load data and initialize logic
data = MovieData(file_path="movies_actors.csv")
data.load_data().clean_data().aggregate_data()

# Build graphs
graph_builder = GraphBuilder(data.movies_df, data.actors_df, data.actor_id_to_name, data)
actor_graph = graph_builder.build_actor_graph()
movie_graph = graph_builder.build_movie_graph(alpha=1.0, beta=0.5, gamma=1.0, delta=0.5)

recommender = Recommender(data.movies_df, data.actors_df, data.actor_id_to_name, actor_graph, movie_graph)
#client = OpenAI()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/influential_actor", methods=["POST"])
def influential_actor():
    """
    Finds the most influential actor in a specified year range.

    Request JSON:
        {
            "start_year": int,  # Start year for the query.
            "end_year": int     # End year for the query.
        }

    Response JSON:
        - Success:
            {
                "success": True,
                "actor_name": str,  # Name of the most influential actor.
                "score": float      # PageRank score of the actor.
            }
        - Failure:
            {
                "success": False,
                "message": str  # Error or failure message.
            }
    """
    data_json = request.get_json()
    start_year = data_json.get("start_year", "").strip() if data_json else ""
    end_year = data_json.get("end_year", "").strip() if data_json else ""
    try:
        start_year = int(start_year)
        end_year = int(end_year)
        if start_year > end_year:
            return jsonify({"success": False, "message": "Start year must be less than or equal to the end year."})
        actor_id, actor_name, score = graph_builder.get_most_influential_actor(start_year, end_year)
        if actor_id:
            return jsonify({"success": True, "actor_name": actor_name, "score": score})
        return jsonify({"success": False, "message": f"No data available for the range {start_year} to {end_year}."})
    except ValueError:
        return jsonify({"success": False, "message": "Invalid year. Please provide valid numeric years."})

@app.route("/actor_connections", methods=["POST"])
def actor_connections():
    """
    Fetches the top 10 actor connections for a given actor.

    Request JSON:
        {
            "actor_name": str  # Name of the actor.
        }

    Response JSON:
        - Success:
            {
                "success": True,
                "connections": [
                    {"actor_name": str}  # List of connected actors (top 10 by weight).
                ]
            }
        - Failure:
            {
                "success": False,
                "message": str  # Error or failure message.
            }
    """
    data_json = request.get_json()
    actor_name = data_json.get("actor_name", "").strip().lower() if data_json else ""
    actor_row = data.actors_df[data.actors_df['actor_name'].str.lower() == actor_name]
    if not actor_row.empty:
        actor_id = actor_row['actor_id'].iloc[0]
        if actor_id in actor_graph:
            # Retrieve neighbors and their weights
            connections = [
                {"actor_name": data.actor_id_to_name[neighbor], "weight": actor_graph[actor_id][neighbor]['weight']}
                for neighbor in actor_graph.neighbors(actor_id)
            ]
            # Sort by weight in descending order
            connections.sort(key=lambda x: x['weight'], reverse=True)
            # Take the top 10 and remove the weight from the returned data
            top_10_actors = [{"actor_name": conn["actor_name"]} for conn in connections[:10]]

            return jsonify({"success": True, "connections": top_10_actors})
        else:
            return jsonify({"success": False, "message": f"No connections found for actor: {actor_name}."})
    return jsonify({"success": False, "message": f"Actor '{actor_name}' not found."})

@app.route("/actor_connections_image", methods=["POST"])
def actor_connections_image():
    """
    Generates a visualization of an actor's top 10 connections.

    Request JSON:
        {
            "actor_name": str  # Name of the actor.
        }

    Response JSON:
        - Success:
            {
                "success": True,
                "image_url": str  # URL of the generated image.
            }
        - Failure:
            {
                "success": False,
                "message": str  # Error or failure message.
            }
    """
    data_json = request.get_json()  # Retrieve JSON data
    actor_name = data_json.get("actor_name", "").strip().lower()

    if not actor_name:
        return jsonify({"success": False, "message": "Actor name is required."})

    # Ensure actor exists in the dataset
    actor_row = data.actors_df[data.actors_df['actor_name'].str.lower() == actor_name]
    if not actor_row.empty:
        actor_id = actor_row['actor_id'].iloc[0]
        # Sanitize the actor name for the filename
        sanitized_actor_name = "".join(c for c in actor_name if c.isalnum() or c in (" ", "_")).strip()
        filename = f"static/{sanitized_actor_name}_connections.png"

        # Visualize the connections
        success = visualize_direct_links(
            graph=actor_graph,
            main_node_id=actor_id,
            filename=filename,
            top_n=10,  # Adjustable parameter for the number of connections
            get_label_func=lambda n: data.actor_id_to_name.get(n, str(n)),  # Map actor_id to actor name
            node_color="pink",  # Set to pink for consistency
            edge_color="skyblue",     # Set edge color to skyblue theme
            background_color="#375662", # Set background to black for aesthetic
            wrap_width=20
        )
        if success:
            return jsonify({"success": True, "image_url": f"/{filename}"})
        return jsonify({"success": False, "message": "Failed to generate connections image."})

    return jsonify({"success": False, "message": f"Actor '{actor_name}' not found."})

@app.route("/shortest_actor_connection", methods=["POST"])
def shortest_actor_connection():
    """
    Finds the shortest connection between two actors in the actor graph.

    Request JSON:
        {
            "actor_name_1": str,  # Name of the first actor.
            "actor_name_2": str   # Name of the second actor.
        }

    Response JSON:
        - Success:
            {
                "success": True,
                "path": [str],       # List of actor names in the shortest path.
                "explanation": str   # Explanation of the connection.
            }
        - Failure:
            {
                "success": False,
                "message": str  # Error or failure message.
            }
    """
    data_json = request.get_json()
    actor_name_1 = data_json.get("actor_name_1", "").strip()
    actor_name_2 = data_json.get("actor_name_2", "").strip()

    shortest_path, error_message = graph_builder.find_shortest_actor_connection(actor_name_1, actor_name_2)
    if shortest_path is not None:
        path_names, explanation = shortest_path, error_message
        return jsonify({"success": True, "path": path_names, "explanation": explanation})
    else:
        return jsonify({"success": False, "message": error_message})

@app.route("/movie_connections", methods=["POST"])
def movie_connections():
    """
    Finds the top 10 connections for a given movie based on shared actors or genres.

    Request JSON:
        {
            "movie_name": str  # Name of the movie.
        }

    Response JSON:
        - Success:
            {
                "success": True,
                "connections": [str]  # List of connected movies (top 10 by weight).
            }
        - Failure:
            {
                "success": False,
                "message": str  # Error or failure message.
            }
    """
    data_json = request.get_json()  # Retrieve JSON data
    movie_name = data_json.get("movie_name", "").strip().lower()

    # Ensure the movie exists in the dataset
    movie_row = data.movies_df[data.movies_df['title'].str.lower() == movie_name]
    if not movie_row.empty:
        movie_id = movie_row['movie_id'].iloc[0]

        # Get neighbors and weights
        if movie_id in movie_graph:
            connections = [
                {
                    "title": movie_graph.nodes[neighbor].get('title', 'Unknown'),  # Safely fetch 'title'
                    "weight": movie_graph[movie_id][neighbor]['weight']  # Get edge weight
                }
                for neighbor in movie_graph.neighbors(movie_id)
            ]

            # Sort connections by weight in descending order
            sorted_connections = sorted(connections, key=lambda x: x['weight'], reverse=True)

            # Take the top 10 connections
            top_10_connections = [conn['title'] for conn in sorted_connections[:10]]

            return jsonify({"success": True, "connections": top_10_connections})

        else:
            return jsonify({"success": False, "message": f"No connections found for movie: {movie_name}."})
    return jsonify({"success": False, "message": f"Movie '{movie_name}' not found."})

@app.route("/movie_connections_image", methods=["POST"])
def movie_connections_image():
    """
    Generates a visualization of a movie's top 10 connections.

    Request JSON:
        {
            "movie_name": str  # Name of the movie.
        }

    Response JSON:
        - Success:
            {
                "success": True,
                "image_url": str  # URL of the generated image.
            }
        - Failure:
            {
                "success": False,
                "message": str  # Error or failure message.
            }
    """
    data_json = request.get_json()  # Retrieve JSON data
    movie_name = data_json.get("movie_name", "").strip().lower()

    if not movie_name:
        return jsonify({"success": False, "message": "Movie name is required."})

    # Ensure movie exists in the dataset
    movie_row = data.movies_df[data.movies_df['title'].str.lower() == movie_name]
    if not movie_row.empty:
        movie_id = movie_row['movie_id'].iloc[0]
        sanitized_movie_name = "".join(c for c in movie_name if c.isalnum() or c in (" ", "_")).strip()
        filename = f"static/{sanitized_movie_name}_connections.png"

        # Visualize the connections
        success = visualize_direct_links(
            graph=movie_graph,
            main_node_id=movie_id,
            filename=filename,
            top_n=10,  # Adjustable parameter for the number of connections
            get_label_func=lambda n: movie_graph.nodes[n].get('title', str(n)),  # Map movie_id to title
            node_color="skyblue",
            edge_color="pink",
            background_color="#375662",
            wrap_width=20
        )
        if success:
            return jsonify({"success": True, "image_url": f"/{filename}"})
        return jsonify({"success": False, "message": "Failed to generate connections image."})

    return jsonify({"success": False, "message": f"Movie '{movie_name}' not found."})


@app.route("/search_movies", methods=["POST"])
def search_movies_route():
    """
    Searches for movies based on a factor (title, actor, or genre) and sorts the results.

    Request JSON:
        {
            "factor": str,       # Search factor ('title', 'actor', or 'genre').
            "query": str,        # Search query.
            "sort_by": str,      # Sorting criterion ('release time', 'average vote rate', etc.).
            "ascending": bool    # Sort order (True for ascending, False for descending).
        }

    Response JSON:
        - Success:
            {
                "success": True,
                "results": [        # List of movies matching the search criteria.
                    {
                        "title": str,
                        "actors": [str],
                        "genres": str,
                        "vote_average": float,
                        "vote_count": int,
                        "movie_popularity": float,
                        "release_date": str,
                        "overview": str,
                        "homepage": str,
                    }
                ]
            }
        - Failure:
            {
                "success": False,
                "message": str  # Error or failure message.
            }
    """
    data_json = request.get_json()
    factor = data_json.get("factor", "").strip().lower()
    query = data_json.get("query", "").strip()
    sort_by = data_json.get("sort_by", "").strip().lower()
    ascending = data_json.get("ascending", True)  # Default to True if not provided

    try:
        results = Recommender.search_movies(
            data.movies_df,
            data.actors_df,
            data.actor_id_to_name,
            factor,
            query,
            sort_by,
            ascending
        )

        if results.empty:
            return jsonify({"success": False, "message": f"No results found for '{query}'."})

        # Add actors for each movie
        movies = results.apply(
            lambda row: {
                "title": row["title"],
                "actors": [data.actor_id_to_name.get(aid, "Unknown") for aid in row["actor_ids"]],
                "genres": row["genres"],
                "vote_average": row["vote_average"],
                "vote_count": row["vote_count"],
                "movie_popularity": row["movie_popularity"],
                "release_date": row["release_date"],
                "overview": row["overview"] or "N/A",
                "homepage": row["homepage"] or "",
            },
            axis=1,
        ).tolist()

        return jsonify({"success": True, "results": movies})
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)})



#def generate_movie_summary(client, movie_name, genres, overview, vote_average):
#    prompt = (
#        f"Write a concise summary about the movie '{movie_name}' in less than 100 words. "
#        f"Genres: {genres}. It has an average rating of {vote_average}. "
#        f"Here is the overview: {overview}"
#        f"Keep the output format relatively fixed. Talk about the genres, vote_average and then a concise description of overview"
#    )
#    completion = client.chat.completions.create(
#        model="gpt-4o-mini",
#        messages=[
#            {"role": "system", "content": "You are a helpful assistant specializing in movies."},
#            {"role": "user", "content": prompt}
#        ]
#    )
#    return completion.choices[0].message.content


if __name__ == "__main__":
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
