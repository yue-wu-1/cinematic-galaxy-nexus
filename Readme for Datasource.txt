### Data Sources Overview
## Origin (URLs for Data and Documentation):
    1.API Token: 
"eyJhbGciOiJIUzI1NiJ9eyJhdWQiOiI1Zjc2NjgzYzZiMzQ2MThlZTVjZTY4ZTU3M2ZlYzc0NyIsIm5iZiI6MTczMjQ4ODAyOS43MjcsInN1YiI6IjY3NDNhYjVkNjg4MjMwMDRjYTljYjJjMiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.KmOn1hc0C0Q0VGZ8HxPraIeu9f9GvYUAK-MbiekEFK0"
    2.Data Source: 
        1)TMDB API (The Movie Database), TMDB API Documentation: https://developer.themoviedb.org/docs
        2)Endpoints Used:
            Movie Details: https://api.themoviedb.org/3/movie/{movie_id}
            Movie Credits: https://api.themoviedb.org/3/movie/{movie_id}/credits
## Formats:
    Input Data Format: 
        JSON
    Output File Formats:
        CSV: movies_actors.csv
## Data Access and Caching:

    1.Access Method: The TMDB API was accessed using HTTP requests through the requests Python library.
    Rate Limiting: A delay of 0.25 seconds was added between API calls to respect TMDB's rate limits.

    2.Caching: The retrieved data was saved locally in three steps to reduce redundant API calls:
        Step 1: Popular movies were fetched using the /discover/movie endpoint and stored in popular_movies.json.
        Step 2: Movie details and credits for each movie were fetched and combined, then stored in movies_data.json.
        Step 3: Extract data from movies_data.json and save as movie_actor.csv
## Data Summary:

    1. Data Collected:
        1)Movies Data (from /movie/{movie_id}):

        Number of Variables: 8
            id (Movie ID)
            title (Movie Title)
            genres (Pipe-separated genres)
            homepage (Movie website)
            overview (Description of the movie)
            release_date (Movie release date)
            vote_average (Average rating)
            popularity (TMDB popularity score)

        2)Actors Data (from /movie/{movie_id}/credits):

            Number of Variables: 5
            id (Actor ID)
            name (Actor Name)
            gender (1 = Female, 2 = Male, 0 = Not Specified)
            popularity (Actor popularity score)
            known_for_department (Filtered only for "Acting")

        3)Final Mergeed Data (movies_actors.csv):

        Total Variables: 13

        movie_id
        title
        genres
        homepage
        overview
        release_date
        vote_average
        vote_count
        movie_popularity
        actor_id
        actor_name
        actor_gender
        actor_popularity
        

Notes:
1. Each row corresponds to an actor appearing in a movie. A movie with multiple actors will have multiple rows.
2. The data covers movies released between January 1, 2015, and December 31, 2024.
3. Popular movies were prioritized using the popularity.desc sorting parameter.
4. Only actors with known_for_department == "Acting" were included in the final output.