# jellyseek
deepseek RAG implamentation for jellyfin

Part 1: Roadmap for Building the Project

This roadmap assumes you are working alone and want working prototypes as fast as possible, not a huge monolithic launch.

Each phase must be completed before moving to the next.
Phase 1: Data Extraction from Jellyfin

Goal: Pull your movie library metadata automatically.

Learn how to call Jellyfin’s API.

    Start with simple queries using tools like curl or Postman.

Write a small script to fetch:

    Movie ID

    Title

    Plot

    Genres

    Runtime

    Year

    Save that data to a local JSON or database file.

Success milestone: You can run a script and get an updated full list of your movies automatically.
Phase 2: Build the Embedding and Vector Storage Layer

Goal: Represent movie descriptions in a machine-searchable way.

Pick an embedding model (small one, like bge-small or e5-small).

    Create a script that:

        Loads your movie data.

        Generates an embedding for each movie (turns text into vectors).

        Stores (embedding, movie ID, title, plot, etc.) into a vector store like FAISS.

Success milestone:
You can query "Find me similar movies to 'funny space adventures'" and get a reasonable list from your own library without DeepSeek yet.
Phase 3: User Input and Retrieval

Goal: Process a user request and retrieve candidate movies.

Write a function that:

    Takes natural language input ("scary but not gory movie").

    Embeds that input.

    Searches your vector store.

    Returns top N matches (say 10–20 movies).

    Add basic filtering options (e.g., by year, by runtime if needed).

Success milestone:
You can type a query and get back 10–20 matching movie titles in seconds.
Phase 4: Context Assembly for DeepSeek

Goal: Prepare a clean input prompt for DeepSeek.

Create a prompt template like:

USER REQUEST: {user_input}

CANDIDATE MOVIES:
1. {Title} – {Synopsis} – {Year}
2. {Title} – {Synopsis} – {Year}
...

INSTRUCTIONS: Recommend 3–5 movies ONLY from this list.

    Make sure the entire text stays under DeepSeek’s context window (about 4,000–8,000 tokens depending on your build).

Success milestone:
You generate a well-structured text file ready to feed into DeepSeek.
Phase 5: Integrate DeepSeek-14B Inference

Goal: Run the DeepSeek model on the input and get smart recommendations.

Set up DeepSeek using a backend like vllm, text-generation-webui, or llama.cpp.

Send the context + prompt to DeepSeek.

Receive and parse the model’s recommendation output.

    Capture outputs cleanly (titles, reasons, etc.).

Success milestone:
DeepSeek reads the input and recommends correct movies from your Jellyfin library.
Phase 6: Basic UI or API

Goal: Expose a way for users to easily interact.

Option 1:
Simple command-line interface (CLI).

Option 2:
Build a basic web page where users type requests and see recommendations.

Option 3:
Develop a lightweight Jellyfin plugin to display recommendations inside Jellyfin.

Success milestone:
You have a front-end where a user can type a request and immediately get a list of playable movie suggestions.
Part 2: Flowchart for the Logic and Data Flow

Here is the flowchart structure for your project:

+-------------------------+
| 1. User Types a Request |
+-----------+-------------+
            |
            v
+----------------------------+
| 2. Embed the User Request  |
| (small embedding model)    |
+-----------+----------------+
            |
            v
+----------------------------------+
| 3. Search Vector Store           |
| (find Top 10-20 matching movies) |
+-----------+----------------------+
            |
            v
+------------------------------------+
| 4. Assemble Context Block         |
| (user request + matching movies)  |
+-----------+------------------------+
            |
            v
+--------------------------------+
| 5. Send to DeepSeek-14B        |
| (model generates suggestions)  |
+-----------+--------------------+
            |
            v
+--------------------------------+
| 6. Parse and Display Results   |
| (titles + why recommended)     |
+-----------+--------------------+
            |
            v
+--------------------------------+
| 7. (Optional) Click to Play    |
| (use Jellyfin API to start)    |
+--------------------------------+

3. Quick Visual Summary
Stage	Responsibility	Tools/Models
Extraction	Jellyfin API or Database	requests, sqlite3
Embedding	Text → Numbers	e5-small, bge-small, huggingface
Storage	Store vectors and metadata	FAISS, Chroma
Retrieval	Find similar movies	Vector search
Prompt Construction	Assemble prompt text	Python script
LLM Inference	Generate recommendations	DeepSeek-14B
Frontend/API	Expose to users	CLI, Web, or Jellyfin Plugin

--------------------------------------------------------------------------------------

2. Definitions of Key Technical Terms
Term	Definition
Metadata	Information about your movies, like their title, description, year, genre, runtime, etc.
API	An "Application Programming Interface" — a way for programs to talk to Jellyfin and pull data out automatically.
Embedding	Turning text (like a movie description) into a set of numbers ("vector") that represent the meaning of the text.
Vector	A list of numbers (example: [0.1, 0.8, -0.3, ...]) that a computer can use to measure similarity between texts.
Vector Store	A database specifically designed to store vectors and quickly find "similar" items (example: FAISS, Chroma).
Retrieval	Searching through your stored movie metadata to find the most relevant movies for a user’s request.
Context Window	The amount of information (number of words/tokens) an AI model like DeepSeek can "see" at one time.
Prompt	The text you send to DeepSeek to ask it to perform a task (e.g., "Recommend a lighthearted movie from this list…").
Inference	The process of asking a model like DeepSeek a question and getting an answer.
Token	Pieces of words that an LLM processes (example: "lovely" → "love" + "ly"). Tokens are how LLMs count text.
RAG (Retrieval-Augmented Generation)	A technique where you fetch relevant information first (retrieval) and then pass it into the AI (generation) to get a better answer.
Deduplication	Removing duplicate or similar entries so the system doesn’t suggest the same movie multiple times.
3. Detailed Step-by-Step Architecture
Step 1: Extract Your Jellyfin Movie Library

Goal: Get a list of all your available movies with titles, descriptions, and other details.

    Jellyfin has a REST API that lets you query it.

    You would use an HTTP request like GET /Users/{user_id}/Items?IncludeItemTypes=Movie.

    The result is a JSON object (a structured text format) listing every movie, with information like:

        Title

        Plot/Synopsis

        Genres

        Runtime

        Year

        Unique ID (needed to link back later)

Alternate method: You can also read directly from Jellyfin’s SQLite database if you're comfortable, but using the API is safer.
Step 2: Create Embeddings for Each Movie

Goal: Turn each movie description into a vector (numbers that represent its meaning).

Why?
Computers can't "understand" language easily. But if you turn the language into numbers in a smart way (embedding), you can search and compare meanings very quickly.

How?

    Use a lightweight, fast model just for embedding (examples: e5-small, bge-small-en).

    For every movie:

        Input: title + plot

        Output: a vector like [0.1, -0.3, 0.8, 0.7, ...].

You don’t use DeepSeek itself here because:

    DeepSeek is huge and slow for embedding tasks.

    Specialized embedding models are far faster.

Step 3: Store the Embeddings in a Vector Store

Goal: Save all movie vectors in a special type of database made for fast similarity search.

Options:

    FAISS (Facebook AI Similarity Search)

    Chroma

    Weaviate

    Milvus

You would store:

    Vector (embedding)

    Movie ID

    Title

    Short description

    Other quick metadata

Think of it as:
"Here’s a room full of movies represented as math objects. Find me the ones most 'close' to my search."
Step 4: Accept User Input

Goal: Allow the user to type natural language like:

    "Recommend a fun 90s action movie under 2 hours"

    "I want something scary but not gory"

    "Give me a romantic movie starring Julia Roberts"

This is the "prompt" or "search request" that kicks everything off.
Step 5: Retrieve Relevant Movies from the Vector Store

Goal: Find the movies most related to the user's input.

Process:

    Turn the user’s input into a vector using the same embedding model.

    Search the vector store for the top N closest matches (e.g., 20 closest movies).

    Optionally: apply additional filters (e.g., only movies under 2 hours).

This way you only give DeepSeek a small, focused list of movies to choose from.
Step 6: Build a Context Block for DeepSeek

Goal: Assemble a clean, readable list of candidates for DeepSeek to "think about."

Example Format:

USER REQUEST: "Light-hearted sci-fi movie under 2 hours."

CANDIDATE MOVIES:
1. Galaxy Quest (1999) – A group of washed-up TV actors are mistaken for real space heroes by an alien race.
2. Men in Black (1997) – A secret agency polices extraterrestrial activity on Earth.
3. Spaceballs (1987) – A parody of classic sci-fi movies as a group tries to save a princess.

INSTRUCTIONS: Recommend 3-5 titles only from this provided list and justify each pick in 1-2 sentences.

Important: You must tell DeepSeek clearly:

    "ONLY recommend from this list."

Otherwise it may hallucinate and suggest random movies you don’t own.
Step 7: Run Inference with DeepSeek

Goal: Send the structured prompt into DeepSeek and get back recommendations.

    DeepSeek processes the context block and user request.

    It selects movies and explains why they match.

    Output example:

1. Galaxy Quest – Its light-hearted tone and sci-fi setting make it a perfect match for a fun experience.
2. Men in Black – Combines humor and action in a sci-fi backdrop, well under 2 hours.

Step 8: Return the Results to the User

Goal: Send the DeepSeek output to the user’s screen.

    If you want it to look nice inside Jellyfin, you could create a Jellyfin plugin.

    Otherwise, a simple web dashboard (HTML/JS) or even a command-line app would work.

When the user clicks a recommendation:

    You use the Jellyfin API to open/play the selected movie.

4. Constraints You Must Handle
Constraint	What It Means
Context Window Limit	DeepSeek can only process a few thousand tokens at once. You can't send 1,000 movies in one go; retrieve only the top matches.
Latency	DeepSeek-14B is large. Expect about 10–15 seconds delay unless you heavily optimize.
Security	You’re exposing metadata about your movie collection. Only allow access from trusted users (your LAN).
Data Freshness	If you add or delete movies, you need to refresh your vector store regularly (automatically if possible).
5. Summary

You will build a Retrieval-Augmented Generation (RAG) system:

    Extract movies from Jellyfin (your knowledge base).

    Build a smart searchable database (vector store).

    Retrieve top matches based on the user’s natural-language question.

    Feed only those into DeepSeek-14B.

    Get natural, helpful recommendations that only pick from your real library.

It is very achievable even for a first project if you break it down into steps and don’t rush.