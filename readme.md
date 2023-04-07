Test project to use GPT to query against 2022 supreme court rulings.

Requires the following environment variables to be set:
OPENAI_API_KEY
PINECONE_API_KEY

See config.yml for Pinecone environment, local path options

Requires a pinecone index to be manually initialized with 1536 dimensions.

loader.py is a script to seed your Pinecone database

main.py has the query code, with a default question and two additional sample questions