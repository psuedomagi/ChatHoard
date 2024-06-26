# Notes for build

## General

- Add username as a user field; use this instead of email for login
- Forbade email-based usernames
- Add a user role field to the user model
-

## Meilisearch implmentation

- Use brotli for compression
- Identify search facets for filtering (filterAttributes)
- Implement search facets
- Identify search attributes for indexing
- Index snapshoting and reference storing in MongoDB
- Implement per-user indexing and authentication (JWT/JWE transmitted per-user API key; we store encrypted in Mongo)
- Implement attribute-based chat splitting to improve search results
- Implement search suggestions
- Implement search results highlighting
- Implement search results sorting
- Implement search results pagination

- Add model selection for semantic search and reranking to config (
    mxbai-embed-large-v1 for embeddings, mxbai-rerank-xsmall-v1 for reranking)

- Add reranker (mxbai-rerank-xsmall-v1)

- meilisync for syncing with MongoDB
- meilisync for progress caching with Redis

- arq queues indexing jobs
- arq queues search jobs

- add http/2 ssl for remote deploys
- SEARCH AUTHENTICATION

later: option for langchain/openai for embeddings instead of sentence-transformers ( multi-qa-MiniLM-L6-cos-v1 )


## MongoDB implementation

- Add user model
- Add user role model
- Add user role permissions
- Add user role-based permissions
- Encrypted storage of per-user API keys

- Add conversation model
- Add search index model

- Add search index snapshot model
- Add search index reference model

- AUTHENTICATION

## Redis implementation

- Add progress caching for meilisync
- Add search result caching
- Add search suggestion caching
- Add search facet caching

## Arq implementation

- Add indexing job queue
- Add search job queue
- Add search suggestion job queue
- Add search facet job queue

## Security/auth

- Integrate Oath2 into JWT for authentication/segregation of user roles
- Add user role-based permissions
- Add user role-based access control
- Add user role-based access control for search results and index access
- Add user RBAC for API generation/access


## Chat interface/abstraction

- Add chat interface for easy integration of different chat providers

## API


## Plugin system?


## Better config system
