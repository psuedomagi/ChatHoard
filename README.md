# ChatHoard - Take Control of Your Chats

ChatHoard is a ready-to-deploy full-stake web app that allows your to upload and store your ChatGPT chats (other chat formats planned). It offers robust search and filtering capabilities, and is lightning fast - based on Sebastián Ramírez's Full-StackFastAPI-Template with FastAPI, Pydantic validation/serialization, SQLModel for ORM, and a front-end with React using Chakra UI. It can be deployed locally with Docker Compose in a minimalist deployment, or publicly with authentication and authorization using OAuth2 with JWT tokens and multi-user support. I created ChatHoard to help me manage my ChatGPT chats, but it's designed to scale robustly and will easily extend to other chat formats with added Pydantic models.

I was continually frustrated by how hard it is to search my ChatGPT chats and the lack of organization capabilities.

The Pydantic models are robust and offer a lot of definition for the data; most of this isn't used in the current version, but it would put anyone wanting to run ML models on the data in a good position.

## Currently in active, early development. More to follow. Stay tuned

## Planned Features

- [ ] Modern full-featured UI with React and Chakra UI
- [ ] Superb search and filtering capabilities (powered by meilisearch)
- [ ] Group and organize chats
- [ ] Tag chats
- [ ] Add notes to chats

- [ ] Minimalist local and full/public deployment options with Docker Compose
- [ ] OAuth2 with JWT tokens
- [ ] Passkey authentication
- [ ] Multi-user support
- [ ] Admin/owner UI panel for managing users, permissions, and settings

### Later Planned Features

- [ ] Support for other chat formats (Gemini and Claude); if you want to help add support for your favorite chat format, please reach out. I'd love to collaborate.
- [ ] ML-powered search assistant
- [ ] Chat sharing and in-line commenting
