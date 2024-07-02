# :dragon::moneybag: ChatHoard: Take Control of Your Precious ChatGPT [+more] History :ring:

ChatHoard is a ready-to-deploy full-stack web app for making the most of your ChatGPT chat history (other chat formats planned). ChatHoard offers robust search and filtering capabilities, and is lightning fast - based on Full-Stack-FastAPI-Template with FastAPI, Pydantic validation/serialization, SQLModel for ORM, and a front-end with React using Chakra UI. You can be deploy ChatHoard locally with Docker Compose in a minimalist deployment, or publicly with authentication and authorization using OAuth2 with JWT tokens and multi-user support. I created ChatHoard to help me manage my ChatGPT chats, but it's designed to scale robustly and can extend to other chat formats with added Pydantic models.

I was continually frustrated by how hard it is to search my ChatGPT chats and the lack of organization capabilities.

The Pydantic models are robust and offer a lot of definition for the data; most of this isn't used in the current version, but it would put anyone wanting to run ML models on the data in a good position.

## Currently in active, early development. Stay tuned. NOT YET FUNCTIONAL

## Planned Initial Features

User Interface:

- [ ] Modern full-featured UI with React and Chakra UI
- [ ] ML powered hybrid vector search powered by Qdrant and DocArray
- [ ] Easy graphical import/export of chats

Backend:

- [ ] Minimalist local and full/public deployment options with Docker Compose
- [ ] OAuth2 with JWT tokens
- [ ] Multi-user support
- [ ] Admin/owner UI panel for managing users, permissions, and settings

### Later Planned Features

- [ ] Rust-wasm powered client-side embedding generation for search indexing; this will be server-side for now
- [ ] Group and organize chats
- [ ] Tag chats
- [ ] Add notes to chats
- [ ] Automated download requests for new chats from ChatGPT (these are emailed to you); and deletion of old chats from OpenAI
- [ ] Additional encryption options (e.g. differential privacy, possibly optional client-side/e2e encryption; I have some ideas for doing this even if I need to do compute on the server side... but it's a bit of a stretch goal)
- [ ] Support for other chat formats (Gemini and Claude); I welcome your help adding support for your favorite chat format, please reach out. I'd love to collaborate. I don't have the data for these formats, so I at least need someone to donate some data to me, or I can help get you started on adding support for your format. (I'd like a sample of at least ~100 chats to iron out idiosyncrasies in the format, but more is better).
- [ ] ML-powered search assistant/RAG
- [ ] Chat sharing and in-line commenting, collaboration
- [ ] Passkey/webauthn authentication
- [ ] NLP-powered groupings and tagging

### Maybe Later Features

- [ ] Fully private subscription-based cloud deployment/hosting or SaaS??? (I'd do it for free if I could but, you know, money)
