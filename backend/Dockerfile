FROM tiangolo/uvicorn-gunicorn-fastapi:latest-2024-06-17

WORKDIR /app/

# Install Poetry
RUN curl -sSf https://rye.astral.sh/get | bash | RYE_HOME=~/.rye/shims/rye && \
    cd /usr/local/bin && \
    ln -s ~/.rye/shims/rye &&

# Copy poetry.lock* in case it doesn't exist in the repo
COPY ./pyproject.toml ./requirements.lock* /app/

# Allow installing dev dependencies to run tests
ARG INSTALL_DEV=false
RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then rye sync -f ; fi"

ENV PYTHONPATH=/app

COPY ./scripts/ /app/

COPY ./prestart.sh /app/

COPY ./tests-start.sh /app/

COPY ./app /app/app
