FROM registry.suse.com/bci/python:3.13

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml pyproject.toml
COPY src/ src/

# Install dependencies using uv
RUN uv sync

# Run the supervisor
CMD ["uv", "run", "python", "-m", "src.supervisor"]
