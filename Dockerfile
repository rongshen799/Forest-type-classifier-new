FROM python:3.11

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    && apt-get clean

WORKDIR /streamlit_app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"]
