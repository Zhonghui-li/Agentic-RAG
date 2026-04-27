## Backend

### 1. Install MongoDB from [here](https://www.mongodb.com/try/download/community)

#### MongoDB on MacOS

```sh
brew tap mongodb/brew
brew update
brew install mongodb-community
```

If the last command errors, do this:

```sh
brew edit mongodb-community
```

Then replace the `sha256` with what is shown as the actual, and then after run:

```sh
brew install mongodb-community
```

Then to start the mongodb, run this:

```sh
brew services start mongodb-community
```

### 2. Clone repository and Install Poetry

```sh
cd backend
pip3 install poetry
```

### 3. Setting python version with pyenv

```bash
pyenv install 3.11.10
pyenv local 3.11.10
poetry env use $(pyenv which python)
```

### 4. Install the backend dependencies

```sh
poetry install --no-root
poetry run python app.py
```

### Backend Configuration

| Key           | Value                   |
| ------------- | ----------------------- |
| python        | ^3.12                   |
| flask         | ^3.0.3                  |
| flask-pymongo | ^2.3.0                  |
| flask-cors    | ^4.0.1                  |
| mongoengine   | ^0.28.2                 |
| pymongo       | ^4.8.0                  |
| requires      | poetry-core             |
| build-backend | poetry.core.masonry.api |
