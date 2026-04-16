# LLM-logic

<!-- ![Build Status](https://img.shields.io/github/actions/workflow/status/shoot649854/llm-logic-react/ci.yml?branch=main)
![License](https://img.shields.io/github/license/shoot649854/llm-logic-react)
![Version](https://img.shields.io/github/package-json/v/shoot649854/llm-logic-react?private=true)
![Dependencies](https://img.shields.io/librariesio/github/shoot649854/llm-logic-react?private=true)
![Dev Dependencies](https://img.shields.io/librariesio/github/shoot649854/llm-logic-react?private=true&type=dev)
![Issues](https://img.shields.io/github/issues/shoot649854/llm-logic-react) -->

<!-- ### Language -->

![Python](https://img.shields.io/badge/-Python-F9DC3E.svg?logo=python&style=flat)
![TypeScript](https://img.shields.io/badge/-TypeScript-007ACC.svg?logo=typescript&style=flat)
![Flask](https://img.shields.io/badge/-Flask-000000.svg?logo=flask&style=flat)
![React](https://img.shields.io/badge/React-18.3.3-blue)
![Next](https://img.shields.io/badge/Next-14.2.4-black)
![Node](https://img.shields.io/badge/Node-20.14.10-green)

## Notion

Here is a [link to the Notion](https://www.notion.so/team/21c564c6-a5b2-47d3-ac19-7b778d8778f6/join), all tasks will be specified there.

### Branch Naming Rules

| Branch Name                    | Description            | Supplemental |
| ------------------------------ | ---------------------- | ------------ |
| main                           | latest release         |              |
| dev/main                       | latest for development |              |
| hotfix/{module name}/{subject} | Hotfix branch          |              |
| sandbox/{anything}             | test code, etc.        |              |

### Basic Branch Operation Rules

- Work is branched from each latest branch
- Delete working branches after merging
- Review as much as possible (have someone do it for you)
- Build, deploy, etc. are discussed separately.

## Installation

1. **Clone the repository:**

   ```sh
   git clone <repository-url>
   ```

2. **Install SWIProlog**
<https://www.swi-prolog.org/>

## Front End

The react app is created using [next.js](https://nextjs.org) because it is the recommended library on the [react getting started](https://react.dev/learn/start-a-new-react-project).

1. **Install dependencies:**

   Install NodeJS
   [node](https://nodejs.org/en/download/package-manager)

   Make sure you have [pnpm](https://pnpm.io/) installed. If not, install it using:

   ```sh
   cd frontend
   npm install -g pnpm
   ```

   Then, install the project dependencies:

   ```sh
   pnpm install
   pnpm run dev
   ```

Make sure to have .env file on `frontend/.env`

```bash
NEXT_PUBLIC_Google_Client_ID=
NEXT_PUBLIC_Google_Client_Secret=
```

## Dependencies

| Package             | Version |
| ------------------- | ------- |
| next                | 14.2.4  |
| react               | ^18.3.1 |
| react-dom           | ^18.3.1 |
| tailwind-merge      | ^2.3.0  |
| tailwindcss-animate | ^1.0.7  |
| eslint              | ^8.57.0 |
| postcss             | ^8.4.39 |
| tailwindcss         | ^3.4.4  |
| typescript          | ^5.5.3  |

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

## Postman

1. **Open Postman and Import the Collection**:

   - Click on `Import` and select the `.json` file from the `backend/collection` directory.

2. **Set Up Environment Variables**:
   - Ensure to set the variables such as `BASE_URL`, `AUTH_TOKEN`, etc., as required by the endpoints.

## Commit message

Please refer to the following template for the commit message.

```plaintext
🐞 Bugs and Performance
#🐛 :bug: bug fixes.
#🚑 :ambulance: fix a critical bug
#🚀 :rocket: performance improvements
#💻 Code quality and style
#👍 :+1: feature improvements
#♻️ :recycle: refactoring
#👕 :shirt: Lint error fixes and code style fixes

🎨 UI/UX and design
#✨ :sparkles: add new features
#🎨 :art: design changes only

🛠️ Development Tools and Settings.
#🚧 :construction: WIP (Work in Progress)
#⚙ :gear: config change
#📦 :package: add new dependency
#🆙 :up: update dependency packages, etc.

documentation and comments.
#📝 :memo: fix wording
#📚 :books: documentation
#💡 :bulb: add new ideas and comments

🛡️ security
#👮 :op: security-related improvements

🧪 testing and CI.
#💚 :green_heart: fix/improve testing and CI

🗂️ file and folder manipulation.
#📂 :file_folder: Folder manipulation
#🚚 :truck: file movement

#📊 :log: logging and tracking
#💢 :anger: conflicts
#🔊 :loud_sound: add log
#🔇 :mute: log deleted.
#📈 :chart_with_upwards_trend: add analytics or tracking code

💡 Other.
#🧐 :monocle_face: code reading and questions.
#🍻 :beers: code that was fun to write.
#🙈 :see_no_evil: .gitignore addition.
#🛠️ :hammer_and_wrench: bug fixes and basic problem solving
```
