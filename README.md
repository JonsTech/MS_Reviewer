# MS Reviewer

A management summary reviewer using Ollama and structured JSON personas.

## Requirements
- **Ollama:** Must be installed and running.
- **Model:** Default is `gpt-oss:20b`, but you can swap this to any model of your choice using environment variables.

## How to Run
1. Install dependencies: `npm install`
2. cd into reviewer
```bash
cd reviewer
```
2. Start the server:
   ```bash
   node server.js
   ```
3. Open in your browser: [http://localhost:3000](http://localhost:3000)