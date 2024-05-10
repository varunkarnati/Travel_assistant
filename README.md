# Travel_assistant
A language-aware assistant powered by LangChain, LangGraph, and OpenAI. Handles tasks like ride booking, flight details, and more. Efficient and intelligent conversational experience.


This repository houses the code for a language-aware assistant powered by LangChain, LangGraph, and OpenAI. The assistant intelligently handles various tasks such as ride booking, flight details, and more. At its core, LangGraph enables seamless state management, allowing the assistant to understand user context and provide accurate responses.

## How It Works

LangGraph plays a crucial role in orchestrating the assistant's functionality. It manages the state of the conversation, allowing for dynamic transitions between user inputs and tool invocations. By leveraging LangGraph's capabilities, the assistant can maintain context across multiple interactions, ensuring a smooth and intuitive conversational experience.

## Features

- **Structured Tools**: Utilizes structured tools for performing tasks such as ride booking and flight information retrieval.
- **OpenAI Language Model**: Harnesses the power of the OpenAI language model to understand and generate natural language responses.
- **Efficient State Management**: LangGraph facilitates efficient state management, enabling seamless transitions between user inputs and tool invocations.
- **Intelligent Responses**: Provides intelligent and contextually relevant responses to user queries, enhancing the conversational experience.

## Usage

1. **Clone the repository** to your local machine:
    ```bash
    gh repo clone varunkarnati/Travel_assistant
    ```

2. **Install the required dependencies**:
    ```bash
    pip install langchain langchain_openai
    ```

3. **Set up the necessary environment variables**:
    ```bash
    export OPENAI_API_KEY="<your-openai-api-key>"
    ```

4. **Run the Python script**:
    ```bash
    python <script-name>.py
    ```

5. **Interact with the assistant** by typing your queries and observing the intelligent responses.

## Contributing

Contributions are welcome! Feel free to open issues or pull requests for any improvements or additional features you'd like to see.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
