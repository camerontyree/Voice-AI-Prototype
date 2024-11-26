import ollama
import speech_recognition as sr

# Initialize speech recognizer
r = sr.Recognizer()

# Limit conversation history for faster processing
conversation_history = []

# Define max conversation history length for prompt
MAX_HISTORY_LENGTH = 5

def record_text():
    """Record and return recognized text from speech."""
    try:
        with sr.Microphone() as source2:
            print("Listening...")
            r.adjust_for_ambient_noise(source2, duration=0.1)  # Shorten ambient noise adjustment
            audio2 = r.listen(source2, timeout=5, phrase_time_limit=10)  # Short timeout for faster feedback
            return r.recognize_google(audio2)
    except sr.WaitTimeoutError:
        print("No speech detected. Try again.")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
    except sr.UnknownValueError:
        print("Could not understand audio. Try again.")
    return None

while True:
    # Get user input
    user_input = record_text()
    if not user_input:
        continue

    # Show the user what was recognized
    print(f"User said: {user_input}")

    # Append user input to history
    conversation_history.append(f"User: {user_input}")

    # Trim conversation history to the last MAX_HISTORY_LENGTH exchanges
    if len(conversation_history) > MAX_HISTORY_LENGTH:
        conversation_history.pop(0)

    myPrompt = "\n".join(conversation_history) + "\nAI:"

    print("Generating response...")
    response = ollama.generate(model='llama3.1', prompt=myPrompt)
    actualResponse = response['response']

    # Append chatbot's response to history
    conversation_history.append(f"AI: {actualResponse}")

    # Show the AI's response
    print(f"AI said: {actualResponse}")

    # Reset conversation on specific command
    if user_input.lower() in ["reset", "clear chat"]:
        print("Resetting conversation...")
        conversation_history = []
