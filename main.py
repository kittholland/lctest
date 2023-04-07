print("Hello! What's your name?")
name = input("> ")
print(f"Nice to meet you, {name}!")

while True:
    message = input("What can I help you with today? (type 'quit' to exit)> ")
    
    if message.lower() == "quit":
        print("Goodbye!")
        break