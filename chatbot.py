from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []

while True:    
    # Create conversation history
    history_string = "\n".join(conversation_history)

    # Get prompt from user
    input_text = input("> ")

    # Tokenize the input text and text history
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    #Generate the response from the model
    outputs = model.generate(**inputs)

    # Decode reponse
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    print(response)

    # Update conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)
