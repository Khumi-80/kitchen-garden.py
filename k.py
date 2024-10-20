!pip install transformers
# 1. Install Hugging Face's transformers library
!pip install transformers

# 2. Import necessary modules
from transformers import pipeline

# 3. Load the pre-trained question-answering model
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# 4. Define the gardening knowledge base (you can expand this with more data)
context_data = """
In Punjab (Pakistan and India), the month of October is suitable for planting cool-season crops such as carrots, spinach, lettuce, radishes, and peas.
Coriander and fenugreek can also be planted in warmer areas. Tomatoes and onions are best planted in early winter.
Ensure regular watering and well-drained soil for all winter crops.
Avoid planting frost-sensitive crops until temperatures stabilize.
Other suitable vegetables for the season include garlic, turnips, and beets.
"""

# 5. Function to use the pre-trained model for answering questions
def ask_garden_bot(question):
    # Get the model's answer from the context
    result = qa_model(question=question, context=context_data)
    return result['answer']

# 6. Interactive loop to ask questions
while True:
    # Get user input
    user_question = input("Ask a question about your kitchen garden (or type 'exit' to stop): ")
    
    # Exit condition
    if user_question.lower() == "exit":
        print("Goodbye!")
        break
    
    # Get the bot's response and print it
    response = ask_garden_bot(user_question)
    print("GardenBot:", response)
