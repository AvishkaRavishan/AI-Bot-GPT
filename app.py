from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import openai

# Load the model and tokenizer for the question-answering task
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad" 
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

def make_predictions(context, question):
    try:
        inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
        outputs = model(**inputs)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        answer_start = torch.argmax(start_logits)
        answer_end = torch.argmax(end_logits) + 1
        answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])
        return answer
    except Exception as e:
        return f"Error making predictions: {str(e)}"

def get_llm_response(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # model engine
            prompt=prompt,
            max_tokens=150,  # number of tokens as needed
            n=1,
            stop=None,
            temperature=0.7  # temperature for more or less creativity
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error getting LLM response: {str(e)}"

context = '''
Welcome to MacHealth, your all-in-one healthcare management platform. Our AI-powered chatbot is here to help with all your healthcare needs. Whether you are a patient, doctor, or hospital admin, our chatbot makes everything easy and efficient.

Patients can book appointments with their preferred doctors and get reminders. Doctors can manage their schedules and see patient appointments. Patients can also manage and access medical records for themselves and their families, while doctors can view these records and shared information.

You can search for medications and find nearby pharmacies where they are available. Our AI model, Dr. Codrive, provides personalized predictions of future health risks, answers personal health questions confidentially, and gives advice on necessary checkups and care.

Doctors can connect with other professionals to share insights, and hospitals can integrate their systems into our platform to manage operations more efficiently. 

Stay tuned for more features and improvements as we continue to develop MacHealth. We aim to provide a user-friendly platform for all your healthcare needs.

Type your question below, and let our chatbot assist you. Your health and convenience are our top priorities. For emergencies or serious health concerns, please contact your healthcare provider directly.
'''

def main():
    while True:
        question = input("Q: ")
        if question.lower() in ["exit", "quit"]:
            break

        # First, try to answer with the question-answering model
        qa_answer = make_predictions(context, question)

        # If the QA model cannot find a good answer, use the LLM
        if not qa_answer or qa_answer.lower() in ["i don't know", "i'm not sure", "", "Error making predictions:"]:
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            llm_answer = get_llm_response(prompt)
            print("LLM answer:", llm_answer)
        else:
            print("Predicted answer:", qa_answer)

if __name__ == "__main__":
    main()
