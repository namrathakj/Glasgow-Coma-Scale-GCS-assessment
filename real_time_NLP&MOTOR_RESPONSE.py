import pandas as pd
import spacy
import speech_recognition as sr
import pyttsx3

# Load patient data from Excel file
patient_data = pd.read_excel(r'/home/gcc/Documents/gcsProject/patient_data2.xlsx')
print(patient_data )

# Define the patient name to search for
target_name = 'Rose'

# Find the row with the target patient name
target_row = patient_data[patient_data['patient_name'] == target_name].values.tolist()
print(target_row)

# Define list of questions to ask patient
questions = ['What is your name?', 'How old are you?', 'Where are you?']

# Initialize speech recognition engine and microphone
r = sr.Recognizer()
mic = sr.Microphone()

# Initialize NLP engine
nlp = spacy.load('en_core_web_sm')

def ask_question(question):
    # Initialize text-to-speech engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Adjust speech rate if needed

    target_index = questions.index(question)
    print(target_index,'target_index')
    # Play the question as audio
    engine.say(question)
    engine.runAndWait()

    with mic as source:
        r.adjust_for_ambient_noise(source)  # Adjust for background noise
        audio = r.listen(source, timeout=5) 
        print(audio,'audioooooooo') # Set a timeout of 5 seconds for speech recognition

    # Convert audio to text using speech recognition
    try:
        response = r.recognize_google(audio)
        print(response,'8'*100)
    except sr.UnknownValueError:
        response = ''
    except sr.WaitTimeoutError:
        response = ''  # If the timeout is reached, set an empty response
    
    # Analyze response using NLP
    doc = nlp(response)
    print('responseeeee',response)
    
    if response == target_row[0][target_index]:
        score = 5  # Coherent response
    elif response!= target_row[0][target_index] :
        score = 4  # Disorientation/confusion 
    elif any(token.tag_ == 'PRP' or token.tag_ == 'PRP$' for token in doc):
        score = 3 # Random articulate speech (inappropriate words) 
    elif not doc:
        score = 2  # Incomprehensible sound
    elif not response:
        score = 1  # No sound or no response    
        
    return response, score
    
    
responses = {}
lowest_score = float('inf')  # Initialize lowest_score
for question in questions:
    response, score = ask_question(question)
    responses[question] = response
    print(f'Response score for "{question}": {score}') 
    if score < lowest_score:
        lowest_score = score

print(f"\n verbal response score: {lowest_score}")


import serial
import pandas as pd
import spacy
import speech_recognition as sr
import pyttsx3
import joblib
# Define list of questions to ask patient
questions = ['raise your arm', 'apply pain on shoulder', 'apply pain on elbow']

# Define a list of label-to-score dictionaries for each question
label_to_score_list = [
    {
        'hand_raise': 6,
         #'normal': 1,
    },
    {
        'handsonchest': 5,
        'reflex': 4
    },
    {
        'handsonchest': 3,
        'wrist_twisted': 2,
        'normal': 1
    }
]

# Initialize speech recognition engine and microphone
r = sr.Recognizer()
mic = sr.Microphone()

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speech rate if needed

# Initialize NLP engine
nlp = spacy.load('en_core_web_sm')

# Load the trained model
model = joblib.load('/home/gcc/Documents/gcsProject/random_forest_model.pkl')

# Define the label names
labels = ['hand_raise', 'handsonchest', 'normal', 'reflex', 'wrist_twisted']
# Load the trained model
model = joblib.load('/home/gcc/Documents/gcsProject/random_forest_model.pkl')

for i, question in enumerate(questions):
    # Play the question as audio
    engine.say(question)
    engine.runAndWait()



ser = serial.Serial('/dev/ttyUSB0', 115200)

while True:
    read_serial = ser.readline()
    decoded_serial = read_serial.decode().strip()
    print(decoded_serial)
    if decoded_serial:  # Skip empty lines
        values = decoded_serial.split(',')
        if len(values) == 6:
            data = {
                "acclr(x)": [float(values[0])],
                "acclr(y)": [float(values[1])],
                "acclr(z)": [float(values[2])],
                "gyro(x)": [float(values[3])],
                "gyro(y)": [float(values[4])],
                "gyro(z)": [float(values[5])]
            }
            
            row_to_predict = pd.DataFrame(data)
            print('row_to_predict ',row_to_predict )
            # Rest of the code for model prediction and scoring
            prediction = model.predict_proba(row_to_predict)
            print(prediction)
            predicted_class = prediction.argmax(axis=1)[0]
            predicted_label = labels[predicted_class]
            
            # Map the predicted value to its corresponding GCS score
            label_to_score = label_to_score_list[i]
            
            if i == 1:
                predicted_score = label_to_score.get(predicted_label, 0)  # assign 0 score if label not found
                print(f"Score: {predicted_score}")
            elif i == 2 and predicted_label == 'handsonchest':
                predicted_score = label_to_score.get(predicted_label, 0)  # assign 0 score if label not found
                print(f"Score: {predicted_score}")
            else:
                predicted_score = label_to_score.get('normal', 0)  # assign 0 score if label not found
                print(f"Score: {predicted_score}")

final_score= predicted_score+total_score

# Print the final GCS score
print(f'Final GCS score: {final_score}')

