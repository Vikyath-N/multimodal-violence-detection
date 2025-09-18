from google import genai
import whisper

model = whisper.load_model("medium")

client = genai.Client(api_key="AIzaSyA160qssryXzXiK_4NAlNPUYUYGowHV49U")

timestamps = [["SPEAKER_01",5.970968750000001,7.03409375],[
                "SPEAKER_02",
                7.03409375,
                8.738468750000003
            ],[
                "SPEAKER_02",
                11.320343750000003,
                14.374718750000003
            ],[
                "SPEAKER_02",
                20.41596875,
                24.870968750000003
            ],[
                "SPEAKER_01",
                22.05284375,
                22.52534375
            ],[
                "SPEAKER_01",
                23.031593750000003,
                23.48721875
            ],[
                "SPEAKER_01",
                23.57159375,
                23.622218750000002
            ],[
                "SPEAKER_01",
                23.723468750000002,
                23.95971875
            ],[
                "SPEAKER_01",
                24.12846875,
                24.16221875
            ],[
                "SPEAKER_02",
                26.00159375,
                28.06034375
            ],[
                "SPEAKER_02",
                28.21221875,
                29.47784375
            ],[
                "SPEAKER_02",
                29.612843750000003,
                29.832218750000003
            ],[
                "SPEAKER_02",
                30.203468750000003,
                31.215968750000002
            ],[
                "SPEAKER_01",
                31.65471875,
                31.688468750000002
            ],[
                "SPEAKER_02",
                31.688468750000002,
                34.48971875
            ],[
                "SPEAKER_02",
                34.59096875,
                37.594718750000006
            ],[
                "SPEAKER_02",
                38.134718750000005,
                38.94471875
            ],[
                "SPEAKER_02",
                39.282218750000006,
                40.463468750000004
            ],[
                "SPEAKER_02",
                40.91909375,
                41.98221875
            ],[
                "SPEAKER_01",
                42.31971875,
                42.775343750000005
            ],[
                "SPEAKER_02",
                43.12971875,
                43.46721875
            ],[
                "SPEAKER_02",
                43.97346875,
                44.766593750000006
            ],[
                "SPEAKER_01",
                45.44159375,
                45.86346875
            ],[
                "SPEAKER_01",
                47.24721875,
                47.314718750000004
            ],[
                "SPEAKER_01",
                50.62221875,
                51.094718750000006
            ],[
                "SPEAKER_01",
                55.27971875,
                55.81971875
            ],[
                "SPEAKER_01",
                59.66721875,
                61.99596875
            ],[
                "SPEAKER_01",
                70.97346875000001,
                76.42409375000001
            ],[
                "SPEAKER_01",
                76.94721875,
                77.79096875
            ],[
                "SPEAKER_01",
                81.84096875,
                82.78596875000001
            ],[
             "SPEAKER_01",
                84.74346875,
                85.03034375
            ],[
                "SPEAKER_01",
                94.66596875,
                96.67409375000001
            ],[
                "SPEAKER_02",
                101.34846875000001,
                102.07409375
            ],[
                "SPEAKER_02",
                106.90034375,
                108.16596875
            ],[
                "SPEAKER_02",
                108.53721875000001,
                110.91659375
            ],[
                "SPEAKER_00",
                111.23721875000001,
                133.96784375000001]]

result = []
conversation = ''

for i in timestamps:
    speaker = i[0]
    start = i[1]
    end = i[2]
    text = model.transcribe("test.wav", clip_timestamps=[start, end])
    # print(text)
    result.append([speaker, text])
    conversation += speaker + ': ' + text["text"]
    print(conversation)

conversation = ''

for i in range(len(result)):
    conversation += result[i][0] + ": " + result[i][1]
    
print(conversation)

prompt = f'''
You are an advanced violence-detection system designed to analyze conversations and identify any violent or abusive language. 
Your task is to:
1. Detect the first instance of violent or abusive language in the provided conversation.
2. Assess its severity based on tone, context, and keywords.
3. Report your findings in the following format:

Output: {{
    speaker: 'Speaker Name',
    score: 'Confidence Score (0-1)',
    description: 'Brief description of the violent or abusive language detected',
    conversation: 'Conversation'
}}

Conversation: {conversation}

Guidelines:
- "Violent or abusive language" includes threats, hate speech, explicit insults, or any language intended to harm or intimidate others.
- Use a confidence score between 0 and 1 to indicate how certain you are about your detection (e.g., 0.9 for high certainty).
- If no violent or abusive language is detected, respond with "No signs of violence detected."
- the conversation extract you found the violent or abusive language in.
'''

response = client.models.generate_content(
    model="gemini-2.0-flash", contents= prompt
)
print(response.text)

