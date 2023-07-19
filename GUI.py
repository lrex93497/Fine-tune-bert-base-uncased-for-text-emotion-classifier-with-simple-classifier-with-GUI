import tkinter as tk
from transformers import BertTokenizer
import tkinter.font as font
from transformers import BertModel
import torch
import torch.nn as nn
import re, string
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


def process():
    
    sentence_input = box_text.get("1.0","end-1c")   #get text inside box_text
    sentence_input = sentence_pre_process(sentence_input)   #preprocess text
    tokenized_preprocessed_text_dataloader = sentence_to_token(sentence_input)    # text tokenize and become data loader
    emotion_no = classify(tokenized_preprocessed_text_dataloader)     #get result, flatten value
    #emotion_no [0-5]  is respective to ['joy', 'fear', 'love', 'anger', 'sadness', 'surprise']
    print(emotion_no)
    change_emotion_emoji(emotion_no)
    return

def sentence_pre_process(sentence_input):       #do the preprocess of the text before feed to BertTokenizer
    sentence_out = re.sub(r'\s+', ' ', sentence_input).strip()        # Remove trailing whitespace
    sentence_out = sentence_out.lower()     #make all word become small capital letters
    sentence_out = sentence_out.translate(str.maketrans('', '', string.punctuation))     # remove all punctuation
    return  sentence_out

def sentence_to_token(sentence_input):
    #load berttokenizaer, 12-layer BERT model, with uncased vocabulary
    #tokenize same ways as training
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    input_ids, attention_masks = [], []
    encoded_sent = tokenizer.encode_plus(
        text=sentence_input,                      #text sentence 
        add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
        max_length=150,                  #max lenght sentense input =150
        pad_to_max_length=True,         #Pad sentence to max length
        return_attention_mask=True      #Return attention mask
        )
        
    input_ids.append(encoded_sent.get('input_ids'))
    attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    text_data = TensorDataset(input_ids, attention_masks)
    text_sampler = SequentialSampler(text_data)
    text_dataloader = DataLoader(text_data, sampler=text_sampler, batch_size=1)     #1 sentence batch_size = 1

    return text_dataloader

def change_emotion_emoji(emotion_no):  #it change emotion emoji(default is ? symbol) to respective emoji according to emotion predicted
    #{'anger': 0, 'fear': 1, 'joy': 2, 'love': 3, 'sadness': 4, 'surprise': 5}
    # respective list for emoji word
    #change what emoji to show according to the emoji_list[emotion_no],
    #each list item have respective emoji to the emotion predicted
    emoji_discription = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
    emoji_list = ["\U0001F621","\U0001F631","\U0001F604","\U0001F60D","\U0001F61E","\U0001F632"]      
    
    emotion_label = tk.Label(frame, text=emoji_list[emotion_no]+"\n"+emoji_discription[emotion_no],font=font.Font(size=90))     #set respective emoji
    emotion_label.grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W)     #update label for emoji change
    return

def initialize_model():
    #start classifier
    bert_classifier = emotionClassifier()
    bert_classifier.to(device)  #put classifier into device choosed before, gpu for my case
    return bert_classifier

def classify(preprocessed_text_dataloader):
    for i in preprocessed_text_dataloader:
        batch_input, batch_mask = tuple(v.to(device) for v in i)

        #get result
        with torch.no_grad():
            result = bert_classifier(batch_input, batch_mask)
        classified_result = torch.argmax(result, dim=1).flatten()       #flatten result
    return classified_result

class emotionClassifier(nn.Module):
    def __init__(self):
         super(emotionClassifier, self).__init__()
         self.bert = BertModel.from_pretrained('bert-base-uncased')     #use bert-base-uncased

        #Classifier use hidden neron fc 768 ->48 relu -> 6 (6 label)
         self.classifier = nn.Sequential(
            nn.Linear(768, 48),
            nn.ReLU(),
            nn.Linear(48, 6)
        )
    
    def forward(self, input_ids, attention_mask):
        # input to BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # get [CLS] at last hidden layer output for classifier to use
        cls_from_bert = outputs[0][:, 0, :]

        # put cls_from_bert into our classifier
        result = self.classifier(cls_from_bert)

        return result


if __name__ == "__main__":
    #set up cuda if available, elase use cpu

    if torch.cuda.is_available():       
        device = torch.device("cuda")
        #deafult use gpu0
        print("Use cuda device:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("use cpu")

    bert_classifier = initialize_model()     #initializa model
    loaded_checkpoint = torch.load('checkpoint.pt')
    bert_classifier.load_state_dict(loaded_checkpoint['state_dict'])    #load from checkpoint

    ##below is GUI
    root = tk.Tk()  #create root
    root.title("Emotion classifier")
    root.minsize(600,400)   #min 600*400 window
    tk.Grid.rowconfigure(root, 0, weight=1)
    tk.Grid.columnconfigure(root, 0, weight=1)
    frame=tk.Frame(root)    # frame at root, frame use grid system
    frame.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)

    tk.Grid.rowconfigure(frame, 0, weight=1)
    tk.Grid.rowconfigure(frame, 1, weight=1)
    tk.Grid.columnconfigure(frame, 0, weight=1)
    tk.Grid.columnconfigure(frame, 1, weight=1)


    # creat element
    box_text = tk.Text(frame, height=20, width=30)
    #scrollbar_box_text = tk.Scrollbar(frame, orient=tk.VERTICAL)
    #box_text.configure(yscrollcommand = scrollbar_box_text.set)     #set scrollbar to box_text
    #scrollbar_box_text.config(command=box_text.yview)           #for vertical scrollbar


    emotion_label = tk.Label(frame, text="?",font=font.Font(size=90)) #create a label inside frame , it shows emotion emoji, initially is "?"

    #click this button to do process()  it is inside frame
    button_process = tk.Button(frame, text="Click to get emotion", fg="#ff3343", bg="#fdfd41", font=font.Font(size=22), command=lambda: [process()] )


    #put all element into the frame
    #scrollbar_box_text.grid(row=0, column=0, sticky=tk.S+tk.E+tk.N)
    box_text.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
    emotion_label.grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W)  
    button_process.grid(row=1, column=0, columnspan=2, sticky=tk.N+tk.S+tk.E+tk.W)  

    root.mainloop()     #so GUI can keep on screen