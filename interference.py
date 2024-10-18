from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline



def format_predictions_with_subtokens(text, predictions):
    """
    Formats NER model predictions and combines sub-tokens into a single word
    """
    print(f"Text: {text}\n")
    formatted_output = []
    
    current_word = ""
    current_label = ""
    current_score = 0
    token_count = 0
    
    for prediction in predictions:
        word = prediction['word']
        entity = prediction['entity']
        score = prediction['score']
        
        
        if entity == "LABEL_0":
            label = "O"
        elif entity == "LABEL_1":
            label = "B-MOUNTAIN"
        elif entity == "LABEL_2":
            label = "I-MOUNTAIN"
        else:
            label = entity
        
        
        if word.startswith("##"):
            current_word += word[2:] 
            current_score += score
            token_count += 1
        else:
            
            if current_word:
                avg_score = current_score / token_count 
                formatted_output.append(f"{current_word}: {current_label} (score: {avg_score:.2f})")
            
           
            current_word = word
            current_label = label
            current_score = score
            token_count = 1
    
   
    if current_word:
        avg_score = current_score / token_count
        formatted_output.append(f"{current_word}: {current_label} (score: {avg_score:.2f})")
    
   
    for formatted in formatted_output:
        print(formatted)
    print("\n" + "="*50 + "\n")




def main():
    model = AutoModelForTokenClassification.from_pretrained("Shah1st/mountain-ner-model")
    tokenizer = AutoTokenizer.from_pretrained("Shah1st/mountain-ner-model")

    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

    while True:
        text = input("Enter a sentence, to exit write exit: ")  
        if text.lower() == "exit":          
            break                           
        else:
            predictions = ner_pipeline(text)
            format_predictions_with_subtokens(text, predictions)

        
if __name__ == "__main__":
    main()