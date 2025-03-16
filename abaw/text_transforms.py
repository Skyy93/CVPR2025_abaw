import re
import nltk
import torch
import random

from dataclasses import dataclass
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from deep_translator import GoogleTranslator
from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer

## Special Tokens
@dataclass
class SpecialTokens:
    #start: str = "[START]"
    #end: str = "[END]"
    time_attention: str = "[ATTENTION]"

    def generate_special_token_dict():
        return {'additional_special_tokens': [SpecialTokens.time_attention]} #, SpecialTokens.start, SpecialTokens.end, ]}

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

class TextAugmentor:
    def __init__(self, enable, device, prob_synonym=0.01, prob_deletion=0.01, prob_swap=0.00, prob_bert=0.025):
        """
        Initializes the TextAugmentor with probabilities for each augmentation method.
        """
        self.prob_synonym = prob_synonym
        self.prob_deletion = prob_deletion
        self.prob_swap = prob_swap
        self.prob_bert = prob_bert
        self.enable = enable

        if self.enable:

            mask_text_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(device)
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

            if device == "cpu":
                self.fill_mask = pipeline("fill-mask", model=mask_text_model, tokenizer=tokenizer, device=-1)
            else:
                self.fill_mask = pipeline("fill-mask", model=mask_text_model, tokenizer=tokenizer)

            # List of languages for back-translation (Google Translate codes)
            self.backtrans_languages = [
                                        "fr", "de", "es", "it", "nl", "ru", "pt", "sv", "pl", "ja", 
                                        "zh-cn", "zh-tw", "ko", "ar", "tr", "fi", "el", "cs", "hu", "hi",  
                                        "no", "da", "uk", "ro", "bg", "hr", "th", "vi", "id", "ms"  
                                        ]
        
    def synonym_replacement(self, sentence, n=1):
        """ Replaces random words with their synonyms using WordNet. """        
        if not isinstance(sentence, str):  
            raise TypeError(f"Expected a string, but got {type(sentence)}")  
        
        words = word_tokenize(sentence)
        new_words = words.copy()
        candidates = [word for word in words if wordnet.synsets(word)]

        if not candidates:
            return sentence

        for _ in range(n):
            word = random.choice(candidates)
            synonyms = {lemma.name().replace("_", " ") for syn in wordnet.synsets(word) for lemma in syn.lemmas()}  
            synonyms.discard(word)

            if synonyms:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if w == word else w for w in new_words]

        return " ".join(new_words)

    def random_deletion(self, sentence, p=0.2):
        """ 
        Randomly removes words with probability `p`.
        Ensures at least one word remains to prevent empty output.
        """
        words = sentence.split()
        
        # If there's only one word, return it (avoid empty sentence)
        if len(words) <= 1:
            return sentence  

        # Apply deletion but ensure at least one word remains
        new_words = [word for word in words if random.uniform(0, 1) > p or word == SpecialTokens.time_attention]

        # If all words were removed, keep one random word
        if not new_words:
            new_words = [random.choice(words)]  # Select a random word from original

        return " ".join(new_words)

    def random_swap(self, sentence, n=1):
        """ 
        Swaps two random words in the sentence.
        """
        words = sentence.split()
        if len(words) < 2:
            return sentence  # Avoid unnecessary swaps
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            if (words[idx1] != SpecialTokens.time_attention) and (words[idx2] != SpecialTokens.time_attention):
                words[idx1], words[idx2] = words[idx2], words[idx1]
        return " ".join(words)

#    def back_translate(self, sentence, lang='fr'):
#        """ 
#        Translates the sentence into another language and back to create variation.
#        """
#        try:
#            translated = GoogleTranslator(source='auto', target=lang).translate(sentence)
#            back_translated = GoogleTranslator(source=lang, target='en').translate(translated)
#            return back_translated
#        except:
#            return sentence  # Return original text if translation fails

    def bert_augmentation(self, sentence, n=1):
        """ 
        Uses BERT to replace `n` randomly selected words with [MASK] tokens.
        Also places `[MASK]` between words with an 85% probability.
        Ensures all `[MASK]` tokens are replaced sequentially with BERT predictions.
        Filters out punctuation in both `[MASK]` placement and BERT predictions.
        """
        words = sentence.split()
        if len(words) < 2:
            return sentence  # Avoid augmenting very short sentences

        # Function to check if a word is a valid token (not punctuation)
        def is_valid_word(word):
            return bool(re.match(r"^[a-zA-Z0-9\-]+$", word))  # Allow words, numbers, and hyphens

        # Filter only valid words for replacement
        valid_indices = [i for i, word in enumerate(words) if is_valid_word(word)]

        if not valid_indices:  # If no valid words, return original sentence
            return sentence

        masked_sentence = words.copy()

        for _ in range(n):
            if random.uniform(0, 1) < 0.75:
                # 75% chance to place a `[MASK]` between words
                insert_idx = random.randint(0, len(masked_sentence)-1)  # Allow inserting at start or end
                masked_sentence.insert(insert_idx, "[MASK]")
            else:
                # Otherwise, replace a valid word with `[MASK]`
                replace_idx = random.choice(valid_indices)
                if masked_sentence[replace_idx] != SpecialTokens.time_attention:
                    masked_sentence[replace_idx] = "[MASK]"

        masked_sentence = " ".join(masked_sentence)

        # Use BERT to predict the masked words
        predictions = self.fill_mask(masked_sentence)

        # Ensure predictions is a list of lists (one per `[MASK]` token)
        if isinstance(predictions, list) and all(isinstance(p, list) for p in predictions):
            for _ in range(5):  # Perform 5 full replacement cycles
                prediction_index = 0  # Reset prediction index after each loop

                while "[MASK]" in masked_sentence and prediction_index < len(predictions):
                    predicted_word = predictions[prediction_index][0]["token_str"]  # Get best prediction

                    # Ensure the predicted word is valid (not punctuation)
                    if not is_valid_word(predicted_word):
                        # Try another prediction if available
                        for alt_pred in predictions[prediction_index]:
                            if is_valid_word(alt_pred["token_str"]):
                                predicted_word = alt_pred["token_str"]
                                break
                        else:
                            # If all predictions are bad, use a random valid word
                            predicted_word = random.choice([word for word in words if is_valid_word(word)])

                    masked_sentence = masked_sentence.replace("[MASK]", predicted_word, 1)  # Replace one `[MASK]`
                    prediction_index += 1

        # If there are still `[MASK]` tokens, replace them randomly
        while "[MASK]" in masked_sentence:
            random_word = random.choice([word for word in words if is_valid_word(word)])  # Choose a valid word
            masked_sentence = masked_sentence.replace("[MASK]", random_word, 1)

        return masked_sentence

    def augment(self, sentence):
        """ 
        Applies augmentations based on predefined probabilities in a random order.
        """
        if not self.enable:
            return sentence

        # Store the original sentence as a fail-safe
        original_sentence = sentence  

        # Ensure word_count is at least 1 to prevent division errors
        word_count = max(1, len(sentence.split()))

        # Fail-safe `n_*` values to prevent `randrange()` errors
        n_synonym = 1 
        p_deletion = random.uniform(0.075, 0.175)
        n_swap = 1 # random.randint(1, min(3, max(1, word_count - 1)))  # Swap max 3, or up to word count - 1
        language = random.choice(self.backtrans_languages)
        n_masks = random.randint(1, max(2, word_count//4)) if word_count > 1 else 1
        #TODO: best                             ->  word_count//4

        augmentations = [
            (self.prob_synonym, lambda s: self.synonym_replacement(s, n_synonym)),
            (self.prob_deletion, lambda s: self.random_deletion(s, p_deletion)),
            (self.prob_swap, lambda s: self.random_swap(s, n_swap)),
            #(self.prob_backtrans, lambda s: self.back_translate(s, language)),
            (self.prob_bert, lambda s: self.bert_augmentation(s, n_masks))
        ]

        # Shuffle the augmentations to apply them in a random order
        random.shuffle(augmentations)

        for prob, func in augmentations:
            if random.uniform(0, 1) < prob:
                # Apply the augmentation
                sentence = func(sentence)  
            # Fail-safe: If the sentence is empty after augmentation, revert to the original sentence
            if not isinstance(sentence, str) or not sentence.strip():          
                sentence = original_sentence

        return sentence

# Example usage
if __name__ == "__main__":
    text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    augmentor = TextAugmentor(text_tokenizer=text_tokenizer, device="cuda")
    text = "Hashtag no filter. I mean, I thought that was kind of annoying. I mean, you use the filter."
    augmented_text = augmentor.augment(text)
    # augmented_text = augmentor.bert_augmentation(text, 2)
    print(augmented_text)