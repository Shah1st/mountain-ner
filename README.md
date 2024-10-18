# Mountain Named Entity Recognition (NER) Model

## Project Overview
This project involves fine-tuning a BERT-based model (`dslim/bert-large-NER`) to perform Named Entity Recognition (NER) on mountain names in text. The model has been trained to identify mentions of mountain names and differentiate them from other geographic entities or non-entities.

### Features:
- Fine-tuned on a custom dataset that includes sentences both with and without mountain names.
- Uses **focal loss** to handle class imbalance, which ensures the model focuses on correctly classifying rare mountain names.
- Token-level classification for identifying the `B-MOUNTAIN`, `I-MOUNTAIN`, and `O` (non-entity) labels.
- Balances training between sentences with mountains (80%) and without mountains (20%).

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/mountain-ner
    cd mountain-ner
    ```
2. Install dependencies: Ensure that you have Python 3.6 or later installed. Then, install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Fine-Tuning the Model
You can fine-tune the model using the custom dataset by running the script:
```bash
python train.py --output_dir ./model_output --learning_rate 2e-5 --num_train_epochs 5
```

### Inference
You can run inference on your own text data to detect mountain names:
```bash
python inference.py --model_dir ./model_output --input_text "Mount Everest is the tallest mountain in the world."
```


This will output the detected mountain names and their corresponding confidence scores.

## Dataset Preparation
The dataset is split into training, validation, and test sets with an 80:20 ratio between sentences with mountains and those without.

To create the dataset for fine-tuning:
1. Prepare the data with both mountain and non-mountain sentences.
2. Tokenize the sentences and assign `B-MOUNTAIN`, `I-MOUNTAIN`, or `O` labels to each token.
3. Use the provided `dataset_preparation.py` script to format the data for training.

### Example Data Format:
```json
{
    "id": "12345",
    "tokens": ["Mount", "Everest", "is", "the", "tallest", "mountain", "in", "the", "world", "."],
    "fine_ner_tags": [1, 2, 0, 0, 0, 0, 0, 0, 0, 0]
}
```


Where:
- `1` = `B-MOUNTAIN` (beginning of a mountain name),
- `2` = `I-MOUNTAIN` (inside the mountain name),
- `0` = `O` (non-mountain entity).

## Model Improvements
Several techniques were applied to improve the model performance, including:
1. **Focal Loss**: This loss function was used to handle class imbalance, ensuring that rare mountain names are given more focus during training.
2. **Fine-tuning on Diverse Data**: The dataset includes a wide range of mountain names, including both well-known and rare mountains, to improve generalization.
3. **Handling Non-Mountain Entities**: Negative samples, such as geographic names that are not mountains (e.g., rivers, valleys), were introduced to reduce false positives.

## Future Enhancements
1. **Expand the Dataset**: Adding more examples of rare mountain names and non-mountain geographic entities will improve the model's robustness.
2. **Contextual Understanding**: Fine-tuning the model on a dataset with metaphorical and complex usages of mountain names will enhance its ability to understand context.
3. **Ensemble Models**: Combining multiple models (e.g., BERT with other architectures) could further improve the accuracy of predictions, especially for rare or ambiguous cases.

## Requirements
All required libraries are listed in the `requirements.txt` file. You can install them with:
```bash
pip install -r requirements.txt
```

## Saving and Loading the Model
To save the fine-tuned model:
```python
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')
```

To load the model for inference:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained('./saved_model')
model = AutoModelForTokenClassification.from_pretrained('./saved_model')

```

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.





