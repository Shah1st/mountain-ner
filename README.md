# Mountains NER
\documentclass{article}
\usepackage{listings}
\usepackage{hyperref}
\usepackage{amsmath}

\title{Mountain Named Entity Recognition (NER) Model}
\author{}
\date{}

\begin{document}

\maketitle

\section*{Project Overview}
This project involves fine-tuning a BERT-based model (\texttt{dslim/bert-large-NER}) to perform Named Entity Recognition (NER) on mountain names in text. The model has been trained to identify mentions of mountain names and differentiate them from other geographic entities or non-entities.

\subsection*{Features:}
\begin{itemize}
    \item Fine-tuned on a custom dataset that includes sentences both with and without mountain names.
    \item Uses \textbf{focal loss} to handle class imbalance, which ensures the model focuses on correctly classifying rare mountain names.
    \item Token-level classification for identifying the \texttt{B-MOUNTAIN}, \texttt{I-MOUNTAIN}, and \texttt{O} (non-entity) labels.
    \item Balances training between sentences with mountains (80\%) and without mountains (20\%).
\end{itemize}

\section*{Installation}
\begin{enumerate}
    \item Clone the repository:
    \begin{lstlisting}[language=bash]
    git clone https://github.com/yourusername/mountain-ner
    cd mountain-ner
    \end{lstlisting}

    \item Install dependencies: Ensure that you have Python 3.6 or later installed. Then, install the required libraries:
    \begin{lstlisting}[language=bash]
    pip install -r requirements.txt
    \end{lstlisting}
\end{enumerate}

\section*{Usage}

\subsection*{Fine-Tuning the Model}
You can fine-tune the model using the custom dataset by running the script:
\begin{lstlisting}[language=bash]
python train_model.py --output_dir ./model_output --learning_rate 2e-5 --num_train_epochs 5
\end{lstlisting}
\begin{itemize}
    \item \texttt{--output_dir}: Directory to save the fine-tuned model and tokenizer.
    \item \texttt{--learning_rate}: Learning rate for training.
    \item \texttt{--num_train_epochs}: Number of epochs to train the model.
\end{itemize}

\subsection*{Evaluation}
After training, you can evaluate the model on the test dataset:
\begin{lstlisting}[language=bash]
python evaluate_model.py --model_dir ./model_output --dataset_path ./data/test_dataset
\end{lstlisting}

This script will compute metrics like precision, recall, and F1 score for the mountain entity recognition.

\subsection*{Inference}
You can run inference on your own text data to detect mountain names:
\begin{lstlisting}[language=bash]
python inference.py --model_dir ./model_output --input_text "Mount Everest is the tallest mountain in the world."
\end{lstlisting}

This will output the detected mountain names and their corresponding confidence scores.

\section*{Dataset Preparation}
The dataset is split into training, validation, and test sets with an 80:20 ratio between sentences with mountains and those without.

To create the dataset for fine-tuning:
\begin{enumerate}
    \item Prepare the data with both mountain and non-mountain sentences.
    \item Tokenize the sentences and assign \texttt{B-MOUNTAIN}, \texttt{I-MOUNTAIN}, or \texttt{O} labels to each token.
    \item Use the provided \texttt{dataset\_preparation.py} script to format the data for training.
\end{enumerate}

\subsection*{Example Data Format:}
\begin{verbatim}
{
    "id": "12345",
    "tokens": ["Mount", "Everest", "is", "the", "tallest", "mountain", "in", "the", "world", "."],
    "fine_ner_tags": [1, 2, 0, 0, 0, 0, 0, 0, 0, 0]
}
\end{verbatim}

Where:
\begin{itemize}
    \item \texttt{1} = \texttt{B-MOUNTAIN} (beginning of a mountain name),
    \item \texttt{2} = \texttt{I-MOUNTAIN} (inside the mountain name),
    \item \texttt{0} = \texttt{O} (non-mountain entity).
\end{itemize}

\section*{Model Improvements}
Several techniques were applied to improve the model performance, including:
\begin{enumerate}
    \item \textbf{Focal Loss}: This loss function was used to handle class imbalance, ensuring that rare mountain names are given more focus during training.
    \item \textbf{Fine-tuning on Diverse Data}: The dataset includes a wide range of mountain names, including both well-known and rare mountains, to improve generalization.
    \item \textbf{Handling Non-Mountain Entities}: Negative samples, such as geographic names that are not mountains (e.g., rivers, valleys), were introduced to reduce false positives.
\end{enumerate}

\section*{Future Enhancements}
\begin{enumerate}
    \item \textbf{Expand the Dataset}: Adding more examples of rare mountain names and non-mountain geographic entities will improve the model's robustness.
    \item \textbf{Contextual Understanding}: Fine-tuning the model on a dataset with metaphorical and complex usages of mountain names will enhance its ability to understand context.
    \item \textbf{Ensemble Models}: Combining multiple models (e.g., BERT with other architectures) could further improve the accuracy of predictions, especially for rare or ambiguous cases.
\end{enumerate}

\section*{Requirements}
All required libraries are listed in the \texttt{requirements.txt} file. You can install them with:
\begin{lstlisting}[language=bash]
pip install -r requirements.txt
\end{lstlisting}

\subsection*{Key Dependencies:}
\begin{itemize}
    \item \texttt{transformers}
    \item \texttt{datasets}
    \item \texttt{torch}
    \item \texttt{matplotlib}
    \item \texttt{seqeval}
    \item \texttt{numpy}
    \item \texttt{pandas}
\end{itemize}

\section*{Saving and Loading the Model}
To save the fine-tuned model:
\begin{lstlisting}[language=python]
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')
\end{lstlisting}

To load the model for inference:
\begin{lstlisting}[language=python]
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained('./saved_model')
model = AutoModelForTokenClassification.from_pretrained('./saved_model')
\end{lstlisting}

\section*{License}
This project is licensed under the MIT License. See the \texttt{LICENSE} file for more details.

\end{document}
