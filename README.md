# T5 Bible Styling

This project fine-tunes a pre-trained T5 (Text-to-Text Transfer Transformer) model for the specific task of converting normal text into KJV-style English.

## Project Structure

- **data/kjv.csv**: Contains the KJV (King James Version) and WEB (World English Bible) Bible text used for fine-tuning the model.
- **main.ipynb**: Jupyter Notebook for loading the T5 model, preprocessing the data, and performing fine-tuning.
- **get_data.ipynb**: Jupyter Notebook for loading in the data.

## Requirements

- Python 3.11
- All libraries in **requirements.txt**

## Setup Instructions

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd Fine-Tuned-Transformer
   ```

2. Create and activate a virtual environment:

   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Follow the steps in the notebook to:

   - Load the pre-trained T5 model.
   - Preprocess the KJV dataset.
   - Fine-tune the model for your specific task.

2. Save the fine-tuned model for inference or further use.

Alternatively, you can use the `train.py` file to run it through the command line. Try using commands like the following to run as a background process: `nohup python -u train.py > output.log 2>&1 &`.

## Notes

- If you encounter a warning about `IProgress` not being found, ensure that `ipywidgets` is installed and up-to-date. Or just ignore it.
- The datasets (`data/kjv.csv`) and (`data/web.csv`) contain the text of the Bible for their respective versions, which is used for training and evaluation. (`data/web_to_kjv`) has these two versions side-by-side.

## License

This project is for educational purposes. Ensure compliance with any applicable licenses for the dataset and pre-trained models.
