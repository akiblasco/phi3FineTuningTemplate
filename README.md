# Phi3 pre-trained model Fine-tuning template
This is a template for pretraining language models. In this particular case, it is a template using Microsoft's Phi-3 pre-trained Model that I wanted to fine-tune for quering specific text based knowledge bases. For those who are unfamiliar with this model, this is the description on the Hugging Face site:

## Model Summary
The Phi-3-Mini-128K-Instruct is a 3.8 billion-parameter, lightweight, state-of-the-art open model trained using the Phi-3 datasets. This dataset includes both synthetic data and filtered publicly available website data, with an emphasis on high-quality and reasoning-dense properties. The model belongs to the Phi-3 family with the Mini version in two variants 4K and 128K which is the context length (in tokens) that it can support.

After initial training, the model underwent a post-training process that involved supervised fine-tuning and direct preference optimization to enhance its ability to follow instructions and adhere to safety measures. When evaluated against benchmarks that test common sense, language understanding, mathematics, coding, long-term context, and logical reasoning, the Phi-3 Mini-128K-Instruct demonstrated robust and state-of-the-art performance among models with fewer than 13 billion parameters. Resources and Technical Documentation:

For more information you can consult instalation and more information here: https://huggingface.co/microsoft/Phi-3-mini-128k-instruct 


## Fine-Tuning

The project was originally made in Google Colab, as a Jupyter notebook in order to external GPUs for computationally heavy tasks such as training the model. However, if one does not have paid services, the time of usage of Google Colab's resources is very limited. I had to complete this task locally on my M2 Apple Silicon MacBook Pro. I plan to add a more general and update version for standarized local GPU usage in the future.

## Fine-Tuning 

1. **Prepare your dataset**: Ensure you have a dataset relevant to your task or domain. This can be in formats like CSV, JSON, or other structured text formats. It is also possible to pass large texts in pdf formats and read PDF files from the web if further function implementation is done.

2. **Fine-tuning script**: This is a fine tuning script example based off the Hugging Face example, and the 'Phi-3 CookBook' repository, the link to it both will be attached below. This is a very general example, but it has some parameter tuning I performed in order to produced desired and consistent results.
    ```python
    # Load the model
    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("path_to_save_your_model")
    tokenizer.save_pretrained("path_to_save_your_model")

    ```

3. **Use the fine-tuned model**:
   
After fine-tuning, you can load and use the fine-tuned model for your specific tasks. Feel free to fine-tune to tailor any specific needs!


## Technical Details

This project utilizes the Hugging Face Transformers library to load, interact with, and fine-tune the Phi-3 model. The model is pre-trained on a diverse dataset, enabling it to provide accurate and contextually relevant responses. There currently is a lot of dependency issues between libraries and specific versions.

### Notes

- **Model Loading**: The Phi-3 model and tokenizer are loaded using the `AutoModelForCausalLM` and `AutoTokenizer` classes from the Transformers library.
- **Apple Silicone Chip**: I am working on an M2 MacBook, which made this process a lot harder than expected when trying to run locally. As I mentioned before, due to being a free user of Google Colab, the time of usage of hardware is limited, therefore I had to adjust this script to run on CPU (I know, horrible, but it is what I have), because M1 and M2 macbooks do not have an Nvidia GPU, and thus not support any of the modules, dependencies and so on and so forth.
- **Text Generation Pipeline**: A pipeline is created for text generation, facilitating the interaction with the model.
- **Fine-Tuning Framework**: The fine-tuning script leverages techniques like LoRA to adapt the model for specific tasks or domains.

### Links
Phi-3 CookBook: https://github.com/microsoft/Phi-3CookBook/blob/main/md/04.Fine-tuning/FineTuning_Lora.md
Hugging Face phi-3 documentation: https://huggingface.co/microsoft/Phi-3-mini-128k-instruct
