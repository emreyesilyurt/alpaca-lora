import PyPDF2  # or pdfplumber
import io
import sys
import os
import fire
import torch
import warnings
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from peft import PeftModel
from utils.prompter import Prompter

warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  
    pass

# def extract_text_from_pdf(pdf_file_path):
#     text = ""
#     # Open the PDF file
#     with open(pdf_file_path, "rb") as file:
#         # Create a PDF reader object
#         pdf_reader = PyPDF2.PdfReader(file)
#         # Iterate through each page in the PDF
#         for page_num in range(len(pdf_reader.pages)):
#             # Get the text content of the page
#             page = pdf_reader.pages[page_num]
#             text += page.extract_text()
#     return text


def extract_text_from_pdf(pdf_file_path):
    text = ""
    # Open the PDF file
    with open(pdf_file_path, "rb") as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Get the text content of the first page
        first_page = pdf_reader.pages[0]
        text += first_page.extract_text()
        
    return text


def generate_text(
    instruction,
    input_text=None,
    load_8bit=False,
    base_model="huggyllama/llama-7b",
    lora_weights="tloen/alpaca-lora-7b",
    prompt_template="",
    stream_output=False,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # Unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # Seems to fix bugs for some users.

    model.eval()

    def evaluate(**kwargs):
        prompt = prompter.generate_prompt(instruction, input_text)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(**kwargs)

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        if stream_output:
            with torch.no_grad():
                for output in model.generate(**generate_params):
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield prompter.get_response(decoded_output)
            return

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield prompter.get_response(output)

    response_generator = evaluate(max_new_tokens=128)  # Set max_new_tokens here

    for response in response_generator:
        print(response)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generator.py <pdf_file_path>")
        sys.exit(1)
    pdf_file_path = sys.argv[1]
    pdf_text = extract_text_from_pdf(pdf_file_path)
    print(pdf_text)
    while True:
        instruction = input("Please enter your instruction: ")
        generate_text(instruction, input_text=pdf_text)
