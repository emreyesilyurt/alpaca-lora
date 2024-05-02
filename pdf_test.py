# import sys
# import PyPDF2 

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


# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python generator.py <pdf_file_path>")
#         sys.exit(1)
#     pdf_file_path = sys.argv[1]
#     pdf_text = extract_text_from_pdf(pdf_file_path)
#     print(pdf_text)
#     # while True:
#     #     instruction = input("Please enter your instruction: ")
#     #     generate_text(instruction, input_text=pdf_text)





import sys
import PyPDF2

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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generator.py <pdf_file_path>")
        sys.exit(1)
    pdf_file_path = sys.argv[1]
    pdf_text = extract_text_from_pdf(pdf_file_path)
    print(pdf_text)
