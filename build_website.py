import sys
import markdown
from bs4 import BeautifulSoup
import os

filenames_to_titles = {
  "index.html": "Vijay Kethanaboyina",
  "reading_log.html": "Vijay's Reading Log",
  "writings.html": "Writings",
  "beginning_of_infinity_summary.html": "Summarizing the Beginning of Infinity",
  "1.html": "1",
  "2.html": "2"
}

def md_to_html(md_file):
    # Read markdown content from file
    with open(md_file, 'r') as f:
        md_content = f.read()

    # Convert markdown to HTML
    html_content = markdown.markdown(md_content)

    return html_content

def insert_content_into_template(template_file, content):
    # Read template content
    with open(template_file, 'r') as f:
        template_content = f.read()

    # Insert content into template
    main_content_index = template_content.find("<!-- Main content -->")
    if main_content_index != -1:
        output_content = (
            template_content[:main_content_index] +
            "<!-- Main content -->" +
            content +
            template_content[main_content_index:]
        )
    else:
        output_content = template_content + "\n" + content

    return output_content

def edit_header_content(html_content, new_title):
    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find <h1> tag with id starting with arbitrary_id
    h1_tag = soup.find('h1', id=lambda x: x and x.startswith('header'))
    assert(h1_tag)
    if h1_tag:
        # Replace content with file name
        h1_tag.string = new_title
    return str(soup)

def convert_file(md_file):
    output_file = md_file.replace('.md', '.html')

    # removes the prefix on the file path
    # e.g. "md/reading_log.md" -> "reading_log.html"
    output_file = os.path.basename(output_file)

    html_content = md_to_html(md_file)
    html_content = insert_content_into_template("template/template.html", html_content)
    html_content = edit_header_content(html_content, filenames_to_titles[output_file])

    # Output HTML to console or save to file
    # If you want to save to file, you can uncomment the following lines:
    output_file = "html/" + output_file
    with open(output_file, 'w') as f:
         f.write(html_content)
    print(f"Saved contents to {output_file}")

def main():
    directory = "md" # Directory with markdown files

    # Iterate over all files in the directory
    for filename in os.listdir("md"):
      # Check if the file ends with ".md"
      if filename.endswith(".md"):
          # Full path to the file
          file_path = os.path.join(directory, filename)
          print(f"Converting {file_path}")
          try:
            convert_file(file_path)
          except Exception as e:
            print("Error converting {file_path}: {e}")
          else:
            print(f"Successfully converted {file_path}\n")

if __name__ == "__main__":
    main()
