import sys
import markdown

def md_to_html(md_file):
    # Read markdown content from file
    with open(md_file, 'r') as f:
        md_content = f.read()

    # Convert markdown to HTML
    html_content = markdown.markdown(md_content)

    return html_content

def main():
    if len(sys.argv) != 2:
        print("Usage: python md_to_html.py <markdown_file>")
        sys.exit(1)

    md_file = sys.argv[1]
    html_content = md_to_html(md_file)

    # Output HTML to console or save to file
    # If you want to save to file, you can uncomment the following lines:
    # output_file = md_file.replace('.md', '.html')
    # with open(output_file, 'w') as f:
    #     f.write(html_content)
    print(html_content)

if __name__ == "__main__":
    main()

