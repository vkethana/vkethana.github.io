---
layout: post
title: How (and why) I am Starting a Blog
---
(Inspired by [this post](https://guzey.com/personal/why-have-a-blog/) by Alexey Guzey).	

## Why I Started a Blog
### No Downside, Infinite Upside
A few decade ago people were at the mercy of publishing companies and mass media, who would arbitrarily decide what could and couldn't be said. People were discriminated against. Good ideas went unheard. It sucked. 
But the internet has changed things: it's now really easy to get your ideas out there. Aside from purchasing the domain, it's basically free (using hosting service like [Github Pages](https://pages.github.com/) or [Netlify](https://www.netlify.com/)). 
### Even Repeated Content is Fine
Even sites that mostly repeat what other people say can be valuable. My thinking is that as I write more and more, my own ideas will start to emerge without me even trying. Every good idea has already been said, but since nobody was listening the first time around, someone else will have to say it again. 
### Ripple Effects
One person starting a blog can trigger other people to do the same. In my case, some of my inspirations include [https://www.arjunkhemani.com/about](Progress Good) (philosophy of science, economics, education), [Gwern](https://gwern.net/index) (machine learning, metascience, and a ton of other stuff), and [Matt Might](https://matt.might.net/) (theoretical computer science, precision medicine, free software).
My hope is that starting this website will encourage other people to do the same thing, and we can reverse the development of "web feudalism": hosting your entire online presence underneath a larger platform like Substack, Twitter, or Reddit. These platforms aren't bad -- they are probably essential for reaching large audiences -- but they should coexist alongside an independent, decentralized web.
## Implementation
Originally, my website used a handwritten Python script which took a directory of markdown files, stored in a folder titled `md/`, and converted them into a folder of HTML files called `output/`.
The script was straightfoward (ChatGPT wrote part of it for me). In fact, it boiled down to just a few lines of code: 
```python
import markdown

def md_to_html(md_file):
    # Read markdown content from file 
		# (Script assumes that blog content is written in Markdown)
    with open(md_file, 'r') as f:
        md_content = f.read()

    # Convert markdown to HTML
    html_content = markdown.markdown(md_content)

    return html_content
```
Then I inserted the HTML output into a pre-made template file which had some extra amenities -- a header, a footer, CSS styling...:
```python
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
```
This approach worked fine, but it got clunky when I wanted to add multiple templates or insert figures/images. 
### Introducing Jekyll
Jekyll fixes all these problems -- easy templating, a large plugin ecosystem, the ability to insert HTML into Markdown files (a godsend if you're trying to include graphs / images), you name it. The only price you pay is that you have to work with Ruby version management which is, well, messy. For example, I spent a whole afternoon trying to fix my buggy Homebrew installation of Ruby, which was preventing me from installing 3rd-party plugins. As far as I understand, Jekyll works best when you build up your site from scratch instead of using pre-made templates which deprecate quickly and often break mysteriously. 
