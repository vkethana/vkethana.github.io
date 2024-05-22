---
layout: post
title: Why (and How) I am Starting a Website 
description: My brief essay on why the internet needs more independent blog sites run by independent tinkerers.
tags: blog, code, web 
---

The reason I'm starting a website is simple: I want to join the ranks of all the other cool writers on the internet. Some of my inspirations include:
- [Gwern](https://gwern.net/index) (machine learning, the scientific method, and a ton of other stuff)
- [Progress Good](https://www.arjunkhemani.com/about) (philosophy of science, economics, education)
- [ftlsid](https://ftlsid.com) (learning Japanese, meditation)
- [Visakan Veerasamy](https://visakanv.com) (how social networks and innovation work)
- [Alexey Guzey](https://guzey.com/) (metascience, philosophy)
- [Matt Might](https://matt.might.net/) (theoretical computer science, medicine, free software)

The longer answer is, I think that the internet is moving in the wrong direction. We're seeing the rise of "web feudalism": hosting your entire online presence underneath a larger platform like Substack, Twitter, or Reddit. These platforms aren't bad -- you need them to reach large audiences -- but they should coexist alongside independently-owned websites.

Some of content on my site is just a repetition of stuff said elsewhere. But right now, I don't think that's an issue. My reasoning is that as I write more and more, my own ideas will start to emerge without me even trying. Every good idea has already been said, but since nobody was listening the first time around, someone else will have to repeat it.

# How my Website Works
## Why my Original Website Sucked
Originally, my website used a handwritten Python script which took a directory of markdown files, stored in a folder titled `md/`, and converted them into a folder of HTML files called `output/`.
The script was straightfoward and basically boiled down to just a few lines of code: 
```python
# Import library that converts markdown text into HTML
import markdown

def md_to_html(md_file):
    # Read markdown content from file (script assumes that blog content is written in Markdown)
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
    # First, look for the "checkpoint" where the main content should go
    main_content_index = template_content.find("<!-- Main content -->")
    if main_content_index != -1:
        output_content = (
            template_content[:main_content_index] +
            "<!-- Main content -->" +
            content +
            template_content[main_content_index:]
        )
    else:
        # If the template file breaks for some reason, the code will still work
        output_content = template_content + "\n" + content

    return output_content
```
While this script was fun to write (and tell other people about), it had too many problems. 
What if you want to have multiple templates? 
What if you want a page that automatically lists out all the blog posts on your site? 
What if you want to insert interactive content into your Markdown files? 
Instead of coming up with ad-hoc solutions for each of these issues, I discovered a tool that does it all for me.
## Introducing Jekyll
Jekyll fixes all these problems -- easy templating, a large plugin ecosystem, the ability to insert HTML into Markdown files (a godsend if you're trying to include graphs / images with captions), you name it. The only price you pay is that you have to deal with Ruby version management, which is almost guaranteed to be frustrating the first time you install the language. For example, I almost gave up on making the site because my Homebrew version of Ruby kept conflicting with dependencies for a custom theme I wanted to use. 

I've found that Jekyll works a lot better when you build up your site from scratch. Premade templates tend to become deprecated, and you run the risk of something mysteriously breaking. For my website, I used the theme [dark-poole](https://andrewhwanpark.github.io/dark-poole/) and adjusted it to my liking. 

(Inspired by [this essay](https://guzey.com/personal/why-have-a-blog/) by Alexey Guzey).	

<script src="https://giscus.app/client.js"
        data-repo="vkethana/vkethana.github.io"
        data-repo-id="R_kgDOLBRagA"
        data-category="Announcements"
        data-category-id="DIC_kwDOLBRagM4Cfi2H"
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="bottom"
        data-theme="noborder_light"
        data-lang="en"
        data-loading="lazy"
        crossorigin="anonymous"
        async>
</script>
