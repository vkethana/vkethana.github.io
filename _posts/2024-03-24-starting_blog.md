---
layout: post
title: Why (and How) I am Starting a Website 
description: My brief essay on why the internet needs more independent blog sites run by independent tinkerers.
tags: blog, code, web 
published: false
---

## 1. No Downside, Infinite Upside
The internet is a pretty new thing. A few decades ago, people were at the mercy of publishing companies and mass media, who would arbitrarily decide what could and couldn't be said. People were discriminated against. Good ideas went unheard. It sucked. 

But the internet has changed things: it's now really easy to get your ideas out there. Aside from purchasing the domain, it's basically free (using hosting service like [Github Pages](https://pages.github.com/) or [Netlify](https://www.netlify.com/)). Since there is no downside to starting a blog, and a potentially significant upside, why not have one?
## 2. Ripple Effects
One person starting a blog can trigger other people to do the same. In my case, some of my inspirations include: 
- [Gwern](https://gwern.net/index) (machine learning, metascience, and a ton of other stuff)
- [Progress Good](https://www.arjunkhemani.com/about) (philosophy of science, economics, education)
- [ftlsid](https://ftlsid.com) (learning Japanese, meditation)
- [Visakan Veerasamy](https://visakanv.com) (how social networks and innovation work)
- [Alexey Guzey](https://guzey.com/) (metascience, philosophy)
- [Matt Might](https://matt.might.net/) (theoretical computer science, precision medicine, free software)

My hope is that starting this website will encourage other people to do the same thing, and we can reverse the development of "web feudalism": hosting your entire online presence underneath a larger platform like Substack, Twitter, or Reddit. These platforms aren't bad -- you need them to reach large audiences -- but they should coexist alongside an independent, decentralized web.
## 3. Even Repeated Content can be Useful
Even sites that mostly repeat what other people say can be valuable. My thinking is that as I write more and more, my own ideas will start to emerge without me even trying. Every good idea has already been said, but since nobody was listening the first time around, someone else will have to say it again. 


# How I'm Implementing my Website 
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
Jekyll fixes all these problems -- easy templating, a large plugin ecosystem, the ability to insert HTML into Markdown files (a godsend if you're trying to include graphs / images), you name it. The only price you pay is that you have to use Ruby, and version management can be very frustrating. I almost gave up on making the site because my Homebrew install of Ruby kept conflciting with some dependencies for a custom theme I wanted to use. 

I've found that Jekyll works a lot better when you build up your site from scratch. Premade templates tend to become deprecated, and you run the risk of something mysteriously breaking. For my website, I used the theme [dark-poole](https://andrewhwanpark.github.io/dark-poole/) and adjusted it to my liking. 

(Inspired by [this essay](https://guzey.com/personal/why-have-a-blog/) by Alexey Guzey).	
